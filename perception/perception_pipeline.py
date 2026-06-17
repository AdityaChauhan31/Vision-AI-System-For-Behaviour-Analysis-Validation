"""
perception/perception_pipeline.py
-----------------------------------
Glue between Stage 2 (FrameEvent) and Stage 4 (VLMPipeline).

This class:
  1. Receives a FrameEvent from the ingestion stage
  2. Runs face recognition (Stage 3a) and zone check (Stage 3b) in parallel
  3. Assembles an EnrichedFrame
  4. Passes it to the VLMPipeline (Stage 4)
  5. Returns VLMResult to any registered downstream callbacks

This is the single class you register as a callback on FeedManager:
    manager.add_global_callback(perception_pipeline.process_frame)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from ingestion.frame_sampler import FrameEvent
from .enriched_frame import EnrichedFrame, FaceDetection, VLMResult, ZoneDetection
from .face_recognizer import FaceRecognizer
from .zone_checker import ZoneChecker
from .vlm_pipeline import VLMPipeline, VLMCallback

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """
    Full Stage 3 + Stage 4 pipeline.

    Usage
    -----
        pipeline = PerceptionPipeline.from_config(
            zones_config="config/zones.yaml",
            identities_config="config/identities.yaml",
            use_cases_config="config/use_cases.yaml",
            vlm_provider="mock",
        )
        pipeline.add_result_callback(rules_engine.evaluate)

        # Register as callback on Stage 2
        feed_manager.add_global_callback(pipeline.process_frame)
    """

    def __init__(
        self,
        face_recognizer:  FaceRecognizer,
        zone_checker:     ZoneChecker,
        vlm_pipeline:     VLMPipeline,
        parallel:         bool = True,
    ) -> None:
        self._face  = face_recognizer
        self._zone  = zone_checker
        self._vlm   = vlm_pipeline
        self._parallel = parallel

    @classmethod
    def from_config(
        cls,
        zones_config:      str = "config/zones.yaml",
        identities_config: str = "config/identities.yaml",
        use_cases_config:  str = "config/use_cases.yaml",
        vlm_provider:      Optional[str] = None,
        context_window:    int = 5,
        parallel:          bool = True,
    ) -> "PerceptionPipeline":
        """Convenience constructor — builds all components from config files."""
        face = FaceRecognizer(identities_config)
        zone = ZoneChecker(zones_config)
        vlm  = VLMPipeline(use_cases_config, vlm_provider, context_window)
        return cls(face, zone, vlm, parallel)

    # ── public API ────────────────────────────────────────

    def add_result_callback(self, fn: VLMCallback) -> None:
        """Register a function to receive VLMResult from every frame."""
        self._vlm.add_callback(fn)

    def end_session(self, session_id: str) -> None:
        """Clear per-session VLM context when a feed's session ends."""
        self._vlm.clear_session(session_id)

    def process_frame(self, event: FrameEvent) -> Optional[VLMResult]:
        """
        Main callback for Stage 2 frames.
        Called by FeedManager for every sampled frame.

        Args:
            event: FrameEvent from the ingestion stage

        Returns:
            VLMResult, or None if something prevented VLM from running.
        """
        t0 = time.monotonic()

        # ── Stage 3: Face + Zone (parallel) ─────────────────────────────────
        if self._parallel:
            face_result, zone_result = self._run_stage3_parallel(event)
        else:
            face_result = self._run_face_recognition(event)
            zone_result = self._run_zone_check(event, face_result)

        enrichment_ms = (time.monotonic() - t0) * 1000

        # ── Assemble EnrichedFrame ────────────────────────────────────────────
        enriched = EnrichedFrame(
            feed_id       = event.feed_id,
            use_case      = event.use_case,
            frame_index   = event.frame_index,
            timestamp_utc = event.timestamp_utc,
            frame_path    = event.frame_path,
            frame         = event.frame,
            session_id    = event.session_id,
            source_type   = event.source_type,
            metadata      = event.metadata,
            face          = face_result,
            zone          = zone_result,
            enrichment_ms = enrichment_ms,
        )

        logger.debug(
            "[%s] Stage 3 done in %.0fms | person=%s | zone=%s",
            event.feed_id,
            enrichment_ms,
            enriched.person_id,
            enriched.zone_label or "none",
        )

        # ── Stage 4: VLM ─────────────────────────────────────────────────────
        return self._vlm.process(enriched)

    # ── internals ─────────────────────────────────────────

    def _run_stage3_parallel(
        self, event: FrameEvent
    ) -> tuple[FaceDetection, ZoneDetection]:
        """
        Run face recognition and zone check concurrently.
        Both are I/O-bound (face: model inference, zone: math),
        so threading gives a real speedup.
        """
        face_result: Optional[FaceDetection] = None
        zone_result: Optional[ZoneDetection] = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            face_future = executor.submit(self._run_face_recognition, event)
            # We need face result for zone check, but zone check also works
            # without it (uses frame center as fallback). Start both immediately.
            zone_future = executor.submit(
                self._zone.check,
                event.frame,
                None,      # face not available yet — zone uses frame center
                event.feed_id,
            )

            face_result = face_future.result()
            zone_result = zone_future.result()

        # If face was recognized, re-check zone with actual person config
        if face_result and face_result.is_known:
            person = self._face.get_person(face_result.person_id)
            if person:
                zone_result = self._zone.check(
                    event.frame,
                    face_result,
                    event.feed_id,
                    allowed_zones=person.allowed_zones,
                    restricted_zones=person.restricted_zones,
                )

        return face_result, zone_result

    def _run_face_recognition(self, event: FrameEvent) -> FaceDetection:
        try:
            return self._face.recognize(event.frame)
        except Exception as exc:
            logger.exception("Face recognition crashed unexpectedly: %s", exc)
            return FaceDetection(
                person_id=None, person_name=None,
                confidence=0.0, bbox=None, is_known=False,
            )

    def _run_zone_check(
        self,
        event: FrameEvent,
        face: Optional[FaceDetection],
    ) -> ZoneDetection:
        try:
            allowed   = []
            restricted = []
            if face and face.is_known:
                person = self._face.get_person(face.person_id)
                if person:
                    allowed    = person.allowed_zones
                    restricted = person.restricted_zones

            return self._zone.check(
                event.frame, face, event.feed_id,
                allowed_zones=allowed,
                restricted_zones=restricted,
            )
        except Exception as exc:
            logger.exception("Zone check crashed unexpectedly: %s", exc)
            return ZoneDetection(
                zone_id=None, zone_label=None,
                is_inside=False, is_restricted=False,
                polygon_tested=None,
            )
