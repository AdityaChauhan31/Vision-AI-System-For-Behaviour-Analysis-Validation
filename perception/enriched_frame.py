"""
perception/enriched_frame.py
-----------------------------
EnrichedFrame is the output of Stage 3 and the input to Stage 4.

It extends FrameEvent (from Stage 2) by adding:
  - person_id        → who is in the frame (or "unknown" / None)
  - face_bbox        → bounding box of the detected face
  - face_confidence  → how confident the face match was
  - zone_id          → which zone the person is in (or None)
  - zone_label       → human-readable zone name
  - is_in_restricted_zone → True if zone is marked restricted for this person
  - enrichment_ms    → how long Stage 3 took (for SLI monitoring)

VLMResult is the structured output of Stage 4 — what the VLM returns.
It is validated by Pydantic so a malformed VLM response fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────
# Stage 3 output / Stage 4 input
# ─────────────────────────────────────────────

@dataclass
class FaceDetection:
    """Result of attempting face recognition on one frame."""
    person_id:        Optional[str]    # enrolled ID or None
    person_name:      Optional[str]    # human-readable name or None
    confidence:       float            # 0.0–1.0 similarity score
    bbox:             Optional[tuple]  # (x, y, w, h) pixels or None
    is_known:         bool             # True if matched an enrolled person
    anonymous_label:  str = "unknown"  # label when not recognized

    @property
    def display_id(self) -> str:
        """Returns person_id if known, else the anonymous label."""
        return self.person_id if self.is_known else self.anonymous_label


@dataclass
class ZoneDetection:
    """Result of checking which zone a person is in."""
    zone_id:              Optional[str]   # zone id from config or None
    zone_label:           Optional[str]   # e.g. "restricted_area", "room_A"
    is_inside:            bool            # True if person centroid is inside polygon
    is_restricted:        bool            # True if this person is not allowed here
    polygon_tested:       Optional[list]  # the polygon that was checked


@dataclass
class EnrichedFrame:
    """
    The complete context object that flows from Stage 3 into Stage 4.

    Think of it as FrameEvent + who + where.
    The VLM receives this and uses it to build a contextual prompt.
    """
    # ── From Stage 2 ────────────────────────────────────────────────────────
    feed_id:         str
    use_case:        str
    frame_index:     int
    timestamp_utc:   datetime
    frame_path:      Path
    frame:           np.ndarray          # raw pixel data
    session_id:      str
    source_type:     str
    metadata:        dict = field(default_factory=dict)

    # ── From Stage 3a — Face Recognition ────────────────────────────────────
    face:            Optional[FaceDetection] = None

    # ── From Stage 3b — Zone Check ──────────────────────────────────────────
    zone:            Optional[ZoneDetection] = None

    # ── Stage 3 performance ─────────────────────────────────────────────────
    enrichment_ms:   float = 0.0

    # ── Convenience properties ───────────────────────────────────────────────

    @property
    def person_id(self) -> str:
        """person_id if known, else 'unknown'."""
        if self.face and self.face.is_known:
            return self.face.person_id
        return "unknown"

    @property
    def zone_label(self) -> Optional[str]:
        if self.zone and self.zone.is_inside:
            return self.zone.zone_label
        return None

    @property
    def is_identity_known(self) -> bool:
        return self.face is not None and self.face.is_known

    @property
    def is_in_restricted_zone(self) -> bool:
        return self.zone is not None and self.zone.is_restricted

    @property
    def timestamp_iso(self) -> str:
        return self.timestamp_utc.isoformat()

    def summary(self) -> dict:
        """Compact dict representation — used in VLM context building."""
        return {
            "feed_id":        self.feed_id,
            "frame_index":    self.frame_index,
            "timestamp":      self.timestamp_iso,
            "person_id":      self.person_id,
            "person_name":    self.face.person_name if self.face and self.face.is_known else None,
            "zone_label":     self.zone_label,
            "is_restricted":  self.is_in_restricted_zone,
        }


# ─────────────────────────────────────────────
# Stage 4 output
# ─────────────────────────────────────────────

@dataclass
class VLMResult:
    """
    Validated output of the VLM for one frame.
    Built from the raw JSON the VLM returns, with fallback on parse failure.
    """
    # ── Core VLM outputs ────────────────────────────────────────────────────
    behaviors_detected:             list[str]
    behaviors_confidence:           dict[str, float]
    person_active:                  bool
    estimated_activity_duration_seconds: Optional[int]
    anomaly_detected:               bool
    anomaly_description:            Optional[str]
    reasoning:                      str

    # ── Metadata ────────────────────────────────────────────────────────────
    frame_index:    int
    session_id:     str
    feed_id:        str
    use_case:       str
    person_id:      str
    zone_label:     Optional[str]
    vlm_model:      str
    latency_ms:     float
    parse_success:  bool = True
    raw_response:   str  = ""

    # ── Context threaded through for the rules engine (Stage 6) ──────────────
    timestamp_utc:      Optional[datetime] = None   # capture time of the frame
    frame_path:         Optional[Path]     = None   # snapshot for alert evidence
    identity_known:     bool               = False  # was the person recognized?
    is_restricted_zone: bool               = False  # is the person in a zone they can't be in?

    @classmethod
    def parse_failure(
        cls,
        enriched: EnrichedFrame,
        raw: str,
        vlm_model: str,
        latency_ms: float,
    ) -> "VLMResult":
        """
        Returns a safe empty result when the VLM response can't be parsed.
        The rules engine treats parse failures as 'no information' — not as
        a rule violation, to avoid false alerts from API errors.
        """
        return cls(
            behaviors_detected=[],
            behaviors_confidence={},
            person_active=False,
            estimated_activity_duration_seconds=None,
            anomaly_detected=False,
            anomaly_description=None,
            reasoning="VLM response could not be parsed",
            frame_index=enriched.frame_index,
            session_id=enriched.session_id,
            feed_id=enriched.feed_id,
            use_case=enriched.use_case,
            person_id=enriched.person_id,
            zone_label=enriched.zone_label,
            vlm_model=vlm_model,
            latency_ms=latency_ms,
            parse_success=False,
            raw_response=raw,
            timestamp_utc=enriched.timestamp_utc,
            frame_path=enriched.frame_path,
            identity_known=enriched.is_identity_known,
            is_restricted_zone=enriched.is_in_restricted_zone,
        )

    def to_dict(self) -> dict:
        return {
            "behaviors_detected":              self.behaviors_detected,
            "behaviors_confidence":            self.behaviors_confidence,
            "person_active":                   self.person_active,
            "estimated_activity_duration_seconds": self.estimated_activity_duration_seconds,
            "anomaly_detected":                self.anomaly_detected,
            "anomaly_description":             self.anomaly_description,
            "reasoning":                       self.reasoning,
            "frame_index":                     self.frame_index,
            "session_id":                      self.session_id,
            "feed_id":                         self.feed_id,
            "use_case":                        self.use_case,
            "person_id":                       self.person_id,
            "zone_label":                      self.zone_label,
            "vlm_model":                       self.vlm_model,
            "latency_ms":                      self.latency_ms,
            "parse_success":                   self.parse_success,
            "identity_known":                  self.identity_known,
            "is_restricted_zone":              self.is_restricted_zone,
        }
