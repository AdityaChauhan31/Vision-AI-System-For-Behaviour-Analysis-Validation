"""
perception/vlm_pipeline.py
---------------------------
Stage 4 — VLM Perception Pipeline

Orchestrates the full Stage 3 → Stage 4 flow:
  1. Receives an EnrichedFrame (from perception_pipeline.py)
  2. Retrieves session context (what was seen in previous frames)
  3. Loads use case config (behaviors to detect, definitions, rules)
  4. Calls the VLM adapter → gets VLMResult
  5. Updates session context for the next frame
  6. Returns VLMResult to the caller (rules engine in Stage 6)

The VLM adapter is selected at startup via environment variable:
    VLM_PROVIDER=mock      (default — no API key needed)
    VLM_PROVIDER=openai
    VLM_PROVIDER=gemini
    VLM_PROVIDER=anthropic
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Optional

import yaml

from .enriched_frame import EnrichedFrame, VLMResult
from .vlm.base import BaseVLMAdapter

logger = logging.getLogger(__name__)

# Callback type for downstream consumers (rules engine)
VLMCallback = Callable[[VLMResult], None]


class VLMPipeline:
    """
    End-to-end Stage 4 processor.

    Usage
    -----
        pipeline = VLMPipeline(
            use_cases_config="config/use_cases.yaml",
            provider="mock",       # or "openai", "gemini", "anthropic"
        )
        pipeline.add_callback(rules_engine.evaluate)
        pipeline.process(enriched_frame)   # called by perception_pipeline
    """

    def __init__(
        self,
        use_cases_config: str = "config/use_cases.yaml",
        provider: Optional[str] = None,
        context_window: int = 5,
    ) -> None:
        """
        Args:
            use_cases_config: Path to use_cases.yaml
            provider:         VLM provider override. If None, reads VLM_PROVIDER env var.
                              Defaults to 'mock' if neither is set.
            context_window:   Number of past VLMResults to include in each prompt.
        """
        self._use_cases: dict[str, dict]  = {}
        self._callbacks: list[VLMCallback] = []
        self._adapter: BaseVLMAdapter
        self._context: dict[str, deque]   = defaultdict(lambda: deque(maxlen=context_window))

        self._load_use_cases(use_cases_config)
        self._adapter = self._build_adapter(provider)

    # ── public API ────────────────────────────────────────

    def add_callback(self, fn: VLMCallback) -> None:
        """Register a function to receive every VLMResult (e.g. rules engine)."""
        self._callbacks.append(fn)

    def process(self, enriched: EnrichedFrame) -> VLMResult:
        """
        Main entry point — called with each EnrichedFrame from Stage 3.
        Returns VLMResult and dispatches it to all registered callbacks.
        """
        use_case_config = self._use_cases.get(enriched.use_case)
        if use_case_config is None:
            logger.warning(
                "No use case config found for '%s' — using empty config", enriched.use_case
            )
            use_case_config = {"id": enriched.use_case, "behaviors_to_detect": []}

        # Build context from previous frames in this session
        session_context = list(self._context[enriched.session_id])

        logger.debug(
            "[%s] Analyzing frame %d | use_case=%s | person=%s | zone=%s",
            enriched.feed_id,
            enriched.frame_index,
            enriched.use_case,
            enriched.person_id,
            enriched.zone_label or "none",
        )

        # Call VLM
        result = self._adapter.analyze(enriched, use_case_config, session_context)

        # Log result
        if result.parse_success:
            logger.info(
                "[%s] Frame %d → behaviors=%s | anomaly=%s | latency=%.0fms",
                enriched.feed_id,
                result.frame_index,
                result.behaviors_detected,
                result.anomaly_detected,
                result.latency_ms,
            )
        else:
            logger.warning(
                "[%s] Frame %d → VLM parse failed (%.0fms)",
                enriched.feed_id,
                result.frame_index,
                result.latency_ms,
            )

        # Update session context for next frame
        self._context[enriched.session_id].append(result.to_dict())

        # Dispatch to registered callbacks (rules engine etc.)
        self._dispatch(result)

        return result

    def clear_session(self, session_id: str) -> None:
        """Clear the session context when a session ends."""
        if session_id in self._context:
            del self._context[session_id]
            logger.debug("Cleared context for session %s", session_id)

    # ── internals ─────────────────────────────────────────

    def _load_use_cases(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            logger.warning("use_cases.yaml not found at %s", path)
            return

        with path.open() as f:
            raw = yaml.safe_load(f)

        for uc in raw.get("use_cases", []):
            self._use_cases[uc["id"]] = uc

        logger.info("Loaded %d use case(s): %s", len(self._use_cases), list(self._use_cases))

    def _build_adapter(self, provider: Optional[str]) -> BaseVLMAdapter:
        """
        Instantiate the right VLM adapter.
        Priority: explicit argument > VLM_PROVIDER env var > default 'mock'
        """
        resolved = provider or os.environ.get("VLM_PROVIDER", "mock")
        logger.info("VLM provider: %s", resolved)

        if resolved == "openai":
            from .vlm.openai_adapter import OpenAIVLMAdapter
            return OpenAIVLMAdapter(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o")
            )
        elif resolved == "gemini":
            from .vlm.adapters import GeminiVLMAdapter
            return GeminiVLMAdapter(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            )
        elif resolved == "groq":
            from .vlm.groq_adapter import GroqVLMAdapter
            return GroqVLMAdapter(
                model=os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
            )
        elif resolved == "huggingface":
            from .vlm.huggingface_adapter import HuggingFaceVLMAdapter
            return HuggingFaceVLMAdapter(
                model=os.environ.get(
                    "HUGGINGFACE_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct"
                )
            )
        elif resolved == "anthropic":
            from .vlm.adapters import AnthropicVLMAdapter
            return AnthropicVLMAdapter(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            )
        elif resolved == "mock":
            from .vlm.adapters import MockVLMAdapter
            return MockVLMAdapter(
                simulated_latency_ms=float(os.environ.get("MOCK_LATENCY_MS", "300")),
                anomaly_rate=float(os.environ.get("MOCK_ANOMALY_RATE", "0.1")),
            )
        else:
            raise ValueError(
                f"Unknown VLM provider: '{resolved}'. "
                f"Valid options: mock, gemini, groq, openai, anthropic, huggingface"
            )

    def _dispatch(self, result: VLMResult) -> None:
        for fn in self._callbacks:
            try:
                fn(result)
            except Exception as exc:
                logger.exception("VLM callback %s raised: %s", fn.__name__, exc)
