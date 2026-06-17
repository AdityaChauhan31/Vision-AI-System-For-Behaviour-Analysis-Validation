"""
perception/vlm/base.py
-----------------------
Abstract base class for all VLM adapters.

The system supports multiple VLM providers (OpenAI, Gemini, Anthropic, Mock).
Which one runs is controlled entirely by environment variables:
  VLM_PROVIDER=openai    → uses OpenAI GPT-4o
  VLM_PROVIDER=gemini    → uses Google Gemini 1.5 Pro
  VLM_PROVIDER=anthropic → uses Claude claude-sonnet-4-6
  VLM_PROVIDER=mock      → deterministic fake responses (testing, no API key needed)

All adapters:
  - Accept the same EnrichedFrame input
  - Return the same VLMResult output
  - Handle their own API errors and retry logic
  - Never let exceptions propagate to the caller
    (they return VLMResult.parse_failure() instead)
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from ..enriched_frame import EnrichedFrame, VLMResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────

class BaseVLMAdapter(ABC):
    """
    All VLM adapters inherit from this class.
    Subclasses implement only `_call_api()`.
    Everything else (prompt building, JSON parsing, retry) is shared here.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0    # base seconds between retries
    VLM_MAX_EDGE = 768   # downscale frames so the long edge is at most this many px
                         # before sending to the VLM — cuts image tokens ~60-75%
                         # (the model doesn't need full HD to recognise behaviours)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier, e.g. 'gpt-4o'."""
        ...

    @abstractmethod
    def _call_api(self, prompt_messages: list[dict], image_b64: str) -> str:
        """
        Make the actual API call. Returns raw text response from the model.
        Raises on API error (caller handles retry).
        """
        ...

    # ── main entry point (called by vlm_pipeline.py) ──────

    def analyze(
        self,
        enriched: EnrichedFrame,
        use_case_config: dict,
        session_context: list[dict],
    ) -> VLMResult:
        """
        Analyze one frame and return a structured VLMResult.

        Args:
            enriched:        The EnrichedFrame from Stage 3
            use_case_config: The use case definition from use_cases.yaml
            session_context: List of previous VLMResult.to_dict() for this session
                             (last N frames — used to build context in prompt)
        """
        t0 = time.monotonic()

        # Step 1: encode frame to base64
        image_b64 = self._encode_frame(enriched.frame)

        # Step 2: build the prompt
        messages = self._build_prompt(enriched, use_case_config, session_context)

        # Step 3: call API with retry
        raw_response = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw_response = self._call_api(messages, image_b64)
                break
            except Exception as exc:
                logger.warning(
                    "[%s] API call attempt %d/%d failed: %s",
                    self.model_name, attempt, self.MAX_RETRIES, exc,
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(self._retry_wait(exc, attempt))
                else:
                    latency_ms = (time.monotonic() - t0) * 1000
                    return VLMResult.parse_failure(enriched, str(exc), self.model_name, latency_ms)

        latency_ms = (time.monotonic() - t0) * 1000
        logger.debug("[%s] API response in %.0fms", self.model_name, latency_ms)

        # Step 4: parse JSON from response
        return self._parse_response(raw_response, enriched, latency_ms)

    # ── prompt building ────────────────────────────────────

    def _build_prompt(
        self,
        enriched: EnrichedFrame,
        use_case_config: dict,
        session_context: list[dict],
    ) -> list[dict]:
        """
        Constructs the system + user messages sent to the VLM.
        This is the most important method — prompt engineering lives here.
        """
        system_prompt = self._system_prompt()
        user_content  = self._user_prompt(enriched, use_case_config, session_context)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

    def _system_prompt(self) -> str:
        return """You are a behavior analysis AI for a surveillance and compliance system.
You analyze video frames to understand what humans are doing.
Your output is used by an automated rules engine — it MUST be valid JSON.

CRITICAL RULES:
- Return ONLY a JSON object. No markdown, no explanation, no preamble.
- Never wrap your response in ```json ``` code blocks.
- If you are uncertain about a behavior, set its confidence to a low value (0.1–0.4).
- Do not invent behaviors not in the provided list.
- Base your reasoning only on what is visible in THIS frame plus the provided context.
- person_active means a person is visibly performing an action (not just present)."""

    def _user_prompt(
        self,
        enriched: EnrichedFrame,
        use_case_config: dict,
        session_context: list[dict],
    ) -> list[dict]:
        """
        Returns the user message as a list of content blocks
        (text + image_url) for multi-modal VLMs.
        """
        behaviors = use_case_config.get("behaviors_to_detect", [])
        definitions = use_case_config.get("behavior_definitions", {})
        required = use_case_config.get("required_behaviors", [])

        # Build behavior list with definitions
        behavior_lines = []
        for b in behaviors:
            defn = definitions.get(b, "")
            line = f"  - {b}" + (f": {defn}" if defn else "")
            behavior_lines.append(line)
        behavior_text = "\n".join(behavior_lines)

        # Build session context summary
        context_text = "No previous frames analyzed yet."
        if session_context:
            prev_behaviors = []
            for ctx in session_context[-5:]:   # last 5 frames
                for b in ctx.get("behaviors_detected", []):
                    if b not in prev_behaviors:
                        prev_behaviors.append(b)
            if prev_behaviors:
                context_text = (
                    f"This is frame {enriched.frame_index} of this session.\n"
                    f"Behaviors detected in previous frames: {prev_behaviors}\n"
                    f"Required behaviors not yet seen: "
                    f"{[b for b in required if b not in prev_behaviors]}"
                )

        # Person and zone info
        person_line = (
            f"Person: {enriched.face.person_name} (ID: {enriched.person_id})"
            if enriched.is_identity_known
            else "Person: Unknown (anonymous mode)"
        )
        zone_line = (
            f"Zone: {enriched.zone_label}"
            if enriched.zone_label
            else "Zone: Not determined"
        )

        text_block = f"""
USE CASE: {use_case_config.get('name', enriched.use_case)}
DESCRIPTION: {use_case_config.get('description', '')}

{person_line}
{zone_line}
Frame index: {enriched.frame_index}
Timestamp: {enriched.timestamp_iso}

BEHAVIORS TO DETECT (choose ONLY from this list):
{behavior_text}

REQUIRED BEHAVIORS (must all occur for compliance): {required}

SESSION CONTEXT:
{context_text}

TASK:
Analyze what is visible in this image and return a JSON object with EXACTLY these fields:

{{
  "behaviors_detected": ["list of behavior ids visible in THIS frame"],
  "behaviors_confidence": {{"behavior_id": 0.0_to_1.0}},
  "person_active": true_or_false,
  "estimated_activity_duration_seconds": integer_or_null,
  "anomaly_detected": true_or_false,
  "anomaly_description": "string or null",
  "reasoning": "One sentence describing what you observe in the frame"
}}

Return ONLY the JSON. Nothing else.""".strip()

        return [
            {"type": "text",      "text": text_block},
            {"type": "image_url", "image_url": {"url": "PLACEHOLDER_IMAGE"}},
        ]

    # ── response parsing ───────────────────────────────────

    def _parse_response(
        self,
        raw: str,
        enriched: EnrichedFrame,
        latency_ms: float,
    ) -> VLMResult:
        """
        Parse the VLM's raw text response into a VLMResult.
        Handles common failure modes:
          - JSON wrapped in markdown code fences
          - Trailing commas (invalid JSON)
          - Extra text before/after the JSON object
        """
        cleaned = raw.strip()

        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$",         "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        # Extract just the JSON object if there's extra text around it
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = self._repair_json(cleaned)
            if repaired is None:
                logger.warning("[%s] JSON parse failed (unrepairable)\nRaw: %s",
                               self.model_name, raw[:300])
                return VLMResult.parse_failure(enriched, raw, self.model_name, latency_ms)
            data = repaired

        # Validate required fields
        try:
            result = VLMResult(
                behaviors_detected   = data.get("behaviors_detected", []),
                behaviors_confidence = data.get("behaviors_confidence", {}),
                person_active        = bool(data.get("person_active", False)),
                estimated_activity_duration_seconds = data.get("estimated_activity_duration_seconds"),
                anomaly_detected     = bool(data.get("anomaly_detected", False)),
                anomaly_description  = data.get("anomaly_description"),
                reasoning            = str(data.get("reasoning", "")),
                frame_index  = enriched.frame_index,
                session_id   = enriched.session_id,
                feed_id      = enriched.feed_id,
                use_case     = enriched.use_case,
                person_id    = enriched.person_id,
                zone_label   = enriched.zone_label,
                vlm_model    = self.model_name,
                latency_ms   = latency_ms,
                parse_success = True,
                raw_response  = raw,
                timestamp_utc = enriched.timestamp_utc,
                frame_path    = enriched.frame_path,
                identity_known = enriched.is_identity_known,
                is_restricted_zone = enriched.is_in_restricted_zone,
            )
            return result
        except Exception as exc:
            logger.warning("[%s] VLMResult construction failed: %s", self.model_name, exc)
            return VLMResult.parse_failure(enriched, raw, self.model_name, latency_ms)

    # ── utilities ──────────────────────────────────────────

    @staticmethod
    def _repair_json(s: str) -> Optional[dict]:
        """
        Best-effort recovery for JSON truncated by a token limit. Strips a
        trailing comma and appends the closing brackets/braces needed to balance
        depth (ignoring brackets inside strings). Returns a dict or None.
        Only salvages the common 'cut after a complete value' case; anything
        cut mid-string stays unrepairable and falls through to parse_failure.
        """
        depth_stack: list[str] = []
        in_str = False
        escape = False
        for ch in s:
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                depth_stack.append(ch)
            elif ch in "}]":
                if depth_stack:
                    depth_stack.pop()
        if in_str:
            return None  # cut inside a string — can't safely repair
        candidate = s.rstrip().rstrip(",")
        for opener in reversed(depth_stack):
            candidate += "}" if opener == "{" else "]"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    @classmethod
    def _encode_frame(cls, frame: np.ndarray) -> str:
        """Downscale (long edge ≤ VLM_MAX_EDGE) then encode to base64 JPEG."""
        h, w = frame.shape[:2]
        long_edge = max(h, w)
        if long_edge > cls.VLM_MAX_EDGE:
            scale = cls.VLM_MAX_EDGE / long_edge
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("Failed to encode frame to JPEG")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
