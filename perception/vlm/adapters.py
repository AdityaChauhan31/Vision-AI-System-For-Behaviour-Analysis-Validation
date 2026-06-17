"""
perception/vlm/gemini_adapter.py  (also contains AnthropicAdapter + MockAdapter)
----------------------------------------------------------------------------------
Three more VLM adapters in one file:
  - GeminiVLMAdapter    → Google Gemini 1.5 Pro / Flash
  - AnthropicVLMAdapter → Claude claude-sonnet-4-6 (via Anthropic API)
  - MockVLMAdapter      → Returns realistic fake JSON — NO API KEY NEEDED
                          Use this for development and testing.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Optional

from .base import BaseVLMAdapter
from .openai_adapter import _inject_image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Gemini Adapter
# ─────────────────────────────────────────────

class GeminiVLMAdapter(BaseVLMAdapter):
    """
    Google Gemini via google-generativeai SDK.

    FREE tier available — no credit card needed.
    Get key: https://aistudio.google.com/app/apikey

    Requires:
        pip install google-generativeai
        GEMINI_API_KEY=... in .env   (loaded automatically)

    Model options (set GEMINI_MODEL in .env):
        gemini-2.0-flash          ← fast, free tier, RECOMMENDED
        gemini-2.0-flash-lite     ← fastest / cheapest
        gemini-1.5-flash-latest   ← previous gen fallback
        gemini-1.5-pro-latest     ← highest quality (lower rate limit)
    """

    def __init__(self, model: str = None) -> None:
        from config.settings import settings
        self._model_id = model or settings.gemini_model
        self._client   = None
        self._setup()

    @property
    def model_name(self) -> str:
        return self._model_id

    def _setup(self) -> None:
        try:
            import google.generativeai as genai
            from config.settings import settings
            genai.configure(api_key=settings.gemini_api_key)
            self._genai = genai
            self._client = genai.GenerativeModel(self._model_id)
            logger.info("Gemini adapter ready (model=%s)", self._model_id)
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed.\n"
                "Run: pip install google-generativeai"
            )

    def _call_api(self, prompt_messages: list[dict], image_b64: str) -> str:
        import base64
        # Gemini uses a different API structure — flatten messages to parts
        system_text = next(
            (m["content"] for m in prompt_messages if m["role"] == "system"), ""
        )
        user_msg = next(
            (m for m in prompt_messages if m["role"] == "user"), None
        )
        text_part = ""
        if user_msg:
            for block in user_msg["content"]:
                if block.get("type") == "text":
                    text_part = block["text"]

        image_bytes = base64.b64decode(image_b64)
        image_part  = {"mime_type": "image/jpeg", "data": image_bytes}

        full_prompt = f"{system_text}\n\n{text_part}"

        response = self._client.generate_content(
            [full_prompt, image_part],
            generation_config=self._genai.GenerationConfig(
                temperature=0.1,
                # gemini-2.5-* are "thinking" models: internal reasoning consumes
                # output tokens. 1000 was too low and truncated the JSON. Give
                # headroom, and force application/json so there are no ``` fences.
                max_output_tokens=4096,
                response_mime_type="application/json",
            ),
        )
        return response.text


# ─────────────────────────────────────────────
# Anthropic Adapter
# ─────────────────────────────────────────────

class AnthropicVLMAdapter(BaseVLMAdapter):
    """
    Anthropic Claude claude-sonnet-4-6.

    Requires:
        pip install anthropic
        ANTHROPIC_API_KEY=... (in .env or environment)
    """

    def __init__(self, model: str = None) -> None:
        from config.settings import settings
        self._model_id = model or settings.anthropic_model
        self._client   = None
        self._setup()

    @property
    def model_name(self) -> str:
        return self._model_id

    def _setup(self) -> None:
        try:
            import anthropic
            from config.settings import settings
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            logger.info("Anthropic adapter ready (model=%s)", self._model_id)
        except ImportError:
            raise ImportError(
                "anthropic package not installed.\n"
                "Run: pip install anthropic"
            )

    def _call_api(self, prompt_messages: list[dict], image_b64: str) -> str:
        # Inject image into messages
        messages = _inject_image(prompt_messages, image_b64, "anthropic")

        # Separate system prompt (Anthropic API uses system as top-level param)
        system_text = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user_messages = [m for m in messages if m["role"] != "system"]

        response = self._client.messages.create(
            model=self._model_id,
            max_tokens=1000,
            system=system_text,
            messages=user_messages,
        )
        return response.content[0].text


# ─────────────────────────────────────────────
# Mock Adapter — for testing without API keys
# ─────────────────────────────────────────────

class MockVLMAdapter(BaseVLMAdapter):
    """
    Returns realistic deterministic-ish fake VLM responses.
    No API key required. Used for:
      - Development without API access
      - Unit and integration testing
      - CI/CD pipelines

    Behavior:
      - Simulates housekeeping behaviors progressing over frames
      - Occasionally detects anomalies to test alert pipeline
      - Adds realistic latency (configurable)
    """

    # Predefined behavior sequences per use case
    BEHAVIOR_SEQUENCES = {
        "housekeeping_validation": [
            ["mopping_floor"],
            ["mopping_floor", "wiping_surfaces"],
            ["wiping_surfaces"],
            ["changing_linen"],
            ["changing_linen", "arranging_items"],
            ["emptying_bins"],
            ["arranging_items"],
        ],
        "housekeeping_demo_short": [
            ["mopping_floor"],
            ["mopping_floor", "wiping_surfaces"],
            ["wiping_surfaces"],
            ["arranging_items"],
        ],
        "loitering_detection": [
            ["walking_through"],
            ["standing_idle"],
            ["standing_idle", "using_phone"],
            ["pacing"],
            ["pacing", "looking_around_suspiciously"],
        ],
        "identity_restriction": [
            ["present_in_zone"],
            ["entering_zone"],
            ["exiting_zone"],
            ["attempting_access"],
        ],
    }

    def __init__(
        self,
        simulated_latency_ms: float = 800,
        anomaly_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        self._latency    = simulated_latency_ms / 1000
        self._anomaly_rate = anomaly_rate
        self._rng        = random.Random(seed)
        self._frame_counter: dict[str, int] = {}
        logger.info("MockVLMAdapter ready (latency=%.0fms, anomaly_rate=%.0f%%)",
                    simulated_latency_ms, anomaly_rate * 100)

    @property
    def model_name(self) -> str:
        return "mock-vlm-v1"

    def _call_api(self, prompt_messages: list[dict], image_b64: str) -> str:
        """Simulate API latency and return fake JSON."""
        time.sleep(self._latency + self._rng.uniform(-0.1, 0.1))

        # Extract use_case from the prompt text
        use_case = "housekeeping_validation"   # default
        for msg in prompt_messages:
            if msg["role"] == "user":
                content = msg.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        for uc in self.BEHAVIOR_SEQUENCES:
                            if uc in text:
                                use_case = uc
                                break

        # Pick behavior sequence based on call count
        seq_key  = use_case
        call_idx = self._frame_counter.get(seq_key, 0)
        self._frame_counter[seq_key] = call_idx + 1

        sequences   = self.BEHAVIOR_SEQUENCES.get(use_case, [["idle_standing"]])
        behaviors   = sequences[call_idx % len(sequences)]

        # Random confidence between 0.72 and 0.97
        confidences = {b: round(self._rng.uniform(0.72, 0.97), 2) for b in behaviors}

        # Occasionally trigger an anomaly
        is_anomaly    = self._rng.random() < self._anomaly_rate
        anomaly_desc  = (
            "Person appears to have skipped required cleaning step" if is_anomaly else None
        )

        duration = self._rng.randint(20, 90)
        reasoning_map = {
            "housekeeping_validation": f"Staff member is visibly performing {', '.join(behaviors)} in the room",
            "loitering_detection":     f"Person is {behaviors[0].replace('_', ' ')} in the monitored area",
            "identity_restriction":    f"Individual is {behaviors[0].replace('_', ' ')} in defined zone",
        }

        response_data = {
            "behaviors_detected": behaviors,
            "behaviors_confidence": confidences,
            "person_active": len(behaviors) > 0 and behaviors[0] != "idle_standing",
            "estimated_activity_duration_seconds": duration,
            "anomaly_detected": is_anomaly,
            "anomaly_description": anomaly_desc,
            "reasoning": reasoning_map.get(use_case, "Behavior observed in frame"),
        }
        return json.dumps(response_data)
