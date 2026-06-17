"""
perception/vlm/groq_adapter.py
-------------------------------
FREE, higher-limit VLM via Groq — recommended when Gemini's free quota is too
tight. Groq is OpenAI-compatible and serves vision through Llama 4 Scout.

WHY GROQ:
  - Free tier: 30 requests/min, 1000 requests/day (6x Gemini Flash's per-minute)
  - No credit card
  - Very fast (LPU hardware) — sub-second inference
  - Vision-capable: meta-llama/llama-4-scout-17b-16e-instruct

GET A FREE KEY:
  1. https://console.groq.com  (sign in with Google/GitHub, no card)
  2. https://console.groq.com/keys → Create API Key
  3. Paste into .env as GROQ_API_KEY=gsk_...

USAGE (.env):
  VLM_PROVIDER=groq
  GROQ_API_KEY=gsk_xxxxxxxx
  GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from .base import BaseVLMAdapter
from .openai_adapter import _inject_image

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqVLMAdapter(BaseVLMAdapter):
    """Calls Groq's OpenAI-compatible chat endpoint with a vision model."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: int = 60) -> None:
        from config.settings import settings
        self._model   = model   or settings.groq_model
        self._api_key = api_key or settings.groq_api_key
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        })
        logger.info("Groq adapter ready (model=%s)", self._model)

    @property
    def model_name(self) -> str:
        return self._model

    def _call_api(self, prompt_messages: list[dict], image_b64: str) -> str:
        # Groq uses the OpenAI message/vision format (image_url with data URI).
        messages = _inject_image(prompt_messages, image_b64, "openai")
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.1,
            # Force strict JSON. If a given model rejects this, we retry without it.
            "response_format": {"type": "json_object"},
        }

        resp = self._session.post(GROQ_URL, json=payload, timeout=self._timeout)

        # Some vision models don't accept response_format — retry plain.
        if resp.status_code == 400 and "response_format" in resp.text:
            payload.pop("response_format", None)
            resp = self._session.post(GROQ_URL, json=payload, timeout=self._timeout)

        if resp.status_code == 429:
            raise RuntimeError(f"Groq rate limit (429): {resp.text[:200]}")
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        return data["choices"][0]["message"]["content"]
