"""
perception/vlm/huggingface_adapter.py
---------------------------------------
FREE open-source Vision LLM via HuggingFace Inference API.

WHY HUGGINGFACE:
  - Free tier with rate limits (no credit card needed)
  - Runs open-source models (Llama, Phi, InternVL, etc.)
  - Same HF token works for all models
  - No per-token billing on free tier

GET YOUR FREE TOKEN:
  1. Go to https://huggingface.co  (create free account)
  2. Go to https://huggingface.co/settings/tokens
  3. Click "New token" → name it "vision-ai" → Role: Read
  4. Copy token → paste into .env as HUGGINGFACE_API_KEY=hf_...

FREE MODELS THAT SUPPORT VISION (pick one in .env):
  ┌──────────────────────────────────────────────────────────────┐
  │ meta-llama/Llama-3.2-11B-Vision-Instruct  ← RECOMMENDED    │
  │   Best quality, 11B params, strong instruction following     │
  │                                                              │
  │ microsoft/Phi-3.5-vision-instruct                           │
  │   Smaller, faster, good for structured JSON output          │
  │                                                              │
  │ HuggingFaceM4/idefics2-8b                                   │
  │   Strong on image + text reasoning                          │
  └──────────────────────────────────────────────────────────────┘

RATE LIMITS (free tier):
  - ~1000 requests/day
  - Max image size: 5MB (we send compressed JPEG ~100-300KB — fine)
  - Timeout: 30s per request
  - If overloaded: returns 503 → we retry automatically

USAGE:
  # In .env:
  VLM_PROVIDER=huggingface
  HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxx
  HUGGINGFACE_MODEL=meta-llama/Llama-3.2-11B-Vision-Instruct

  # Then just run:
  python main.py --video data/sample.mp4
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import requests

from .base import BaseVLMAdapter

logger = logging.getLogger(__name__)

# HuggingFace Inference API endpoint
HF_API_BASE = "https://api-inference.huggingface.co/models"


class HuggingFaceVLMAdapter(BaseVLMAdapter):
    """
    Calls HuggingFace Inference API with a vision-capable open-source model.
    Uses the chat completions compatible endpoint for newer models.
    Falls back to the legacy inference endpoint for older models.
    """

    # Time to wait when HF says model is loading (cold start)
    MODEL_LOADING_WAIT = 20

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        from config.settings import settings

        self._model   = model   or settings.huggingface_model
        self._api_key = api_key or settings.huggingface_api_key
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        })

        logger.info("HuggingFace adapter ready (model=%s)", self._model)

    @property
    def model_name(self) -> str:
        return self._model

    def _call_api(self, prompt_messages: list[dict], image_b64: str) -> str:
        """
        Try the chat-completions endpoint first (newer models like Llama 3.2).
        If that fails with 404, fall back to legacy text-generation endpoint.
        """
        # First attempt: OpenAI-compatible chat endpoint
        try:
            return self._call_chat_endpoint(prompt_messages, image_b64)
        except _EndpointNotFound:
            logger.debug("Chat endpoint not available for %s, trying legacy...", self._model)
            return self._call_legacy_endpoint(prompt_messages, image_b64)

    # ── Chat completions endpoint (Llama 3.2, Phi-3.5) ────────────────────────

    def _call_chat_endpoint(self, prompt_messages: list[dict], image_b64: str) -> str:
        """
        OpenAI-compatible endpoint: /v1/chat/completions
        Supported by: Llama 3.2 Vision, Phi-3.5-vision, InternVL2
        """
        # NOTE: the old free serverless endpoint (api-inference.huggingface.co/models/...)
        # was retired in favour of "Inference Providers". This router is OpenAI-compatible
        # but routes to paid partners (Novita/SambaNova/etc.) and needs HF credits.
        # For a genuinely-free VLM use --vlm gemini instead.
        url = "https://router.huggingface.co/v1/chat/completions"

        # Build messages with inline base64 image
        messages = self._build_hf_messages(prompt_messages, image_b64)

        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.1,
            "stream": False,
        }

        response = self._post(url, payload)
        data     = response.json()

        if "choices" not in data:
            raise ValueError(f"Unexpected response format: {list(data.keys())}")

        return data["choices"][0]["message"]["content"]

    # ── Legacy text-generation endpoint ───────────────────────────────────────

    def _call_legacy_endpoint(self, prompt_messages: list[dict], image_b64: str) -> str:
        """
        Legacy endpoint for older HF models.
        Flattens messages into a single prompt string.
        """
        url = f"{HF_API_BASE}/{self._model}"

        # Flatten all message text into one prompt
        prompt_parts = []
        for msg in prompt_messages:
            role = msg["role"].upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        prompt_parts.append(f"[{role}]: {block['text']}")
            elif isinstance(content, str):
                prompt_parts.append(f"[{role}]: {content}")

        prompt_text = "\n\n".join(prompt_parts)
        prompt_text += "\n\n[ASSISTANT]:"

        payload = {
            "inputs": prompt_text,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.1,
                "return_full_text": False,
            },
        }

        response = self._post(url, payload)
        data     = response.json()

        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]

        raise ValueError(f"Unexpected legacy response: {str(data)[:200]}")

    # ── Shared request helper ──────────────────────────────────────────────────

    def _post(self, url: str, payload: dict) -> requests.Response:
        """
        POST to HF API with automatic handling of:
          - 503 model loading (waits and retries)
          - 404 endpoint not found (raises _EndpointNotFound)
          - 429 rate limit (waits and retries)
          - Other errors (raises)
        """
        for attempt in range(3):
            resp = self._session.post(url, json=payload, timeout=self._timeout)

            if resp.status_code == 200:
                return resp

            elif resp.status_code == 404:
                raise _EndpointNotFound(f"404 at {url}")

            elif resp.status_code == 503:
                # Model is loading on HF servers (cold start) — wait and retry
                try:
                    wait = resp.json().get("estimated_time", self.MODEL_LOADING_WAIT)
                except Exception:
                    wait = self.MODEL_LOADING_WAIT
                logger.warning(
                    "HuggingFace model loading... waiting %.0fs (attempt %d/3)",
                    wait, attempt + 1,
                )
                time.sleep(min(float(wait), 30))

            elif resp.status_code == 429:
                # Rate limit — back off
                logger.warning("HuggingFace rate limit hit. Waiting 10s...")
                time.sleep(10)

            else:
                # Unexpected error
                raise RuntimeError(
                    f"HuggingFace API error {resp.status_code}: {resp.text[:300]}"
                )

        raise RuntimeError("HuggingFace API: max retries exceeded")

    # ── Message builder ────────────────────────────────────────────────────────

    def _build_hf_messages(
        self,
        prompt_messages: list[dict],
        image_b64: str,
    ) -> list[dict]:
        """
        Convert our internal message format to HF chat format with inline image.
        HF uses the same format as OpenAI for vision: image_url with data URI.
        """
        messages = []
        for msg in prompt_messages:
            role    = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                # Some HF models don't support system role — merge into first user message
                messages.append({
                    "role": "system",
                    "content": content if isinstance(content, str) else str(content),
                })
                continue

            if isinstance(content, list):
                # Multi-modal user message — inject real image
                new_content = []
                for block in content:
                    if block.get("type") == "text":
                        new_content.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image_url":
                        new_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        })
                messages.append({"role": role, "content": new_content})
            else:
                messages.append({"role": role, "content": content})

        return messages


class _EndpointNotFound(Exception):
    """Raised when the chat-completions endpoint returns 404."""
    pass
