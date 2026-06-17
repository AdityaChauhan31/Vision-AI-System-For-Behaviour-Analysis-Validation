"""
perception/vlm/openai_adapter.py
---------------------------------
VLM adapter for OpenAI GPT-4o (and GPT-4-vision-preview).

Requires:
    pip install openai
    OPENAI_API_KEY=sk-... (in .env or environment)

Model strings:
    gpt-4o                  → recommended, cheaper, fast
    gpt-4o-mini             → cheapest, good enough for most cases
    gpt-4-turbo             → older, more expensive
"""

from __future__ import annotations

import logging
import os

from .base import BaseVLMAdapter

logger = logging.getLogger(__name__)


class OpenAIVLMAdapter(BaseVLMAdapter):

    def __init__(self, model: str = None) -> None:
        from config.settings import settings
        self._model  = model or settings.openai_model
        self._client = None
        self._setup()

    @property
    def model_name(self) -> str:
        return self._model

    def _setup(self) -> None:
        try:
            from openai import OpenAI
            from config.settings import settings
            self._client = OpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI adapter ready (model=%s)", self._model)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def _call_api(self, prompt_messages: list[dict], image_b64: str) -> str:
        """
        Replace the placeholder image URL in the user message with the real base64 image,
        then call the OpenAI chat completions API.
        """
        # Inject real image into the user content blocks
        messages = _inject_image(prompt_messages, image_b64, "openai")

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=1000,
            temperature=0.1,   # low temperature = more consistent JSON
        )
        return response.choices[0].message.content


# ─────────────────────────────────────────────
# Shared image injection utility
# ─────────────────────────────────────────────

def _inject_image(
    prompt_messages: list[dict],
    image_b64: str,
    provider: str,
) -> list[dict]:
    """
    Replace the placeholder image_url in the user message content
    with the actual base64-encoded image in the format the provider expects.
    """
    messages = []
    for msg in prompt_messages:
        if msg["role"] != "user":
            messages.append(msg)
            continue

        new_content = []
        for block in msg["content"]:
            if block.get("type") == "image_url":
                if provider == "openai":
                    new_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high",
                        },
                    })
                elif provider == "gemini":
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    })
                elif provider == "anthropic":
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    })
            else:
                new_content.append(block)

        messages.append({"role": msg["role"], "content": new_content})

    return messages
