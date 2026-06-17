"""
config/settings.py
-------------------
Loads .env automatically at import time.
Every other module imports from here instead of reading os.environ directly.

Usage
-----
    from config.settings import settings

    print(settings.vlm_provider)      # "gemini"
    print(settings.gemini_api_key)    # "AIza..."
    print(settings.gemini_model)      # "gemini-2.0-flash"

Why this exists
---------------
Without this, every file does os.environ.get("GEMINI_API_KEY") and you have
to remember to export variables in the terminal. With this, you just fill in
.env once and forget about it — everything is loaded automatically on startup.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_env_file(env_path: Path) -> None:
    """
    Minimal .env parser — no external library needed.
    Reads KEY=VALUE lines and sets them in os.environ.
    Skips comments (#) and blank lines.
    Does NOT override already-set environment variables
    (so real exports still take priority over .env).
    """
    if not env_path.exists():
        logger.debug(".env not found at %s — using system environment only", env_path)
        return

    loaded = 0
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")

            # Don't override real environment variables
            if key not in os.environ:
                os.environ[key] = value
                loaded += 1

    logger.debug("Loaded %d key(s) from %s", loaded, env_path)


# ── Auto-load .env when this module is imported ───────────────────────────────
# Look for .env in current directory AND project root (one level up from config/)
_here = Path(__file__).parent
for _candidate in [Path(".env"), _here.parent / ".env", _here / ".env"]:
    if _candidate.exists():
        _load_env_file(_candidate.resolve())
        break


# ── Typed settings accessor ───────────────────────────────────────────────────

class Settings:
    """
    Single place to read all environment variables.
    Gives you autocomplete and clear error messages.
    """

    # ── VLM ──────────────────────────────────────────────────────────────────

    @property
    def vlm_provider(self) -> str:
        return os.environ.get("VLM_PROVIDER", "mock").lower()

    # ── Gemini ────────────────────────────────────────────────────────────────

    @property
    def gemini_api_key(self) -> str:
        return self._require("GEMINI_API_KEY")

    @property
    def gemini_model(self) -> str:
        # gemini-2.5-flash is the recommended free-tier model (2026): 10 RPM / 250 req-day.
        # For higher daily volume use gemini-2.5-flash-lite (15 RPM / 1000 req-day).
        return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # ── Groq ──────────────────────────────────────────────────────────────────

    @property
    def groq_api_key(self) -> str:
        return self._require("GROQ_API_KEY")

    @property
    def groq_model(self) -> str:
        # Llama 4 Scout is vision-capable and on Groq's high-quota free tier
        # (30 RPM / 1000 RPD). No credit card.
        return os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    # ── HuggingFace ───────────────────────────────────────────────────────────

    @property
    def huggingface_api_key(self) -> str:
        return self._require("HUGGINGFACE_API_KEY")

    @property
    def huggingface_model(self) -> str:
        return os.environ.get(
            "HUGGINGFACE_MODEL",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
        )

    # ── OpenAI ────────────────────────────────────────────────────────────────

    @property
    def openai_api_key(self) -> str:
        return self._require("OPENAI_API_KEY")

    @property
    def openai_model(self) -> str:
        return os.environ.get("OPENAI_MODEL", "gpt-4o")

    # ── Anthropic ─────────────────────────────────────────────────────────────

    @property
    def anthropic_api_key(self) -> str:
        return self._require("ANTHROPIC_API_KEY")

    @property
    def anthropic_model(self) -> str:
        return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # ── Mock ──────────────────────────────────────────────────────────────────

    @property
    def mock_latency_ms(self) -> float:
        return float(os.environ.get("MOCK_LATENCY_MS", "300"))

    @property
    def mock_anomaly_rate(self) -> float:
        return float(os.environ.get("MOCK_ANOMALY_RATE", "0.1"))

    # ── General ───────────────────────────────────────────────────────────────

    @property
    def log_level(self) -> str:
        return os.environ.get("LOG_LEVEL", "INFO").upper()

    @property
    def default_sample_interval(self) -> float:
        return float(os.environ.get("DEFAULT_SAMPLE_INTERVAL", "5"))

    # ── Helper ────────────────────────────────────────────────────────────────

    def _require(self, key: str) -> str:
        value = os.environ.get(key, "")
        if not value or value.startswith("your_"):
            raise EnvironmentError(
                f"\n\n  Missing API key: {key}\n"
                f"  Open .env and replace 'your_..._here' with your actual key.\n"
                f"  Get a free Groq key (recommended): https://console.groq.com/keys\n"
                f"  Get a free Gemini key: https://aistudio.google.com/app/apikey\n"
            )
        return value

    def print_status(self) -> None:
        """Print which providers are configured — useful for debugging."""
        def _check(key): return "✓ set" if (
            os.environ.get(key, "").strip() and
            not os.environ.get(key, "").startswith("your_")
        ) else "✗ not set"

        print("\n── API Key Status ──────────────────────────────────")
        print(f"  VLM_PROVIDER      : {self.vlm_provider}")
        print(f"  GEMINI_API_KEY    : {_check('GEMINI_API_KEY')}   (model: {self.gemini_model})")
        print(f"  GROQ_API_KEY      : {_check('GROQ_API_KEY')}   (model: {self.groq_model})")
        print(f"  HUGGINGFACE_API_KEY: {_check('HUGGINGFACE_API_KEY')}   (model: {self.huggingface_model})")
        print(f"  OPENAI_API_KEY    : {_check('OPENAI_API_KEY')}")
        print(f"  ANTHROPIC_API_KEY : {_check('ANTHROPIC_API_KEY')}")
        print("────────────────────────────────────────────────────\n")


# Singleton instance — import this everywhere
settings = Settings()
