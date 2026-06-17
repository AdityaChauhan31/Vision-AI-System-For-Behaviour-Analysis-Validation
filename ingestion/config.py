"""
ingestion/config.py
-------------------
Pydantic models that validate and parse feeds.yaml.
Any bad config (wrong types, missing fields) raises a clear error at startup,
not mid-run when a camera has already been connected for 20 minutes.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class SourceType(str, Enum):
    RTSP   = "rtsp"
    FILE   = "file"
    WEBCAM = "webcam"


# ─────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────

class ResizeConfig(BaseModel):
    width:  int = Field(gt=0, le=7680, description="Target width in pixels")
    height: int = Field(gt=0, le=4320, description="Target height in pixels")


# ─────────────────────────────────────────────
# Main feed model
# ─────────────────────────────────────────────

class FeedConfig(BaseModel):
    id:                        str
    name:                      str
    source_type:               SourceType
    source:                    Union[str, int]        # URL / path / device index
    use_case:                  str
    sample_interval_seconds:   float = Field(gt=0, default=5.0)
    enabled:                   bool  = True
    output_dir:                str   = "frames"
    reconnect_attempts:        int   = Field(ge=0, default=5)
    reconnect_delay_seconds:   float = Field(ge=0, default=3.0)
    resize:                    Optional[ResizeConfig] = None
    loop:                      bool  = False           # file source only

    # ── validators ──────────────────────────────────

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: Union[str, int]) -> Union[str, int]:
        if isinstance(v, int) and v < 0:
            raise ValueError("Webcam device index must be >= 0")
        return v

    @model_validator(mode="after")
    def validate_source_type_match(self) -> "FeedConfig":
        if self.source_type == SourceType.WEBCAM and not isinstance(self.source, int):
            raise ValueError(
                f"Feed '{self.id}': webcam source must be an integer device index, got '{self.source}'"
            )
        if self.source_type in (SourceType.RTSP, SourceType.FILE) and not isinstance(self.source, str):
            raise ValueError(
                f"Feed '{self.id}': rtsp/file source must be a string, got '{type(self.source).__name__}'"
            )
        return self

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def cv2_source(self) -> Union[str, int]:
        """The value passed directly to cv2.VideoCapture()."""
        return self.source  # type: ignore[return-value]


# ─────────────────────────────────────────────
# Top-level loader
# ─────────────────────────────────────────────

class FeedsConfig(BaseModel):
    feeds: list[FeedConfig]

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "FeedsConfig":
        """
        Load and validate feeds.yaml.

        Raises:
            FileNotFoundError  – config file missing
            pydantic.ValidationError – malformed config
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path.resolve()}")

        with config_path.open("r") as f:
            raw = yaml.safe_load(f)

        return cls(**raw)

    def enabled_feeds(self) -> list[FeedConfig]:
        """Return only feeds that have enabled=true."""
        return [f for f in self.feeds if f.enabled]

    def get_feed(self, feed_id: str) -> Optional[FeedConfig]:
        """Look up a feed by its id."""
        return next((f for f in self.feeds if f.id == feed_id), None)
