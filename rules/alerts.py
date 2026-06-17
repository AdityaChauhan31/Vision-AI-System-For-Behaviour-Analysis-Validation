"""
rules/alerts.py
----------------
Stage 7 — Alerting.

An Alert is the platform's output when a rule is met or violated. Each alert
carries the context the brief asks for: feed, rule/trigger, person, zone,
timestamp, and a snapshot path for evidence.

AlertSink is an open interface so new delivery channels (webhook, email,
FastAPI endpoint, DB) drop in without touching the rules engine:
  - LogAlertSink   → human-readable line in the logs (always on)
  - JsonFileSink   → append-only JSONL alert record (the "database" for the demo)

Both anonymous and identity-bound paths emit the SAME Alert type through the
SAME sinks — satisfying the "one engine for both paths" hard rule.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


SEVERITY_INFO     = "info"
SEVERITY_WARNING  = "warning"
SEVERITY_CRITICAL = "critical"


@dataclass
class Alert:
    """One rule outcome worth surfacing to an operator."""
    trigger:       str                 # e.g. "missing_required_step"
    severity:      str                 # info | warning | critical
    message:       str                 # human-readable summary
    feed_id:       str
    session_id:    str
    use_case:      str
    person_id:     str = "unknown"
    zone_label:    Optional[str] = None
    frame_index:   Optional[int] = None
    snapshot_path: Optional[str] = None      # JPEG on disk = evidence
    raised_at:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details:       dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AlertSink(Protocol):
    """Anything that can receive an Alert. Implement `emit`."""
    def emit(self, alert: Alert) -> None: ...


class LogAlertSink:
    """Writes a clear, operator-readable line to the logger."""

    _ICON = {SEVERITY_INFO: "ℹ️", SEVERITY_WARNING: "⚠️", SEVERITY_CRITICAL: "🚨"}

    def emit(self, alert: Alert) -> None:
        icon = self._ICON.get(alert.severity, "•")
        logger.warning(
            "%s ALERT [%s] %s | feed=%s session=%s person=%s zone=%s frame=%s",
            icon, alert.trigger, alert.message,
            alert.feed_id, alert.session_id, alert.person_id,
            alert.zone_label or "none", alert.frame_index,
        )


class JsonFileSink:
    """
    Append-only JSONL alert log. One JSON object per line.
    This is the demo's stand-in for a real alert database — swap for a DB
    sink later without changing the engine.
    """

    def __init__(self, path: str = "logs/alerts.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, alert: Alert) -> None:
        try:
            with self._path.open("a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as exc:
            logger.error("Failed to write alert to %s: %s", self._path, exc)
