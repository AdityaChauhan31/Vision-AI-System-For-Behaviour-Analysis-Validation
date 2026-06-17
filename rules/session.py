"""
rules/session.py
-----------------
Accumulates VLM observations for ONE session (one continuous run of one feed).

The rules engine is stateful: a single frame can't tell you whether a required
step was skipped or whether a visit was too short — only the accumulated session
can. This object is that accumulator. It holds no rule logic itself; it just
records what was seen so the engine can evaluate declarative rules against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class FrameObservation:
    frame_index: int
    timestamp:   Optional[datetime]
    behaviors:   list[str]
    active:      bool
    vlm_anomaly: bool
    parse_ok:    bool


@dataclass
class SessionState:
    session_id: str
    feed_id:    str
    use_case:   str
    person_id:  str = "unknown"
    identity_known: bool = False

    observations:   list[FrameObservation] = field(default_factory=list)
    behaviors_seen: set[str]               = field(default_factory=set)
    fired_triggers: set[str]               = field(default_factory=set)  # de-dupe streaming alerts
    finalized:      bool                   = False

    # ── derived helpers ──────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        return len(self.observations)

    @property
    def start_time(self) -> Optional[datetime]:
        ts = [o.timestamp for o in self.observations if o.timestamp]
        return min(ts) if ts else None

    @property
    def end_time(self) -> Optional[datetime]:
        ts = [o.timestamp for o in self.observations if o.timestamp]
        return max(ts) if ts else None

    @property
    def duration_seconds(self) -> float:
        s, e = self.start_time, self.end_time
        return (e - s).total_seconds() if s and e else 0.0

    def record(self, obs: FrameObservation) -> None:
        self.observations.append(obs)
        if obs.parse_ok:
            self.behaviors_seen.update(obs.behaviors)

    def idle_seconds(self, idle_behaviors: set[str]) -> float:
        """
        Total seconds the person spent idle. A frame counts as idle when it is
        not active, or its only behaviors are in `idle_behaviors`. Per-frame
        dwell is the gap to the next observation (median gap for the last one).
        """
        if len(self.observations) < 2:
            return 0.0
        gaps = []
        for i in range(len(self.observations) - 1):
            a, b = self.observations[i], self.observations[i + 1]
            if a.timestamp and b.timestamp:
                gaps.append((b.timestamp - a.timestamp).total_seconds())
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0

        total = 0.0
        for i, o in enumerate(self.observations):
            gap = gaps[i] if i < len(gaps) else median_gap
            behset = set(o.behaviors)
            is_idle = (not o.active) or (behset and behset.issubset(idle_behaviors))
            if is_idle:
                total += gap
        return total

    def summary(self) -> dict:
        return {
            "session_id":      self.session_id,
            "feed_id":         self.feed_id,
            "use_case":        self.use_case,
            "person_id":       self.person_id,
            "identity_known":  self.identity_known,
            "frame_count":     self.frame_count,
            "duration_seconds": round(self.duration_seconds, 1),
            "behaviors_seen":  sorted(self.behaviors_seen),
            "start_time":      self.start_time.isoformat() if self.start_time else None,
            "end_time":        self.end_time.isoformat() if self.end_time else None,
        }
