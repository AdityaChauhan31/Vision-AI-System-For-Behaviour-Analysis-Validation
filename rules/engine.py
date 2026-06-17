"""
rules/engine.py
----------------
Stage 6 — Rules & Validation Engine.

This is the deterministic half of the system. The VLM (Stage 4) says WHAT is
happening; this engine decides whether what's happening is OK, using rules that
live entirely in use_cases.yaml — no rule logic is hardcoded per use case.

Design
------
- One SessionState per session_id, built up frame by frame.
- Two classes of rule:
    * streaming  → evaluated every frame (idle, loitering, unauthorized entry)
    * completion → evaluated once at session end (missing steps, duration)
- A trigger only fires if it is listed in that use case's `alert_triggers`,
  AND its condition holds. Thresholds come from the use case's `rules` block.
- Adding a new use case that reuses this trigger vocabulary = YAML only.
  Adding a brand-new KIND of check = one new method here (documented trade-off).

The same engine serves anonymous and identity-bound use cases; identity-bound
triggers simply read the identity/zone flags the perception stage attached.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import yaml

from perception.enriched_frame import VLMResult
from .alerts import (
    Alert, AlertSink, JsonFileSink, LogAlertSink,
    SEVERITY_CRITICAL, SEVERITY_INFO, SEVERITY_WARNING,
)
from .session import FrameObservation, SessionState

logger = logging.getLogger(__name__)

_DEFAULT_IDLE_BEHAVIORS       = {"idle_standing", "standing_idle"}
_DEFAULT_SUSPICIOUS_BEHAVIORS = {"looking_around_suspiciously"}


def _seconds(rules: dict, sec_key: str, min_key: str, default: float = 0.0) -> float:
    """Read a threshold that may be given in seconds or minutes."""
    if sec_key in rules and rules[sec_key] is not None:
        return float(rules[sec_key])
    if min_key in rules and rules[min_key] is not None:
        return float(rules[min_key]) * 60.0
    return default


class RulesEngine:
    """
    Register as a VLMResult callback:
        engine = RulesEngine.from_config("config/use_cases.yaml")
        perception.add_result_callback(engine.evaluate)
        ...
        engine.finalize(session_id)   # at session end → emits verdict
    """

    def __init__(self, use_cases: dict[str, dict], sinks: Optional[list[AlertSink]] = None,
                 session_log_dir: str = "logs/sessions") -> None:
        self._use_cases = use_cases
        self._sinks     = sinks if sinks is not None else [LogAlertSink(), JsonFileSink()]
        self._sessions: dict[str, SessionState] = {}
        self._log_dir   = Path(session_log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, use_cases_config: str = "config/use_cases.yaml",
                    sinks: Optional[list[AlertSink]] = None) -> "RulesEngine":
        path = Path(use_cases_config)
        use_cases: dict[str, dict] = {}
        if path.exists():
            with path.open() as f:
                raw = yaml.safe_load(f) or {}
            for uc in raw.get("use_cases", []):
                use_cases[uc["id"]] = uc
            logger.info("RulesEngine loaded %d use case(s)", len(use_cases))
        else:
            logger.warning("use_cases.yaml not found at %s — engine has no rules", path)
        return cls(use_cases, sinks)

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self, result: VLMResult) -> None:
        """Called for every frame's VLMResult. Updates state, fires streaming alerts."""
        state = self._sessions.get(result.session_id)
        if state is None:
            state = SessionState(
                session_id=result.session_id, feed_id=result.feed_id,
                use_case=result.use_case, person_id=result.person_id,
                identity_known=result.identity_known,
            )
            self._sessions[result.session_id] = state

        # keep the latest known identity for the session
        if result.identity_known:
            state.person_id = result.person_id
            state.identity_known = True

        state.record(FrameObservation(
            frame_index=result.frame_index,
            timestamp=result.timestamp_utc,
            behaviors=list(result.behaviors_detected),
            active=result.person_active,
            vlm_anomaly=result.anomaly_detected,
            parse_ok=result.parse_success,
        ))

        rules    = self._rules_for(result.use_case)
        triggers = self._triggers_for(result.use_case)
        self._check_streaming(state, result, rules, triggers)

    def finalize(self, session_id: str) -> Optional[dict]:
        """Called at session end. Fires completion alerts and writes a verdict."""
        state = self._sessions.get(session_id)
        if state is None or state.finalized:
            return None
        state.finalized = True

        rules    = self._rules_for(state.use_case)
        triggers = self._triggers_for(state.use_case)
        completion_alerts = self._check_completion(state, rules, triggers)

        compliant = len(state.fired_triggers) == 0 and len(completion_alerts) == 0
        verdict = {
            "compliant":      compliant,
            "fired_triggers": sorted(state.fired_triggers | {a.trigger for a in completion_alerts}),
            "session":        state.summary(),
        }
        self._write_verdict(state, verdict)
        logger.info(
            "SESSION VERDICT [%s] %s | triggers=%s | %s",
            session_id, "COMPLIANT" if compliant else "VIOLATION",
            verdict["fired_triggers"] or "none",
            f"{state.duration_seconds:.0f}s / {state.frame_count} frames",
        )
        return verdict

    def finalize_all(self) -> None:
        for sid in list(self._sessions):
            self.finalize(sid)

    # ── streaming checks (per frame) ───────────────────────────────────────────

    def _check_streaming(self, state: SessionState, result: VLMResult,
                         rules: dict, triggers: set[str]) -> None:

        # unauthorized entry / unknown person in restricted zone (identity-bound path)
        if result.is_restricted_zone:
            if "unknown_person_in_restricted_zone" in triggers and not result.identity_known:
                self._fire_once(state, Alert(
                    trigger="unknown_person_in_restricted_zone", severity=SEVERITY_CRITICAL,
                    message=f"Unknown person detected in restricted zone '{result.zone_label}'",
                    **self._ctx(state, result)))
            elif "unauthorized_zone_entry" in triggers:
                self._fire_once(state, Alert(
                    trigger="unauthorized_zone_entry", severity=SEVERITY_CRITICAL,
                    message=f"{result.person_id} is not authorized for zone '{result.zone_label}'",
                    **self._ctx(state, result)))

        # suspicious behaviour
        if "suspicious_behavior_detected" in triggers:
            suspicious = set(rules.get("suspicious_behaviors", _DEFAULT_SUSPICIOUS_BEHAVIORS))
            if rules.get("alert_on_pacing"):
                suspicious.add("pacing")
            hit = suspicious.intersection(result.behaviors_detected)
            if hit:
                self._fire_once(state, Alert(
                    trigger="suspicious_behavior_detected", severity=SEVERITY_WARNING,
                    message=f"Suspicious behaviour observed: {sorted(hit)}",
                    **self._ctx(state, result)), key=f"suspicious:{sorted(hit)}")

        # idle / loitering (time-accumulated)
        idle_behaviors = set(rules.get("idle_behaviors", _DEFAULT_IDLE_BEHAVIORS))
        idle = state.idle_seconds(idle_behaviors)

        if "extended_idle" in triggers:
            thr = _seconds(rules, "alert_on_idle_seconds", "alert_on_idle_minutes")
            if thr and idle >= thr:
                self._fire_once(state, Alert(
                    trigger="extended_idle", severity=SEVERITY_WARNING,
                    message=f"Idle for {idle:.0f}s (threshold {thr:.0f}s)",
                    **self._ctx(state, result), ))

        if "loitering_threshold_exceeded" in triggers:
            thr = float(rules.get("max_idle_duration_seconds", 0) or 0)
            if thr and idle >= thr:
                self._fire_once(state, Alert(
                    trigger="loitering_threshold_exceeded", severity=SEVERITY_WARNING,
                    message=f"Loitering: idle {idle:.0f}s (threshold {thr:.0f}s)",
                    **self._ctx(state, result)))

    # ── completion checks (session end) ─────────────────────────────────────────

    def _check_completion(self, state: SessionState, rules: dict,
                          triggers: set[str]) -> list[Alert]:
        uc = self._use_cases.get(state.use_case, {})
        alerts: list[Alert] = []

        # missing required step
        if "missing_required_step" in triggers:
            required = set(uc.get("required_behaviors", []))
            must_all = rules.get("all_required_steps_must_complete", bool(required))
            if required and must_all:
                missing = required - state.behaviors_seen
                if missing:
                    alerts.append(Alert(
                        trigger="missing_required_step", severity=SEVERITY_CRITICAL,
                        message=f"Required steps not observed: {sorted(missing)}",
                        **self._ctx(state), details={"missing": sorted(missing),
                                                     "seen": sorted(state.behaviors_seen)}))

        dur = state.duration_seconds
        # visit too short
        if "visit_too_short" in triggers:
            mn = _seconds(rules, "min_duration_seconds", "min_duration_minutes")
            if mn and dur < mn:
                alerts.append(Alert(
                    trigger="visit_too_short", severity=SEVERITY_WARNING,
                    message=f"Visit {dur:.0f}s shorter than required {mn:.0f}s",
                    **self._ctx(state), details={"duration_s": round(dur, 1), "min_s": mn}))

        # visit too long
        if "visit_too_long" in triggers:
            mx = _seconds(rules, "max_duration_seconds", "max_duration_minutes")
            if mx and dur > mx:
                alerts.append(Alert(
                    trigger="visit_too_long", severity=SEVERITY_WARNING,
                    message=f"Visit {dur:.0f}s exceeded allowed {mx:.0f}s",
                    **self._ctx(state), details={"duration_s": round(dur, 1), "max_s": mx}))

        for a in alerts:
            self._emit(a)
        return alerts

    # ── helpers ──────────────────────────────────────────────────────────────

    def _rules_for(self, use_case: str) -> dict:
        return self._use_cases.get(use_case, {}).get("rules", {}) or {}

    def _triggers_for(self, use_case: str) -> set[str]:
        return set(self._use_cases.get(use_case, {}).get("alert_triggers", []))

    def _ctx(self, state: SessionState, result: Optional[VLMResult] = None) -> dict:
        """Common Alert context fields from session (+ frame if available)."""
        return {
            "feed_id":       state.feed_id,
            "session_id":    state.session_id,
            "use_case":      state.use_case,
            "person_id":     result.person_id if result else state.person_id,
            "zone_label":    result.zone_label if result else None,
            "frame_index":   result.frame_index if result else None,
            "snapshot_path": str(result.frame_path) if (result and result.frame_path) else None,
        }

    def _fire_once(self, state: SessionState, alert: Alert, key: Optional[str] = None) -> None:
        """Fire a streaming alert at most once per session (per optional key)."""
        dedupe = key or alert.trigger
        if dedupe in state.fired_triggers:
            return
        state.fired_triggers.add(dedupe)
        # store canonical trigger name too (for verdict)
        state.fired_triggers.add(alert.trigger)
        self._emit(alert)

    def _emit(self, alert: Alert) -> None:
        for sink in self._sinks:
            try:
                sink.emit(alert)
            except Exception as exc:
                logger.error("Alert sink %s failed: %s", type(sink).__name__, exc)

    def _write_verdict(self, state: SessionState, verdict: dict) -> None:
        out = self._log_dir / f"{state.session_id}.json"
        try:
            out.write_text(json.dumps(verdict, indent=2))
        except Exception as exc:
            logger.error("Failed to write session verdict %s: %s", out, exc)
