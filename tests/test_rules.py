"""
tests/test_rules.py
--------------------
Tests the Stage 6 rules engine directly with synthetic VLMResults — no video,
no VLM, no API keys. Verifies behaviour (alerts fired / verdict) not internals.

Run:  pytest tests/test_rules.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from perception.enriched_frame import VLMResult
from rules.alerts import Alert
from rules.engine import RulesEngine


class _CaptureSink:
    def __init__(self):
        self.alerts: list[Alert] = []
    def emit(self, alert: Alert) -> None:
        self.alerts.append(alert)


_USE_CASES = {
    "housekeeping_demo_short": {
        "id": "housekeeping_demo_short",
        "required_behaviors": ["mopping_floor", "wiping_surfaces"],
        "rules": {
            "min_duration_seconds": 5,
            "max_duration_seconds": 600,
            "all_required_steps_must_complete": True,
        },
        "alert_triggers": ["missing_required_step", "visit_too_short", "visit_too_long"],
    },
    "identity_restriction": {
        "id": "identity_restriction",
        "required_behaviors": [],
        "rules": {},
        "alert_triggers": ["unauthorized_zone_entry", "unknown_person_in_restricted_zone"],
    },
}

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _result(session_id, frame_index, behaviors, *, t_offset_s=0, use_case="housekeeping_demo_short",
            person_id="unknown", identity_known=False, restricted=False, zone=None):
    return VLMResult(
        behaviors_detected=behaviors, behaviors_confidence={b: 0.9 for b in behaviors},
        person_active=bool(behaviors), estimated_activity_duration_seconds=30,
        anomaly_detected=False, anomaly_description=None, reasoning="test",
        frame_index=frame_index, session_id=session_id, feed_id="f1", use_case=use_case,
        person_id=person_id, zone_label=zone, vlm_model="mock", latency_ms=10.0,
        timestamp_utc=_T0 + timedelta(seconds=t_offset_s),
        identity_known=identity_known, is_restricted_zone=restricted,
    )


def _engine():
    sink = _CaptureSink()
    return RulesEngine(_USE_CASES, sinks=[sink], session_log_dir="/tmp/vai_test_sessions"), sink


def test_compliant_session_fires_no_alerts():
    eng, sink = _engine()
    sid = "s_ok"
    eng.evaluate(_result(sid, 1, ["mopping_floor"], t_offset_s=0))
    eng.evaluate(_result(sid, 2, ["wiping_surfaces"], t_offset_s=10))
    verdict = eng.finalize(sid)
    assert verdict["compliant"] is True
    assert sink.alerts == []


def test_missing_required_step_fires():
    eng, sink = _engine()
    sid = "s_missing"
    eng.evaluate(_result(sid, 1, ["mopping_floor"], t_offset_s=0))
    eng.evaluate(_result(sid, 2, ["arranging_items"], t_offset_s=10))   # wiping never seen
    verdict = eng.finalize(sid)
    assert verdict["compliant"] is False
    assert "missing_required_step" in verdict["fired_triggers"]
    triggers = {a.trigger for a in sink.alerts}
    assert "missing_required_step" in triggers


def test_visit_too_short_fires():
    eng, sink = _engine()
    sid = "s_short"
    eng.evaluate(_result(sid, 1, ["mopping_floor"], t_offset_s=0))
    eng.evaluate(_result(sid, 2, ["wiping_surfaces"], t_offset_s=2))    # 2s < 5s min
    verdict = eng.finalize(sid)
    assert "visit_too_short" in verdict["fired_triggers"]


def test_unknown_person_in_restricted_zone_is_critical():
    eng, sink = _engine()
    sid = "s_intruder"
    eng.evaluate(_result(sid, 1, ["present_in_zone"], use_case="identity_restriction",
                         restricted=True, identity_known=False, zone="restricted_area"))
    eng.finalize(sid)
    crit = [a for a in sink.alerts if a.trigger == "unknown_person_in_restricted_zone"]
    assert len(crit) == 1
    assert crit[0].severity == "critical"


def test_streaming_alert_fires_once_per_session():
    eng, sink = _engine()
    sid = "s_dedupe"
    for i in range(4):
        eng.evaluate(_result(sid, i + 1, ["present_in_zone"], use_case="identity_restriction",
                             restricted=True, identity_known=False, zone="restricted_area",
                             t_offset_s=i * 3))
    eng.finalize(sid)
    intruder_alerts = [a for a in sink.alerts if a.trigger == "unknown_person_in_restricted_zone"]
    assert len(intruder_alerts) == 1   # de-duped despite 4 frames
