"""
api/jobs.py
------------
Async analysis jobs. Each uploaded (or demo) video becomes a Job that runs the
existing pipeline (ingestion → perception → rules → alerts) in a background
thread, streaming per-frame results into an in-memory store the API polls.

Why a job pattern: analysing a clip takes seconds-to-minutes (one VLM call per
sampled frame). That doesn't fit a single blocking HTTP request, so we return a
job_id immediately and let the UI poll for progress + the final verdict.

A per-job frame cap protects the deployment's shared VLM quota.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2

from ingestion import FeedConfig, FrameSampler
from ingestion.config import SourceType
from perception import PerceptionPipeline, VLMResult
from rules import RulesEngine
from rules.alerts import Alert, LogAlertSink

logger = logging.getLogger(__name__)

FRAMES_ROOT = Path(os.environ.get("FRAMES_ROOT", "frames/api"))
MAX_FRAMES  = int(os.environ.get("MAX_FRAMES_PER_JOB", "30"))   # protect shared quota
CONFIG_DIR  = "config"

KEY_ENV = {
    "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
}
KNOWN_PROVIDERS = ["mock", "groq", "gemini", "openai", "anthropic"]


def provider_status() -> list[dict]:
    """Which providers are usable right now (mock always; others need their key)."""
    return [
        {"id": p, "ready": True if p == "mock" else bool(os.environ.get(KEY_ENV[p]))}
        for p in KNOWN_PROVIDERS
    ]


def effective_provider(requested: Optional[str]) -> tuple[str, Optional[str]]:
    """
    Resolve which VLM provider to actually use. Falls back to 'mock' (which needs
    no key and never fails) when the requested provider's key is absent, so a
    fresh deploy with no secrets still runs. Returns (provider, note).
    """
    p = (requested or os.environ.get("VLM_PROVIDER", "mock")).lower()
    needed = KEY_ENV.get(p)
    if needed and not os.environ.get(needed):
        return "mock", f"No {needed} set — running in MOCK mode (simulated output, not a real VLM)."
    return p, None


class JobAlertSink:
    """Routes rule alerts into the owning job."""
    def __init__(self, job: "Job") -> None:
        self._job = job
    def emit(self, alert: Alert) -> None:
        self._job.alerts.append(alert.to_dict())


class Job:
    def __init__(self, video_path: Path, use_case: str, interval: float,
                 provider: Optional[str]) -> None:
        self.id        = uuid.uuid4().hex[:12]
        self.video     = video_path
        self.use_case  = use_case
        self.interval  = max(1.0, float(interval))
        self.provider, self.note = effective_provider(provider)

        self.status    = "queued"          # queued | running | done | error
        self.frames:   list[dict] = []
        self.alerts:   list[dict] = []
        self.verdict:  Optional[dict] = None
        self.error:    Optional[str]  = None
        self.total     = self._estimate_total()
        self.created   = time.time()

        self._stop     = False
        self._sampler: Optional[FrameSampler] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _estimate_total(self) -> int:
        try:
            cap = cv2.VideoCapture(str(self.video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n   = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            secs = (n / fps) if fps else 0
            return max(1, min(MAX_FRAMES, math.ceil(secs / self.interval) + 1))
        except Exception:
            return MAX_FRAMES

    def _run(self) -> None:
        self.status = "running"
        try:
            pipeline = PerceptionPipeline.from_config(
                zones_config=f"{CONFIG_DIR}/zones.yaml",
                identities_config=f"{CONFIG_DIR}/identities.yaml",
                use_cases_config=f"{CONFIG_DIR}/use_cases.yaml",
                vlm_provider=self.provider,
            )
            rules = RulesEngine.from_config(
                f"{CONFIG_DIR}/use_cases.yaml",
                sinks=[LogAlertSink(), JobAlertSink(self)],
            )
            pipeline.add_result_callback(self._on_result)
            pipeline.add_result_callback(rules.evaluate)

            cfg = FeedConfig(
                id=self.id, name="api_upload", source_type=SourceType.FILE,
                source=str(self.video), use_case=self.use_case,
                sample_interval_seconds=self.interval,
                output_dir=str(FRAMES_ROOT / self.id),
                reconnect_attempts=0, reconnect_delay_seconds=0,
            )
            sampler = FrameSampler(cfg)
            self._sampler = sampler
            sampler.add_callback(pipeline.process_frame)

            def on_end(session_id: str) -> None:
                verdict = rules.finalize(session_id)
                pipeline.end_session(session_id)
                if verdict and self.verdict is None:
                    self.verdict = verdict

            sampler.add_session_end_callback(on_end)
            sampler.start()
            while sampler.is_running and not self._stop:
                time.sleep(0.3)
            sampler.stop()                      # idempotent; fires session end once
            if self.verdict is None:            # frame-cap path: finalize manually
                on_end(sampler._session_id)
            self.status = "done"
        except Exception as exc:                # noqa: BLE001
            logger.exception("Job %s failed", self.id)
            self.status = "error"
            self.error  = str(exc)

    def _on_result(self, r: VLMResult) -> None:
        fname = Path(r.frame_path).name if r.frame_path else None
        self.frames.append({
            "frame_index":   r.frame_index,
            "behaviors":     r.behaviors_detected,
            "reasoning":     r.reasoning,
            "person_active": r.person_active,
            "anomaly":       r.anomaly_detected,
            "latency_ms":    round(r.latency_ms),
            "parse_success": r.parse_success,
            "frame_url":     f"/api/frames/{self.id}/{fname}" if fname else None,
        })
        if len(self.frames) >= MAX_FRAMES:
            self._stop = True               # worker loop will stop the sampler

    # ── serialisation ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "id":       self.id,
            "status":   self.status,
            "use_case": self.use_case,
            "provider": self.provider,
            "note":     self.note,
            "interval": self.interval,
            "processed": len(self.frames),
            "total":    self.total,
            "capped":   self._stop,
            "frames":   self.frames,
            "alerts":   self.alerts,
            "verdict":  self.verdict,
            "error":    self.error,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, video_path: Path, use_case: str, interval: float,
               provider: Optional[str]) -> Job:
        job = Job(video_path, use_case, interval, provider)
        with self._lock:
            self._jobs[job.id] = job
        job.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)


STORE = JobStore()
