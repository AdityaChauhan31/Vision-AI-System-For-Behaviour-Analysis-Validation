"""
ingestion/frame_sampler.py
--------------------------
Handles ONE video feed:
  - Connects via OpenCV (RTSP / file / webcam)
  - Samples one frame every N seconds
  - Saves frames as timestamped JPEGs
  - Auto-reconnects on disconnect (with backoff)
  - Emits a FrameEvent dataclass for downstream consumers (VLM, rules engine)
  - Runs in its own thread so multiple feeds run in parallel
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .config import FeedConfig, SourceType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data contract passed to downstream stages
# ─────────────────────────────────────────────

@dataclass
class FrameEvent:
    """
    Everything downstream stages need to know about one sampled frame.
    This is the unit of work that flows through the entire pipeline.
    """
    feed_id:          str                  # which camera
    use_case:         str                  # which behavioral ruleset to apply
    frame_index:      int                  # sequential counter within this session
    timestamp_utc:    datetime             # when the frame was captured
    frame_path:       Path                 # saved JPEG on disk
    frame:            np.ndarray           # raw pixel data (H x W x 3, BGR)
    source_type:      str                  # rtsp | file | webcam
    session_id:       str                  # groups frames from one continuous run
    metadata:         dict = field(default_factory=dict)   # extensible

    @property
    def timestamp_iso(self) -> str:
        return self.timestamp_utc.strftime("%Y%m%dT%H%M%S")

    @property
    def filename(self) -> str:
        return self.frame_path.name


# ─────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────

# Callback type: receives a FrameEvent, returns nothing
FrameCallback = Callable[[FrameEvent], None]


class FrameSampler:
    """
    Connects to a single video source and samples frames at a
    configurable interval. Each sampled frame is:
      1. Optionally resized
      2. Saved to disk as a JPEG
      3. Passed to every registered callback (e.g. the VLM pipeline)

    Usage
    -----
        sampler = FrameSampler(config)
        sampler.add_callback(my_vlm_pipeline.process)
        sampler.start()          # non-blocking — runs in background thread
        ...
        sampler.stop()
    """

    JPEG_QUALITY = 92            # 0-100; 92 is high quality, reasonable size

    def __init__(self, config: FeedConfig) -> None:
        self.config       = config
        self._callbacks:  list[FrameCallback] = []
        self._session_end_callbacks: list[Callable[[str], None]] = []
        self._thread:     Optional[threading.Thread] = None
        self._stop_event  = threading.Event()
        self._frame_count = 0
        self._session_id  = self._make_session_id()
        self._session_start_dt = datetime.now(tz=timezone.utc)
        self._session_ended = False

        # Ensure output directory exists
        self.config.output_path.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────

    def add_callback(self, fn: FrameCallback) -> None:
        """Register a function to receive every FrameEvent."""
        self._callbacks.append(fn)

    def add_session_end_callback(self, fn: Callable[[str], None]) -> None:
        """Register a function called with session_id when the session ends."""
        self._session_end_callbacks.append(fn)

    def start(self) -> None:
        """Start sampling in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("[%s] Sampler already running", self.config.id)
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"sampler-{self.config.id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("[%s] Sampler started (interval=%.1fs)", self.config.id, self.config.sample_interval_seconds)

    def stop(self) -> None:
        """Signal the sampler to stop after the current frame."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._fire_session_end()
        logger.info("[%s] Sampler stopped. Total frames captured: %d", self.config.id, self._frame_count)

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── internal ──────────────────────────────────────────

    def _run_loop(self) -> None:
        """
        Main loop:
          1. Connect (with retries)
          2. Sample frames at the configured interval
          3. On disconnect, retry up to reconnect_attempts times
          4. Exit cleanly when stop() is called
        """
        attempt = 0
        max_attempts = self.config.reconnect_attempts

        while not self._stop_event.is_set():
            cap = self._connect()

            if cap is None:
                attempt += 1
                if max_attempts > 0 and attempt > max_attempts:
                    logger.error(
                        "[%s] Giving up after %d reconnect attempts.",
                        self.config.id, max_attempts
                    )
                    break
                wait = self.config.reconnect_delay_seconds * (2 ** min(attempt - 1, 4))  # exponential backoff
                logger.warning(
                    "[%s] Reconnect attempt %d/%s in %.0fs ...",
                    self.config.id, attempt,
                    str(max_attempts) if max_attempts > 0 else "∞",
                    wait,
                )
                time.sleep(wait)
                continue

            attempt = 0   # reset counter on successful connect
            logger.info("[%s] Connected to source: %s", self.config.id, self.config.source)

            disconnected = self._sample_from_capture(cap)
            cap.release()

            # If we stopped intentionally, exit
            if self._stop_event.is_set():
                break

            # If source was a file and it ended cleanly
            if self.config.source_type == SourceType.FILE:
                if self.config.loop:
                    logger.info("[%s] File ended — looping.", self.config.id)
                    continue
                else:
                    logger.info("[%s] File ended — stopping sampler.", self.config.id)
                    break

            # Otherwise it was an unexpected disconnect — retry
            if disconnected:
                logger.warning("[%s] Feed disconnected unexpectedly.", self.config.id)

        self._fire_session_end()

    def _fire_session_end(self) -> None:
        if self._session_ended:
            return
        self._session_ended = True
        for fn in self._session_end_callbacks:
            try:
                fn(self._session_id)
            except Exception as exc:
                logger.exception("[%s] Session-end callback failed: %s", self.config.id, exc)

    def _connect(self) -> Optional[cv2.VideoCapture]:
        """
        Open a cv2.VideoCapture. Returns None on failure.
        Sets RTSP transport to TCP for stability on network cameras.
        """
        source = self.config.cv2_source

        if self.config.source_type == SourceType.RTSP:
            # Force TCP transport — avoids UDP packet loss on busy networks
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimal buffer = less latency
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error("[%s] Failed to open source: %s", self.config.id, source)
            cap.release()
            return None

        return cap

    def _sample_from_capture(self, cap: cv2.VideoCapture) -> bool:
        """
        Pull frames at the configured interval.

        File sources are sampled by VIDEO timestamp (CAP_PROP_POS_MSEC) so a
        15s clip sampled every 5s deterministically yields frames at ~0/5/10/15s
        on any machine, regardless of how fast OpenCV decodes the file.

        Live sources (RTSP / webcam) are sampled by WALL-CLOCK time, since their
        playback rate is real-time and frames arrive as the world produces them.

        Returns True if the feed disconnected unexpectedly, False on clean stop.
        """
        if self.config.source_type == SourceType.FILE:
            return self._sample_file_by_video_time(cap)
        return self._sample_live_by_wall_clock(cap)

    def _sample_file_by_video_time(self, cap: cv2.VideoCapture) -> bool:
        interval_ms   = self.config.sample_interval_seconds * 1000.0
        next_pos_ms   = 0.0
        while not self._stop_event.is_set():
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                return True   # end of file
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms + 1e-3 < next_pos_ms:
                continue      # not yet at the next sample point — keep decoding
            next_pos_ms = pos_ms + interval_ms
            frame = self._preprocess(raw_frame)
            # Stamp the frame with SCENE time (session start + video position) so
            # downstream duration rules reason in video seconds, not decode time.
            content_ts = self._session_start_dt + timedelta(milliseconds=pos_ms)
            event = self._build_event(frame, content_ts=content_ts)
            self._save_frame(event)
            self._dispatch(event)
        return False

    def _sample_live_by_wall_clock(self, cap: cv2.VideoCapture) -> bool:
        next_sample_time = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                return True   # unexpected disconnect
            if now < next_sample_time:
                time.sleep(0.05)   # avoid busy-spin
                continue
            next_sample_time = now + self.config.sample_interval_seconds
            frame = self._preprocess(raw_frame)
            event = self._build_event(frame)
            self._save_frame(event)
            self._dispatch(event)
        return False

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize if configured. Add more preprocessing here if needed."""
        if self.config.resize:
            frame = cv2.resize(
                frame,
                (self.config.resize.width, self.config.resize.height),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _build_event(self, frame: np.ndarray, content_ts: Optional[datetime] = None) -> FrameEvent:
        self._frame_count += 1
        ts = content_ts or datetime.now(tz=timezone.utc)
        filename = f"{self.config.id}_{ts.strftime('%Y%m%dT%H%M%S%f')[:-3]}_f{self._frame_count:06d}.jpg"
        frame_path = self.config.output_path / filename

        return FrameEvent(
            feed_id       = self.config.id,
            use_case      = self.config.use_case,
            frame_index   = self._frame_count,
            timestamp_utc = ts,
            frame_path    = frame_path,
            frame         = frame,
            source_type   = self.config.source_type.value,
            session_id    = self._session_id,
            metadata      = {
                "feed_name":        self.config.name,
                "sample_interval":  self.config.sample_interval_seconds,
                "resolution":       f"{frame.shape[1]}x{frame.shape[0]}",
            },
        )

    def _save_frame(self, event: FrameEvent) -> None:
        """Write the frame to disk as a JPEG."""
        success = cv2.imwrite(
            str(event.frame_path),
            event.frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY],
        )
        if not success:
            logger.error("[%s] Failed to write frame: %s", self.config.id, event.frame_path)
        else:
            logger.debug("[%s] Saved frame %d → %s", self.config.id, event.frame_index, event.filename)

    def _dispatch(self, event: FrameEvent) -> None:
        """Call every registered callback with the FrameEvent."""
        for fn in self._callbacks:
            try:
                fn(event)
            except Exception as exc:
                logger.exception(
                    "[%s] Callback %s raised an exception: %s",
                    self.config.id, fn.__name__, exc
                )

    @staticmethod
    def _make_session_id() -> str:
        """Unique ID per sampler instance — groups all frames from one run."""
        return datetime.now(tz=timezone.utc).strftime("session_%Y%m%dT%H%M%S")
