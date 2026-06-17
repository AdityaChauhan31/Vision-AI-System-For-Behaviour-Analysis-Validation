"""
tests/test_ingestion.py
-----------------------
Tests for Stage 1 & 2 — runs without any real camera by generating
a synthetic MP4 test video on the fly using OpenCV.

Run:
    pip install pytest opencv-python pydantic pyyaml numpy
    pytest tests/test_ingestion.py -v
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from ingestion.config import FeedConfig, FeedsConfig, SourceType
from ingestion.frame_sampler import FrameEvent, FrameSampler


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def make_test_video(path: Path, num_frames: int = 60, fps: int = 30) -> Path:
    """
    Generate a synthetic .mp4 with colored frames and frame numbers.
    Used instead of a real camera in all tests.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))

    colors = [
        (200, 50,  50),   # blue-ish
        (50,  200, 50),   # green-ish
        (50,  50,  200),  # red-ish
    ]
    for i in range(num_frames):
        frame = np.full((480, 640, 3), colors[i % len(colors)], dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)

    writer.release()
    return path


@pytest.fixture
def test_video(tmp_path: Path) -> Path:
    return make_test_video(tmp_path / "test_clip.mp4", num_frames=90)


@pytest.fixture
def file_feed_config(test_video: Path, tmp_path: Path) -> FeedConfig:
    return FeedConfig(
        id="test_feed",
        name="Test Feed",
        source_type=SourceType.FILE,
        source=str(test_video),
        use_case="housekeeping_validation",
        sample_interval_seconds=0.1,   # fast for tests
        enabled=True,
        output_dir=str(tmp_path / "frames"),
        reconnect_attempts=1,
        reconnect_delay_seconds=0,
        loop=False,
    )


# ─────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────

class TestFeedConfig:

    def test_valid_file_config(self, file_feed_config):
        assert file_feed_config.id == "test_feed"
        assert file_feed_config.source_type == SourceType.FILE
        assert file_feed_config.sample_interval_seconds == 0.1

    def test_valid_webcam_config(self):
        cfg = FeedConfig(
            id="webcam",
            name="Webcam",
            source_type=SourceType.WEBCAM,
            source=0,
            use_case="test",
            output_dir="/tmp/frames",
        )
        assert cfg.cv2_source == 0

    def test_valid_rtsp_config(self):
        cfg = FeedConfig(
            id="cam1",
            name="Cam",
            source_type=SourceType.RTSP,
            source="rtsp://192.168.1.1:554/stream",
            use_case="loitering",
            output_dir="/tmp/frames",
        )
        assert cfg.source_type == SourceType.RTSP

    def test_webcam_rejects_string_source(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="integer device index"):
            FeedConfig(
                id="bad", name="Bad", source_type=SourceType.WEBCAM,
                source="not_an_int", use_case="test", output_dir="/tmp",
            )

    def test_output_path_property(self, file_feed_config, tmp_path):
        assert file_feed_config.output_path == tmp_path / "frames"

    def test_zero_interval_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FeedConfig(
                id="bad", name="Bad", source_type=SourceType.FILE,
                source="/tmp/x.mp4", use_case="test",
                output_dir="/tmp", sample_interval_seconds=0,
            )


class TestFeedsConfig:

    def test_load_yaml(self, tmp_path):
        yaml_content = """
feeds:
  - id: cam1
    name: Test
    source_type: file
    source: /tmp/test.mp4
    use_case: housekeeping
    output_dir: /tmp/frames
"""
        config_file = tmp_path / "feeds.yaml"
        config_file.write_text(yaml_content)
        cfg = FeedsConfig.from_yaml(config_file)
        assert len(cfg.feeds) == 1
        assert cfg.feeds[0].id == "cam1"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            FeedsConfig.from_yaml("/nonexistent/path/feeds.yaml")

    def test_enabled_feeds_filter(self, tmp_path):
        yaml_content = """
feeds:
  - id: cam1
    name: A
    source_type: file
    source: /tmp/a.mp4
    use_case: test
    output_dir: /tmp
    enabled: true
  - id: cam2
    name: B
    source_type: file
    source: /tmp/b.mp4
    use_case: test
    output_dir: /tmp
    enabled: false
"""
        config_file = tmp_path / "feeds.yaml"
        config_file.write_text(yaml_content)
        cfg = FeedsConfig.from_yaml(config_file)
        assert len(cfg.enabled_feeds()) == 1
        assert cfg.enabled_feeds()[0].id == "cam1"

    def test_get_feed_by_id(self, tmp_path):
        yaml_content = """
feeds:
  - id: my_cam
    name: Mine
    source_type: file
    source: /tmp/x.mp4
    use_case: test
    output_dir: /tmp
"""
        config_file = tmp_path / "feeds.yaml"
        config_file.write_text(yaml_content)
        cfg = FeedsConfig.from_yaml(config_file)
        assert cfg.get_feed("my_cam") is not None
        assert cfg.get_feed("missing") is None


# ─────────────────────────────────────────────
# FrameSampler tests
# ─────────────────────────────────────────────

class TestFrameSampler:

    def test_captures_frames_from_file(self, file_feed_config):
        """Sampler should capture multiple frames from a synthetic video."""
        events: list[FrameEvent] = []
        sampler = FrameSampler(file_feed_config)
        sampler.add_callback(events.append)

        sampler.start()
        # Give it up to 5 seconds to process the short clip
        deadline = time.monotonic() + 5.0
        while sampler.is_running and time.monotonic() < deadline:
            time.sleep(0.1)
        sampler.stop()

        assert len(events) >= 3, f"Expected ≥3 frames, got {len(events)}"

    def test_frame_event_fields(self, file_feed_config):
        """FrameEvent must have all required fields populated correctly."""
        events: list[FrameEvent] = []
        sampler = FrameSampler(file_feed_config)
        sampler.add_callback(events.append)

        sampler.start()
        deadline = time.monotonic() + 3.0
        while len(events) < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        sampler.stop()

        assert len(events) >= 1
        ev = events[0]

        assert ev.feed_id   == "test_feed"
        assert ev.use_case  == "housekeeping_validation"
        assert ev.frame_index == 1
        assert ev.session_id.startswith("session_")
        assert ev.timestamp_utc is not None
        assert ev.frame is not None
        assert ev.frame.ndim == 3               # H x W x 3
        assert "resolution" in ev.metadata

    def test_frames_saved_to_disk(self, file_feed_config):
        """Each FrameEvent's frame_path should exist on disk."""
        events: list[FrameEvent] = []
        sampler = FrameSampler(file_feed_config)
        sampler.add_callback(events.append)

        sampler.start()
        deadline = time.monotonic() + 3.0
        while len(events) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        sampler.stop()

        for ev in events:
            assert ev.frame_path.exists(), f"Frame not saved: {ev.frame_path}"
            assert ev.frame_path.suffix == ".jpg"
            assert ev.frame_path.stat().st_size > 0

    def test_resize_applied(self, test_video, tmp_path):
        """If resize is configured, the saved frame should match the target size."""
        from ingestion.config import ResizeConfig
        cfg = FeedConfig(
            id="resize_test",
            name="Resize Test",
            source_type=SourceType.FILE,
            source=str(test_video),
            use_case="test",
            sample_interval_seconds=0.1,
            output_dir=str(tmp_path / "frames"),
            reconnect_attempts=1,
            reconnect_delay_seconds=0,
            resize=ResizeConfig(width=320, height=240),
        )

        events: list[FrameEvent] = []
        sampler = FrameSampler(cfg)
        sampler.add_callback(events.append)

        sampler.start()
        deadline = time.monotonic() + 3.0
        while len(events) < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        sampler.stop()

        assert len(events) >= 1
        h, w = events[0].frame.shape[:2]
        assert w == 320 and h == 240

    def test_multiple_callbacks(self, file_feed_config):
        """Multiple callbacks should all receive the same events."""
        bucket_a: list[FrameEvent] = []
        bucket_b: list[FrameEvent] = []

        sampler = FrameSampler(file_feed_config)
        sampler.add_callback(bucket_a.append)
        sampler.add_callback(bucket_b.append)

        sampler.start()
        deadline = time.monotonic() + 3.0
        while len(bucket_a) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        sampler.stop()

        assert len(bucket_a) == len(bucket_b)
        assert all(a.frame_index == b.frame_index for a, b in zip(bucket_a, bucket_b))

    def test_callback_exception_does_not_crash_sampler(self, file_feed_config):
        """A crashing callback must not kill the sampler or affect other callbacks."""
        good_events: list[FrameEvent] = []

        def bad_callback(event):
            raise RuntimeError("Simulated downstream failure")

        sampler = FrameSampler(file_feed_config)
        sampler.add_callback(bad_callback)
        sampler.add_callback(good_events.append)

        sampler.start()
        deadline = time.monotonic() + 3.0
        while len(good_events) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        sampler.stop()

        assert len(good_events) >= 2   # good callback still received events

    def test_frame_indices_are_sequential(self, file_feed_config):
        events: list[FrameEvent] = []
        sampler = FrameSampler(file_feed_config)
        sampler.add_callback(events.append)

        sampler.start()
        deadline = time.monotonic() + 3.0
        while len(events) < 5 and time.monotonic() < deadline:
            time.sleep(0.05)
        sampler.stop()

        indices = [e.frame_index for e in events]
        assert indices == list(range(1, len(indices) + 1)), f"Non-sequential: {indices}"

    def test_all_events_share_session_id(self, file_feed_config):
        events: list[FrameEvent] = []
        sampler = FrameSampler(file_feed_config)
        sampler.add_callback(events.append)

        sampler.start()
        deadline = time.monotonic() + 3.0
        while len(events) < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        sampler.stop()

        session_ids = {e.session_id for e in events}
        assert len(session_ids) == 1, "All frames must share the same session_id"
