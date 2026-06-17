"""
ingestion/feed_manager.py
-------------------------
Manages ALL enabled feeds from feeds.yaml.
Starts each FrameSampler in its own thread, handles graceful shutdown,
and provides a single place to register callbacks for all feeds.

Usage
-----
    from ingestion.feed_manager import FeedManager

    def my_pipeline(event: FrameEvent) -> None:
        print(f"Got frame {event.frame_index} from {event.feed_id}")

    manager = FeedManager("config/feeds.yaml")
    manager.add_global_callback(my_pipeline)
    manager.start_all()

    # ... your app runs here ...

    manager.stop_all()
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Union

from .config import FeedConfig, FeedsConfig, SourceType
from .frame_sampler import FrameCallback, FrameEvent, FrameSampler

logger = logging.getLogger(__name__)


class FeedManager:
    """
    Lifecycle manager for all video feeds.

    - Loads feeds.yaml (or any YAML at the given path)
    - Creates one FrameSampler per enabled feed
    - Propagates global and per-feed callbacks
    - Handles SIGINT / SIGTERM for clean shutdown
    """

    def __init__(self, config_path: Union[str, Path] = "config/feeds.yaml") -> None:
        self._config      = FeedsConfig.from_yaml(config_path)
        self._samplers:   dict[str, FrameSampler] = {}
        self._global_cbs: list[FrameCallback]     = []
        self._global_session_end_cbs: list = []
        self._running     = False

        self._build_samplers()

    # ── public API ────────────────────────────────────────

    def add_global_callback(self, fn: FrameCallback) -> None:
        """
        Register a callback that receives FrameEvents from ALL feeds.
        Call before start_all() so no frames are missed.
        """
        self._global_cbs.append(fn)
        # Also register on already-built samplers (safe to call multiple times)
        for sampler in self._samplers.values():
            sampler.add_callback(fn)

    def add_global_session_end_callback(self, fn) -> None:
        """Register a callback called with session_id when ANY feed's session ends."""
        self._global_session_end_cbs.append(fn)
        for sampler in self._samplers.values():
            sampler.add_session_end_callback(fn)

    def add_feed_callback(self, feed_id: str, fn: FrameCallback) -> None:
        """Register a callback for ONE specific feed only."""
        sampler = self._samplers.get(feed_id)
        if sampler is None:
            raise KeyError(f"No enabled feed with id='{feed_id}'. Available: {list(self._samplers)}")
        sampler.add_callback(fn)

    def start_all(self, block: bool = False) -> None:
        """
        Start all enabled feed samplers.

        Args:
            block: If True, blocks until KeyboardInterrupt or SIGTERM,
                   then automatically calls stop_all(). Useful for
                   running the manager as a standalone process.
        """
        if not self._samplers:
            logger.warning("No enabled feeds found in config. Nothing to start.")
            return

        logger.info("Starting %d feed(s)...", len(self._samplers))
        for sampler in self._samplers.values():
            sampler.start()

        self._running = True

        if block:
            self._register_signal_handlers()
            try:
                while self._running:
                    time.sleep(1)
                    self._log_status()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received.")
            finally:
                self.stop_all()

    def stop_all(self) -> None:
        """Stop all running samplers gracefully."""
        logger.info("Stopping all feeds...")
        for sampler in self._samplers.values():
            sampler.stop()
        self._running = False
        logger.info("All feeds stopped.")

    def status(self) -> dict[str, bool]:
        """Return {feed_id: is_running} for every feed."""
        return {fid: s.is_running for fid, s in self._samplers.items()}

    def reload_config(self, config_path: Union[str, Path]) -> None:
        """
        Hot-reload feeds.yaml without restarting the process.
        Stops all current samplers, loads new config, restarts.
        Useful for adding/removing feeds at runtime.
        """
        logger.info("Reloading config from %s ...", config_path)
        was_running = self._running
        self.stop_all()
        self._config = FeedsConfig.from_yaml(config_path)
        self._samplers.clear()
        self._build_samplers()
        # Re-attach global callbacks
        for fn in self._global_cbs:
            for sampler in self._samplers.values():
                sampler.add_callback(fn)
        if was_running:
            self.start_all()

    # ── internal ──────────────────────────────────────────

    def _build_samplers(self) -> None:
        enabled = self._config.enabled_feeds()
        logger.info("Found %d enabled feed(s) in config.", len(enabled))

        for feed_cfg in enabled:
            if feed_cfg.source_type == SourceType.FILE:
                source_path = Path(feed_cfg.source)
                if source_path.is_dir():
                    video_files = sorted(source_path.glob("*.mp4"))
                    if not video_files:
                        logger.warning(
                            "[%s] No .mp4 files found in source directory: %s",
                            feed_cfg.id,
                            source_path,
                        )
                        continue

                    logger.info(
                        "[%s] Expanding directory source %s into %d file feeds.",
                        feed_cfg.id,
                        source_path,
                        len(video_files),
                    )

                    for file_path in video_files:
                        sub_id = f"{feed_cfg.id}_{file_path.stem}"
                        sub_cfg = FeedConfig(
                            **{
                                **feed_cfg.model_dump(),
                                "id": sub_id,
                                "source": str(file_path),
                            }
                        )
                        sampler = FrameSampler(sub_cfg)
                        for fn in self._global_cbs:
                            sampler.add_callback(fn)
                        for fn in self._global_session_end_cbs:
                            sampler.add_session_end_callback(fn)
                        self._samplers[sub_id] = sampler
                        logger.debug(
                            "Built sampler for feed: %s (%s)",
                            sub_cfg.id,
                            sub_cfg.source_type.value,
                        )
                    continue

            sampler = FrameSampler(feed_cfg)
            # Attach any callbacks registered before start
            for fn in self._global_cbs:
                sampler.add_callback(fn)
            for fn in self._global_session_end_cbs:
                sampler.add_session_end_callback(fn)
            self._samplers[feed_cfg.id] = sampler
            logger.debug("Built sampler for feed: %s (%s)", feed_cfg.id, feed_cfg.source_type.value)

    def _log_status(self) -> None:
        for fid, running in self.status().items():
            if not running:
                logger.warning("[%s] Sampler is no longer running!", fid)

    def _register_signal_handlers(self) -> None:
        def _handler(signum, frame):  # noqa: ARG001
            logger.info("Signal %d received — shutting down.", signum)
            self._running = False

        signal.signal(signal.SIGINT,  _handler)
        signal.signal(signal.SIGTERM, _handler)
