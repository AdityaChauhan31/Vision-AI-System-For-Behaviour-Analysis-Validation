"""
Ingestion module for VisionAI.
Handles feed configuration, frame sampling, and video ingestion.
"""

from ingestion.config import FeedConfig, FeedsConfig, SourceType
from ingestion.feed_manager import FeedManager
from ingestion.frame_sampler import FrameEvent, FrameSampler

__all__ = [
    "FeedConfig",
    "FeedsConfig",
    "SourceType",
    "FeedManager",
    "FrameEvent",
    "FrameSampler",
]
