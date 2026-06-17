"""
main.py — Vision AI Platform entry point (Stages 1–7)

Wires the full pipeline:
  ingestion (1-2) → perception: face+zone+VLM (3-4) → rules engine (6) → alerts (7)

Run modes:
  python main.py                         # all enabled feeds from config/feeds.yaml
  python main.py --video data/clip.mp4   # single ad-hoc file, no config edits
  python main.py --vlm gemini            # pick VLM provider (default: env or mock)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Create log dir BEFORE logging is configured (FileHandler opens the file eagerly).
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/pipeline.log", mode="a")],
)
logger = logging.getLogger(__name__)

from ingestion import FeedConfig, FeedManager, FrameSampler
from ingestion.config import SourceType
from perception import PerceptionPipeline, VLMResult
from rules import RulesEngine


def print_vlm_result(result: VLMResult) -> None:
    alert = " 🚨 ANOMALY" if result.anomaly_detected else ""
    logger.info(
        "Frame %d | behaviors=%s | person=%s | zone=%s | %.0fms%s",
        result.frame_index, result.behaviors_detected, result.person_id,
        result.zone_label or "none", result.latency_ms, alert,
    )
    if result.reasoning:
        logger.info("   → %s", result.reasoning)


def parse_args():
    p = argparse.ArgumentParser(description="Vision AI Platform — Stages 1–7")
    p.add_argument("--video",    default=None, help="Path to a single demo MP4 (skips feeds.yaml)")
    p.add_argument("--config",   default="config/feeds.yaml")
    p.add_argument("--interval", type=float, default=5.0, help="Sample interval (s) for --video")
    p.add_argument("--use-case", default="housekeeping_validation")
    p.add_argument("--vlm",      default=None, help="mock|gemini|openai|anthropic|huggingface")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def build_pipeline(vlm_provider):
    perception = PerceptionPipeline.from_config(
        zones_config="config/zones.yaml",
        identities_config="config/identities.yaml",
        use_cases_config="config/use_cases.yaml",
        vlm_provider=vlm_provider,
    )
    rules = RulesEngine.from_config("config/use_cases.yaml")

    perception.add_result_callback(print_vlm_result)
    perception.add_result_callback(rules.evaluate)

    def on_session_end(session_id: str) -> None:
        rules.finalize(session_id)        # → completion alerts + verdict
        perception.end_session(session_id)

    return perception, rules, on_session_end


def main():
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    perception, rules, on_session_end = build_pipeline(args.vlm)

    if args.video:
        cfg = FeedConfig(
            id="demo_feed", name="Demo Feed",
            source_type=SourceType.FILE, source=args.video,
            use_case=args.use_case, sample_interval_seconds=args.interval,
            output_dir="frames/demo", reconnect_attempts=1, reconnect_delay_seconds=0,
        )
        sampler = FrameSampler(cfg)
        sampler.add_callback(perception.process_frame)
        sampler.add_session_end_callback(on_session_end)
        sampler.start()
        try:
            while sampler.is_running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            sampler.stop()
    else:
        manager = FeedManager(args.config)
        manager.add_global_callback(perception.process_frame)
        manager.add_global_session_end_callback(on_session_end)
        manager.start_all(block=True)

    logger.info("Pipeline finished. Alerts → logs/alerts.jsonl | Verdicts → logs/sessions/")


if __name__ == "__main__":
    main()
