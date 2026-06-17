"""
tools/make_demo_video.py
--------------------------
Generate a synthetic housekeeping-style demo clip so the whole pipeline runs
end-to-end with zero real footage. It draws a labelled "room" with a moving
figure and on-screen captions of the cleaning step, then writes an MP4.

Usage:
    python tools/make_demo_video.py                      # data/demo_housekeeping.mp4, 20s
    python tools/make_demo_video.py --seconds 15 --out data/clip2.mp4

This is for plumbing/demo only. For real behaviour analysis, drop your own
short clips into data/ and point a feed at them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

_STEPS = [
    ("mopping the floor",   (60, 200, 90)),
    ("wiping surfaces",     (200, 160, 60)),
    ("changing linen",      (90, 120, 220)),
    ("emptying bins",       (180, 90, 200)),
]


def make_video(out_path: str, seconds: int = 20, fps: int = 24,
               w: int = 1280, h: int = 720) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    total = seconds * fps
    for i in range(total):
        t = i / total
        frame = np.full((h, w, 3), (40, 40, 45), dtype=np.uint8)
        # "bed" rectangle
        cv2.rectangle(frame, (700, 360), (1180, 620), (180, 175, 165), -1)
        # moving figure
        px = int(120 + t * (w - 360))
        cv2.rectangle(frame, (px, 380), (px + 70, 560), (60, 60, 70), -1)   # body
        cv2.circle(frame, (px + 35, 350), 32, (90, 90, 110), -1)            # head
        # current step caption
        step, color = _STEPS[int(t * len(_STEPS)) % len(_STEPS)]
        cv2.putText(frame, f"Housekeeping demo  t={i/fps:5.1f}s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, f"step: {step}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"Wrote {out_path}  ({seconds}s @ {fps}fps, {w}x{h})")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Generate a synthetic housekeeping demo clip")
    p.add_argument("--out", default="data/demo_housekeeping.mp4")
    p.add_argument("--seconds", type=int, default=20)
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()
    make_video(args.out, args.seconds, args.fps)


if __name__ == "__main__":
    main()
