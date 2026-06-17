"""
tools/define_zones.py
----------------------
Interactive tool to define zone polygons for any video.

HOW TO USE:
    python tools/define_zones.py --video data/my_housekeeping_video.mp4

WHAT IT DOES:
    1. Opens the first frame of your video in a window
    2. You click the corners of the zone you want to define
    3. Press ENTER when done — it prints the polygon coordinates
    4. Paste those coordinates into config/zones.yaml

CONTROLS:
    Left-click     → add a point
    Right-click    → remove last point
    ENTER          → confirm polygon and print coordinates
    ESC            → cancel
    R              → reset all points

FOR DEMO HOUSEKEEPING VIDEOS:
    If you just want the full frame as a zone (simplest approach),
    set mode: full_frame in zones.yaml — no need to run this tool.
    Use this tool only if you want to define a specific region
    (e.g., just the bathroom, or just the bed area).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Points collected so far
_points: list[tuple[int, int]] = []
_window_name = "Define Zone — Click corners, ENTER to confirm, R to reset"


def _mouse_callback(event, x, y, flags, param):
    global _points
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append((x, y))
        print(f"  Added point {len(_points)}: ({x}, {y})")


def _draw_state(frame: np.ndarray) -> np.ndarray:
    """Draw current points and polygon on the frame."""
    display = frame.copy()

    if len(_points) >= 2:
        pts = np.array(_points, dtype=np.int32)
        cv2.polylines(display, [pts], isClosed=len(_points) >= 3, color=(0, 255, 0), thickness=2)
        if len(_points) >= 3:
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
            display = cv2.addWeighted(overlay, 0.25, display, 0.75, 0)

    for i, (x, y) in enumerate(_points):
        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(display, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    instructions = [
        "Left-click: add point",
        "Right-click: remove last",
        "ENTER: confirm",
        "R: reset",
        "ESC: cancel",
        f"Points: {len(_points)}",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(display, text, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return display


def define_zone(video_path: str, zone_id: str, feed_id: str) -> None:
    global _points
    _points = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Cannot read first frame from video")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"\nVideo resolution: {w} x {h}")
    print("Click to define zone polygon corners.")
    print("Press ENTER when done.\n")

    cv2.namedWindow(_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_window_name, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(_window_name, _mouse_callback)

    while True:
        display = _draw_state(frame)
        cv2.imshow(_window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 or key == 10:    # ENTER
            if len(_points) < 3:
                print("Need at least 3 points to define a polygon.")
                continue
            break
        elif key == ord('r') or key == ord('R'):
            _points = []
            print("Reset — start clicking again.")
        elif key == 2:                 # right arrow (some systems)
            if _points:
                removed = _points.pop()
                print(f"  Removed point: {removed}")
        elif key == 27:                # ESC
            print("Cancelled.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    # Output the YAML block to paste into zones.yaml
    polygon_yaml = "\n".join(f"      - [{x}, {y}]" for x, y in _points)

    print("\n" + "="*60)
    print("COPY THIS INTO config/zones.yaml:")
    print("="*60)
    print(f"""
  - id: {zone_id}
    name: "Zone defined from {Path(video_path).name}"
    feed_id: {feed_id}
    mode: polygon
    polygon:
{polygon_yaml}
    label: {zone_id}
""")
    print("="*60)
    print(f"\nPolygon coordinates (JSON): {json.dumps(_points)}")


def main():
    parser = argparse.ArgumentParser(description="Define zone polygons for Vision AI")
    parser.add_argument("--video",   required=True, help="Path to video file")
    parser.add_argument("--zone-id", default="my_zone", help="Zone identifier (default: my_zone)")
    parser.add_argument("--feed-id", default="demo_feed",  help="Feed ID this zone belongs to")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    define_zone(args.video, args.zone_id, args.feed_id)


if __name__ == "__main__":
    main()
