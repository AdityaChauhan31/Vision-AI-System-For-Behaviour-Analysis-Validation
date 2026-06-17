"""
perception/zone_checker.py
---------------------------
Stage 3b — Zone / Region-of-Interest Check

Responsibilities:
  1. Load zone polygon definitions from zones.yaml
  2. For each frame, find which zone(s) the detected person is in
  3. Check whether this person (known or unknown) is allowed in that zone
  4. Return ZoneDetection with zone_id, zone_label, is_restricted

HOW ZONE CHECKING WORKS:
  - Each zone has a polygon (list of [x,y] corner coordinates)
  - We check if the center of the detected face bbox is inside the polygon
  - If no face was detected, we use the frame center as the test point
    (reasonable for single-person scenarios like housekeeping)
  - mode: full_frame means the entire frame is one zone — used for demo videos

DEMO VIDEO NOTE:
  For downloaded housekeeping videos, use mode: full_frame in zones.yaml.
  This means any person visible in the frame is automatically "in zone".
  The housekeeping use case doesn't need precise zone boundaries —
  if the camera covers the room, being in frame = being in the room.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .enriched_frame import FaceDetection, ZoneDetection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Zone record
# ─────────────────────────────────────────────

class Zone:
    def __init__(self, data: dict) -> None:
        self.id       = data["id"]
        self.name     = data["name"]
        self.feed_id  = data["feed_id"]
        self.mode     = data.get("mode", "polygon")   # "polygon" | "full_frame"
        self.label    = data.get("label", self.id)
        raw_polygon   = data.get("polygon", [])
        self.polygon: Optional[np.ndarray] = (
            np.array(raw_polygon, dtype=np.int32) if raw_polygon else None
        )

    def contains_point(self, x: int, y: int, frame_shape: tuple) -> bool:
        """
        Returns True if point (x, y) is inside this zone.
        frame_shape = (height, width, channels) from frame.shape
        """
        if self.mode == "full_frame":
            return True   # entire frame is the zone

        if self.polygon is None or len(self.polygon) < 3:
            logger.warning("Zone %s has no valid polygon — treating as full_frame", self.id)
            return True

        # cv2.pointPolygonTest returns:
        #   +ve distance → inside
        #   0            → on boundary
        #   -ve distance → outside
        result = cv2.pointPolygonTest(self.polygon, (float(x), float(y)), measureDist=False)
        return result >= 0


# ─────────────────────────────────────────────
# Zone Checker
# ─────────────────────────────────────────────

class ZoneChecker:
    """
    Loads zones.yaml and for each incoming frame determines:
      1. Which zone the person is in
      2. Whether they are authorized to be there

    Authorization logic:
      - If person is known (face recognized) → check their allowed/restricted zones
      - If person is unknown → check use_case default policy
        (for housekeeping: unknown people in the room still trigger behavior checks)

    Usage
    -----
        checker = ZoneChecker("config/zones.yaml")
        zone = checker.check(frame, face_detection, feed_id, person_config)
        print(zone.is_restricted)   # True / False
    """

    def __init__(self, config_path: str = "config/zones.yaml") -> None:
        self._config_path = Path(config_path)
        self._zones_by_feed: dict[str, list[Zone]] = {}
        self._load_config()

    # ── public API ────────────────────────────────────────

    def check(
        self,
        frame:          np.ndarray,
        face:           Optional[FaceDetection],
        feed_id:        str,
        allowed_zones:  list[str] = [],
        restricted_zones: list[str] = [],
    ) -> ZoneDetection:
        """
        Check which zone the person is in and whether they are authorized.

        Args:
            frame:            The current video frame (H x W x 3)
            face:             FaceDetection from Stage 3a (may be anonymous)
            feed_id:          Which camera feed this frame is from
            allowed_zones:    Zone labels this person is allowed in
            restricted_zones: Zone labels this person is NOT allowed in

        Returns:
            ZoneDetection with zone info and restriction flag
        """
        zones_for_feed = self._zones_by_feed.get(feed_id, [])

        if not zones_for_feed:
            logger.debug("No zones configured for feed '%s' — skipping zone check", feed_id)
            return ZoneDetection(
                zone_id=None, zone_label=None,
                is_inside=False, is_restricted=False,
                polygon_tested=None,
            )

        # Determine the test point (where is the person?)
        test_x, test_y = self._get_test_point(frame, face)

        # Check each zone for this feed
        for zone in zones_for_feed:
            if zone.contains_point(test_x, test_y, frame.shape):
                is_restricted = self._is_restricted(
                    zone.label, allowed_zones, restricted_zones, face
                )
                logger.debug(
                    "Person at (%d,%d) is in zone '%s' (restricted=%s)",
                    test_x, test_y, zone.label, is_restricted
                )
                return ZoneDetection(
                    zone_id=zone.id,
                    zone_label=zone.label,
                    is_inside=True,
                    is_restricted=is_restricted,
                    polygon_tested=zone.polygon.tolist() if zone.polygon is not None else None,
                )

        # Not in any defined zone
        return ZoneDetection(
            zone_id=None, zone_label=None,
            is_inside=False, is_restricted=False,
            polygon_tested=None,
        )

    def get_zones_for_feed(self, feed_id: str) -> list[Zone]:
        return self._zones_by_feed.get(feed_id, [])

    def draw_zones(self, frame: np.ndarray, feed_id: str) -> np.ndarray:
        """
        Draw zone polygons onto a frame (for debugging / visualization).
        Returns annotated copy of the frame.
        """
        annotated = frame.copy()
        zones = self._zones_by_feed.get(feed_id, [])

        for zone in zones:
            if zone.mode == "full_frame":
                h, w = frame.shape[:2]
                pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.int32)
            elif zone.polygon is not None:
                pts = zone.polygon
            else:
                continue

            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(
                annotated, zone.label,
                tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        return annotated

    # ── internals ─────────────────────────────────────────

    def _load_config(self) -> None:
        if not self._config_path.exists():
            logger.warning("zones.yaml not found at %s — zone checking disabled", self._config_path)
            return

        with self._config_path.open() as f:
            raw = yaml.safe_load(f)

        zones_raw = raw.get("zones", [])
        for z_data in zones_raw:
            zone = Zone(z_data)
            feed_id = zone.feed_id
            if feed_id not in self._zones_by_feed:
                self._zones_by_feed[feed_id] = []
            self._zones_by_feed[feed_id].append(zone)

        total = sum(len(v) for v in self._zones_by_feed.values())
        logger.info("Loaded %d zone(s) across %d feed(s)", total, len(self._zones_by_feed))

    def _get_test_point(
        self,
        frame: np.ndarray,
        face: Optional[FaceDetection],
    ) -> tuple[int, int]:
        """
        Determine the (x, y) point to test for zone containment.

        Priority:
          1. Center of detected face bounding box
          2. Frame center (fallback for anonymous / no face detected)
        """
        if face and face.bbox:
            x, y, w, h = face.bbox
            return int(x + w / 2), int(y + h / 2)

        # Fallback: use frame center
        h, w = frame.shape[:2]
        return w // 2, h // 2

    def _is_restricted(
        self,
        zone_label: str,
        allowed_zones: list[str],
        restricted_zones: list[str],
        face: Optional[FaceDetection],
    ) -> bool:
        """
        Determine if this person should be in this zone.

        Rules (in priority order):
          1. If zone is in person's restricted_zones list → restricted
          2. If allowed_zones is defined and zone is NOT in it → restricted
          3. If person is unknown and zone is restricted → restricted
          4. Otherwise → allowed
        """
        # Explicitly restricted zone for this person
        if zone_label in restricted_zones:
            return True

        # Allowed list is defined but this zone isn't in it
        if allowed_zones and zone_label not in allowed_zones:
            return True

        return False
