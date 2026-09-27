"""Vehicles moving against the drawn travel direction of their lane.

Headings come from a ~1 s displacement of the road contact point so box
jitter on slow or queued vehicles cannot flip the sign. The event starts
when the opposing motion is first seen and must hold for MIN_WRONG_SEC.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

try:
    from classes import BUS, CAR, MOTORCYCLE, TRUCK
except ImportError:
    from detections.classes import BUS, CAR, MOTORCYCLE, TRUCK

VEHICLES = {CAR, BUS, TRUCK, MOTORCYCLE}
HEADING_WINDOW_SEC = 1.0
# The displacement over the window must exceed this share of the box width.
MIN_MOVE_RATIO = 0.35
# cos(angle) to the lane direction; -0.5 means more than 120 degrees off.
OPPOSING_COS = -0.5
MIN_WRONG_SEC = 1.5
TRACK_MEMORY_SEC = 3.0


def _lane_of(point, scene) -> str | None:
    for name, lane in getattr(scene, "lanes", {}).items():
        polygon = np.asarray(lane.polygon, dtype=np.float32)
        if cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0:
            return name
    return None


def evaluate_wrong_way_spatial(detections, scene, history, t_sec):
    """Return a mask of wrong-way vehicles; starts go to ``history["starts"]``."""
    count = len(detections)
    mask = np.zeros(count, dtype=bool)
    tracks = history.setdefault("tracks", {})
    starts = history.setdefault("starts", {})
    ids = getattr(detections, "tracker_id", None)
    directions = getattr(scene, "lane_directions", {})
    if ids is None or count == 0 or not directions:
        return mask
    classes = np.asarray(detections.class_id)

    for index, tracker_id in enumerate(ids):
        if classes[index] not in VEHICLES or tracker_id < 0:
            continue
        key = int(tracker_id)
        x1, y1, x2, y2 = np.asarray(detections.xyxy[index], dtype=float)
        point = np.array([(x1 + x2) / 2.0, y2])
        track = tracks.setdefault(key, {"points": deque(), "since": None})
        track["points"].append((t_sec, point))
        while len(track["points"]) > 2 and t_sec - track["points"][1][0] >= HEADING_WINDOW_SEC:
            track["points"].popleft()
        track["seen"] = t_sec

        old_t, old_point = track["points"][0]
        lane = _lane_of(point, scene)
        opposing = False
        if lane in directions and t_sec - old_t >= 0.5 * HEADING_WINDOW_SEC:
            move = point - old_point
            distance = float(np.linalg.norm(move))
            if distance >= MIN_MOVE_RATIO * (x2 - x1):
                opposing = float(np.dot(move / distance, directions[lane])) <= OPPOSING_COS
        if opposing:
            if track["since"] is None:
                track["since"] = old_t
            if t_sec - track["since"] >= MIN_WRONG_SEC:
                mask[index] = True
                starts.setdefault(("wrong_way", key), track["since"])
        else:
            # Brief gaps inside a confirmed event are bridged by the caller.
            track["since"] = None
            starts.pop(("wrong_way", key), None)

    for key in [k for k, v in tracks.items() if t_sec - v["seen"] > TRACK_MEMORY_SEC]:
        del tracks[key]
    return mask
