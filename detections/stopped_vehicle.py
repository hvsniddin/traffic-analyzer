"""Vehicles stationary on the carriageway for 10 s or more, outside signal queues.

A track is stationary while its road contact point stays within a small,
box-relative radius of the point where it stopped. Approach lanes queue at
the signal, so stops there only count when the signal stayed green for the
whole stop (a vehicle that does not move off on green is not queuing).
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from classes import BUS, CAR, MOTORCYCLE, TRUCK
except ImportError:
    from detections.classes import BUS, CAR, MOTORCYCLE, TRUCK

VEHICLES = {CAR, BUS, TRUCK, MOTORCYCLE}
MIN_STOP_SEC = 10.0
# Radius as a share of the box width; absorbs detector jitter at any depth.
STILL_RADIUS = 0.25
MIN_STILL_RADIUS_PX = 6.0
# Lanes that end at the stop line; everything else is free-flowing.
SIGNAL_QUEUE_LANES = {"lane_1", "lane_3", "lane_4", "lane_5", "lane_6"}
# A track missing for this long starts a new stop when it reappears.
TRACK_MEMORY_SEC = 3.0
# Another vehicle standing within this many box widths means a queue (behind
# a bus at a stop, or a jam), which the class definition excludes.
QUEUE_RADIUS = 2.5
QUEUE_STILL_SEC = 3.0


def _inside(point, zone) -> bool:
    polygon = np.asarray(zone.polygon, dtype=np.float32)
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def _zone_name(point, scene) -> str | None:
    for name, lane in getattr(scene, "lanes", {}).items():
        if _inside(point, lane):
            return name
    for name, area in getattr(scene, "intersections", {}).items():
        if _inside(point, area):
            return name
    return None


def evaluate_stopped_vehicle_spatial(detections, scene, history, t_sec,
                                     signal_is_red=False, congested=False):
    """Return a mask of vehicles stopped >= MIN_STOP_SEC; starts go to ``history["starts"]``."""
    count = len(detections)
    mask = np.zeros(count, dtype=bool)
    tracks = history.setdefault("tracks", {})
    starts = history.setdefault("starts", {})
    ids = getattr(detections, "tracker_id", None)
    if ids is None or count == 0:
        return mask
    classes = np.asarray(detections.class_id)
    current = []

    for index, tracker_id in enumerate(ids):
        if classes[index] not in VEHICLES or tracker_id < 0:
            continue
        key = int(tracker_id)
        x1, y1, x2, y2 = np.asarray(detections.xyxy[index], dtype=float)
        point = np.array([(x1 + x2) / 2.0, y2])
        radius = max(MIN_STILL_RADIUS_PX, STILL_RADIUS * (x2 - x1))
        zone = _zone_name(point, scene)
        track = tracks.get(key)
        if (track is None or t_sec - track["seen"] > TRACK_MEMORY_SEC
                or np.linalg.norm(point - track["anchor"]) > radius or zone is None):
            track = {"anchor": point, "since": t_sec, "queued": False}
            tracks[key] = track
            starts.pop(("stopped_vehicle", key), None)
        track["seen"] = t_sec
        # Any red phase or queue during the stop means it may just be waiting.
        if zone in SIGNAL_QUEUE_LANES and (signal_is_red or congested):
            track["queued"] = True
        current.append((index, key, point, x2 - x1, zone, track))

    for index, key, point, width, zone, track in current:
        if zone is None or track["queued"] or t_sec - track["since"] < MIN_STOP_SEC:
            continue
        if any(other_key != key and t_sec - other["since"] >= QUEUE_STILL_SEC
               and np.linalg.norm(other_point - point) <= QUEUE_RADIUS * max(width, other_width)
               for _, other_key, other_point, other_width, _, other in current):
            track["queued"] = True
            continue
        mask[index] = True
        starts.setdefault(("stopped_vehicle", key), track["since"])

    for key in [k for k, v in tracks.items() if t_sec - v["seen"] > TRACK_MEMORY_SEC]:
        del tracks[key]
    return mask
