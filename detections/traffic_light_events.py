"""Track stop-line stops and completed red-light crossings together."""

import cv2
import numpy as np

try:
    from classes import BUS, CAR, GREENLIGHT, MOTORCYCLE, REDLIGHT, TRUCK
except ImportError:
    from detections.classes import BUS, CAR, GREENLIGHT, MOTORCYCLE, REDLIGHT, TRUCK

VEHICLES = {CAR, BUS, TRUCK, MOTORCYCLE}
STOP_SPEED_PX_PER_SEC = 8.0
STOP_CONFIRM_SEC = 0.5
LINE_TOLERANCE_PX = 20.0
GREEN_SIGNAL_MEMORY_FRAMES = 25
# Allow for small annotation and tracking error at the physical line endpoints.
CROSSING_ENDPOINT_TOLERANCE = 0.03


def _endpoints(line):
    vector = getattr(line, "vector", line)
    try:
        return (np.array([vector.start.x, vector.start.y], dtype=float),
                np.array([vector.end.x, vector.end.y], dtype=float))
    except AttributeError:
        return None


def _side(point, start, end):
    direction = end - start
    offset = point - start
    return float((direction[0] * offset[1] - direction[1] * offset[0]) /
                 np.linalg.norm(direction))


def _near_segment(point, start, end, tolerance, extension=0.05):
    direction = end - start
    squared = float(np.dot(direction, direction))
    if squared == 0:
        return False
    fraction = float(np.dot(point - start, direction) / squared)
    return -extension <= fraction <= 1 + extension and abs(_side(point, start, end)) <= tolerance


def _ends(box, direction):
    x1, y1, x2, y2 = box
    front = np.array([x2 if direction[0] >= 0 else x1,
                      y2 if direction[1] >= 0 else y1])
    rear = np.array([x1 if direction[0] >= 0 else x2,
                     y1 if direction[1] >= 0 else y2])
    return front, rear


def _inside_intersection(point, scene):
    areas = getattr(scene, "intersections", {})
    return any(cv2.pointPolygonTest(np.asarray(area.polygon, dtype=np.float32),
                                    tuple(map(float, point)), False) >= 0
               for area in areas.values())


def evaluate_traffic_light_events(detections, scene, history, t_sec,
                                  tolerance_px=LINE_TOLERANCE_PX):
    """Return red and stop masks; starts are saved under ``event_starts``.

    Red crossings become confirmed when the rear passes the line, but retain
    the front-crossing timestamp. A stop on the line cancels that candidate.
    """
    count = len(detections)
    red_mask = np.zeros(count, dtype=bool)
    stop_mask = np.zeros(count, dtype=bool)
    classes = np.asarray(getattr(detections, "class_id", np.full(count, -1)))
    frame_index = history.get("frame_index", -1) + 1
    history["frame_index"] = frame_index
    green_seen = np.any(classes == GREENLIGHT)
    red_seen = np.any(classes == REDLIGHT)
    if green_seen:
        history["last_green_frame"] = frame_index
    recent_green = frame_index - history.get("last_green_frame", -GREEN_SIGNAL_MEMORY_FRAMES - 1) <= GREEN_SIGNAL_MEMORY_FRAMES
    is_red = bool(red_seen or not recent_green) and not green_seen
    history["signal_is_red"] = is_red
    starts = history.setdefault("event_starts", {})
    tracks = history.setdefault("tracks", {})
    lines = [pair for line in getattr(scene, "stop_lines", {}).values()
             if (pair := _endpoints(line)) is not None]
    ids = getattr(detections, "tracker_id", None)
    if ids is None:
        ids = np.arange(count)
    seen = set()

    for index, tracker_id in enumerate(ids):
        if classes[index] not in VEHICLES:
            continue
        key = int(tracker_id)
        seen.add(key)
        box = np.asarray(detections.xyxy[index], dtype=float)
        center = (box[:2] + box[2:]) / 2
        track = tracks.setdefault(key, {"center": center, "box": box, "time": t_sec,
                                        "direction": None, "candidate": None,
                                        "stopped": False, "stop_since": None,
                                        "red_active": False, "entered": False})
        elapsed = max(t_sec - track["time"], 1e-6)
        movement = center - track["center"]
        speed = float(np.linalg.norm(movement) / elapsed)
        if speed > STOP_SPEED_PX_PER_SEC:
            track["direction"] = movement / np.linalg.norm(movement)
        direction = track["direction"]
        front, rear = _ends(box, direction) if direction is not None else (None, None)
        previous_front = _ends(track["box"], direction)[0] if direction is not None else None
        bottom = np.array([center[0], box[3]])
        # The bottom-center is a proxy for the vehicle's road contact point.
        # A tall box can span the line long after the vehicle has passed it.
        on_line = any(_near_segment(bottom, start, end, tolerance_px)
                      for start, end in lines)

        if not is_red:
            track["stopped"] = False
            track["stop_since"] = None
            track["candidate"] = None
            starts.pop(("stop_line", key), None)
        elif on_line and speed <= STOP_SPEED_PX_PER_SEC:
            if track["stop_since"] is None:
                track["stop_since"] = track["time"] if track["time"] < t_sec else t_sec
            if t_sec - track["stop_since"] >= STOP_CONFIRM_SEC:
                track["stopped"] = True
                track["candidate"] = None
                track["red_active"] = False
                starts.setdefault(("stop_line", key), track["stop_since"])
        else:
            track["stop_since"] = None

        if is_red and not track["stopped"] and not track["red_active"] and direction is not None:
            if track["candidate"] is None:
                for start, end in lines:
                    old_side = _side(previous_front, start, end)
                    new_side = _side(front, start, end)
                    if old_side * new_side <= 0 and new_side != 0:
                        fraction = abs(old_side) / (abs(old_side) + abs(new_side))
                        crossing_point = previous_front + fraction * (front - previous_front)
                        if not _near_segment(crossing_point, start, end, tolerance_px,
                                             extension=CROSSING_ENDPOINT_TOLERANCE):
                            continue
                        track["candidate"] = (start, end, np.sign(new_side),
                                               track["time"] + fraction * (t_sec - track["time"]))
                        break
            if track["candidate"] is not None:
                start, end, target, crossed_at = track["candidate"]
                if target * _side(rear, start, end) > 0:
                    track["red_active"] = True
                    track["entered"] = _inside_intersection(center, scene)
                    starts[("red_light", key)] = crossed_at
                    track["candidate"] = None

        if track["red_active"]:
            inside = _inside_intersection(center, scene)
            if inside:
                track["entered"] = True
            elif track["entered"] and getattr(scene, "intersections", {}):
                track["red_active"] = False
            red_mask[index] = track["red_active"]
        stop_mask[index] = is_red and track["stopped"]
        track["box"] = box
        track["center"] = center
        track["time"] = t_sec

    for key in set(tracks) - seen:
        del tracks[key]
    return red_mask, stop_mask
