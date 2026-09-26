"""Detect stationary or crawling queues in lanes and intersections."""

import argparse
import json
from collections import deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

try:
    from alignment import align_capture
    from classes import BUS, CAR, MOTORCYCLE, TRUCK
    from scene import SceneGeometry
except ImportError:
    from detections.alignment import align_capture
    from detections.classes import BUS, CAR, MOTORCYCLE, TRUCK
    from detections.scene import SceneGeometry


VEHICLE_CLASSES = {BUS, CAR, MOTORCYCLE, TRUCK}
SPEED_WINDOW_SEC = 0.8
MAX_SPEED_BOX_HEIGHTS_PER_SEC = 0.6
MIN_VEHICLES = 2
SLOW_FRACTION = 0.75
CONFIRM_SEC = 2.0
CLEAR_GRACE_SEC = 1.0
ROAD_QUEUE_MIN_VEHICLES = 6
ROAD_QUEUE_SLOW_FRACTION = 0.65
ROAD_QUEUE_MIN_WIDTH_FRACTION = 0.25
ROAD_QUEUE_MIN_HEIGHT_FRACTION = 0.12
INTERSECTION_MIN_VEHICLES = 3


def _direction_groups(scene):
    """Group lanes travelling in roughly the same direction."""
    groups = []
    for name, direction in scene.lane_directions.items():
        for group in groups:
            if np.dot(direction, group["direction"]) >= 0.7:
                group["lanes"].append(name)
                break
        else:
            groups.append({"direction": direction, "lanes": [name]})
    return [group["lanes"] for group in groups]


def evaluate_congestion_spatial(detections, scene, track_history, t_sec):
    """Return congestion status, slow track IDs, and affected scene zones.

    Speed is measured in vehicle box heights per second so the threshold works
    for vehicles near and far from the camera. Only stable tracked vehicles
    contribute to the decision; an empty lane is not evidence of congestion.
    """
    lane_vehicles = {name: set() for name in scene.lanes}
    intersection_vehicles = {name: set() for name in scene.intersections}
    slow_ids = set()
    road_vehicles = {}
    congested_zones = set()
    visible_ids = set()
    ids = getattr(detections, "tracker_id", None)
    if ids is not None:
        for index, tracker_id in enumerate(ids):
            if detections.class_id[index] not in VEHICLE_CLASSES or tracker_id is None:
                continue
            key = int(tracker_id)
            visible_ids.add(key)
            x1, y1, x2, y2 = map(float, detections.xyxy[index])
            height = max(y2 - y1, 1.0)
            point = np.array([(x1 + x2) / 2.0, y2], dtype=float)
            history = track_history.setdefault(key, deque())
            history.append((t_sec, point, height))
            while len(history) > 1 and t_sec - history[1][0] >= SPEED_WINDOW_SEC:
                history.popleft()
            elapsed = t_sec - history[0][0]
            if elapsed >= SPEED_WINDOW_SEC and elapsed > 0:
                distance = float(np.linalg.norm(point - history[0][1]))
                scale = max((height + history[0][2]) / 2.0, 1.0)
                if distance / scale / elapsed <= MAX_SPEED_BOX_HEIGHTS_PER_SEC:
                    slow_ids.add(key)
            for name, lane in scene.lanes.items():
                if cv2.pointPolygonTest(
                    np.asarray(lane.polygon, dtype=np.float32), tuple(point), False
                ) >= 0:
                    lane_vehicles[name].add(key)
            for name, intersection in scene.intersections.items():
                if cv2.pointPolygonTest(
                    np.asarray(intersection.polygon, dtype=np.float32), tuple(point), False
                ) >= 0:
                    intersection_vehicles[name].add(key)
            road = getattr(scene, "road_zone", None)
            if road is not None and cv2.pointPolygonTest(
                np.asarray(road.polygon, dtype=np.float32), tuple(point), False
            ) >= 0:
                road_vehicles[key] = point

    for key in set(track_history) - visible_ids:
        track_history.pop(key)

    for name, members in intersection_vehicles.items():
        if len(members) >= INTERSECTION_MIN_VEHICLES and (
            len(members & slow_ids) / len(members) >= SLOW_FRACTION
        ):
            congested_zones.add(("intersection", name))

    for names in _direction_groups(scene):
        if not names or len(set().union(*(lane_vehicles[name] for name in names))) < MIN_VEHICLES:
            continue
        if all(
            lane_vehicles[name]
            and len(lane_vehicles[name] & slow_ids) / len(lane_vehicles[name]) >= SLOW_FRACTION
            for name in names
        ):
            congested_zones.update(("lane", name) for name in names)

    # The annotated lane polygons stop near the intersection. A long queue can
    # extend beyond them, so also detect a broad, mostly stationary road queue.
    road_slow_ids = set(road_vehicles) & slow_ids
    broad_queue = False
    if len(road_slow_ids) >= ROAD_QUEUE_MIN_VEHICLES and (
        len(road_slow_ids) / len(road_vehicles) >= ROAD_QUEUE_SLOW_FRACTION
    ):
        points = np.asarray([road_vehicles[key] for key in road_slow_ids])
        span = np.ptp(points, axis=0)
        if (span[0] >= ROAD_QUEUE_MIN_WIDTH_FRACTION * scene.frame_width
                and span[1] >= ROAD_QUEUE_MIN_HEIGHT_FRACTION * scene.frame_height):
            broad_queue = True
            for name, members in lane_vehicles.items():
                if len(members) >= MIN_VEHICLES and (
                    len(members & slow_ids) / len(members) >= SLOW_FRACTION
                ):
                    congested_zones.add(("lane", name))
            if not congested_zones:
                membership = {
                    **{("lane", name): len(members & slow_ids)
                       for name, members in lane_vehicles.items()},
                    **{("intersection", name): len(members & slow_ids)
                       for name, members in intersection_vehicles.items()},
                }
                if membership and max(membership.values()) > 0:
                    congested_zones.add(max(membership, key=membership.get))
    return bool(congested_zones) or broad_queue, slow_ids, congested_zones


def update_congestion_event(state, congested, t_sec, min_duration=0.0):
    """Confirm a queue, then close it when traffic has cleared for a second."""
    events = []
    if congested:
        state["clear_since"] = None
        if state["candidate_since"] is None:
            state["candidate_since"] = t_sec
        if state["active_start"] is None and t_sec - state["candidate_since"] >= CONFIRM_SEC:
            state["active_start"] = max(0.0, state["candidate_since"] - SPEED_WINDOW_SEC)
    else:
        if state["active_start"] is None:
            state["candidate_since"] = None
        else:
            if state["clear_since"] is None:
                state["clear_since"] = t_sec
            if t_sec - state["clear_since"] >= CLEAR_GRACE_SEC:
                start, end = state["active_start"], state["clear_since"]
                if end > start and end - start >= min_duration:
                    events.append([round(start, 2), round(end, 2), "congestion"])
                state.update(candidate_since=None, active_start=None, clear_since=None)
    return events


def annotate_frame(frame, detections, slow_ids, congested_zones, annotation_scene, annotators):
    zones = {
        **{("lane", name): zone for name, zone in annotation_scene.lanes.items()},
        **{("intersection", name): zone for name, zone in annotation_scene.intersections.items()},
    }
    ids = getattr(detections, "tracker_id", None)
    if ids is not None:
        slow = detections[np.isin(ids, list(slow_ids))]
        if len(slow):
            frame = annotators["box"].annotate(scene=frame, detections=slow)
            frame = annotators["label"].annotate(
                scene=frame, detections=slow,
                labels=[f"#{tracker_id} SLOW" for tracker_id in slow.tracker_id],
            )
    for key, zone in sorted(zones.items(), key=lambda item: item[0] in congested_zones):
        polygon = np.asarray(zone.polygon, dtype=np.int32)
        active = key in congested_zones
        color = (0, 0, 255) if active else (0, 255, 0)
        cv2.polylines(frame, [polygon], True, color, 7 if active else 3, cv2.LINE_AA)
    for key in congested_zones:
        zone = zones.get(key)
        if zone is None:
            continue
        polygon = np.asarray(zone.polygon, dtype=np.int32)
        label = f"{key[1].upper()} CONGESTION"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_width, text_height = cv2.getTextSize(label, font, 0.9, 2)[0]
        padding = 8
        x, y = np.min(polygon, axis=0)
        x = max(0, min(int(x), frame.shape[1] - text_width - 2 * padding))
        y = max(0, min(int(y), frame.shape[0] - text_height - 2 * padding))
        cv2.rectangle(frame, (x, y), (x + text_width + 2 * padding,
                                     y + text_height + 2 * padding), (0, 0, 255), -1)
        cv2.putText(frame, label, (x + padding, y + text_height + padding),
                    font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def detect_congestion_events(
    video_path: str,
    model_path: str = "weights/best.pt",
    device: str = "cpu",
    save_video: bool = False,
    output_path: str = "output_congestion.mp4",
    min_duration: float = 0.0,
) -> list[list]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alignment = align_capture(cap, fps)
    if not alignment.valid:
        cap.release()
        print(f"Suppressing congestion for {video_path}: {alignment.reason}")
        return []
    scene = SceneGeometry("scene.json", frame_width=alignment.reference_size[0],
                          frame_height=alignment.reference_size[1])
    model = YOLO(model_path)

    out = None
    annotators = {}
    if save_video:
        annotation_scene = SceneGeometry(
            "scene.json", frame_width=alignment.reference_size[0],
            frame_height=alignment.reference_size[1],
            point_transform=np.linalg.inv(alignment.video_to_reference),
        )
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        annotators = {
            "box": sv.BoxAnnotator(color=sv.Color.RED, thickness=3),
            "label": sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.8, text_thickness=2),
        }

    track_history = {}
    state = {"candidate_since": None, "active_start": None, "clear_since": None}
    last_congested_zones = set()
    events = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = frame_idx / fps
            result = model.track(frame, device=device, tracker="bytetrack.yaml",
                                 persist=True, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            congested, slow_ids, congested_zones = evaluate_congestion_spatial(
                alignment.detections(detections), scene, track_history, t_sec
            )
            events.extend(update_congestion_event(state, congested, t_sec, min_duration))
            if congested_zones:
                last_congested_zones = congested_zones
            elif state["active_start"] is None:
                last_congested_zones = set()
            if out is not None:
                red_zones = last_congested_zones if state["active_start"] is not None else set()
                out.write(annotate_frame(frame, detections, slow_ids, red_zones,
                                         annotation_scene, annotators))
            frame_idx += 1
    finally:
        end_sec = frame_idx / fps
        if state["active_start"] is not None:
            start = state["active_start"]
            end = state["clear_since"] if state["clear_since"] is not None else end_sec
            if end > start and end - start >= min_duration:
                events.append([round(start, 2), round(end, 2), "congestion"])
        cap.release()
        if out is not None:
            out.release()
    return events


def main():
    parser = argparse.ArgumentParser(description="Detect lane and intersection congestion")
    parser.add_argument("video_path")
    parser.add_argument("--model", default="weights/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output", default="output_congestion.mp4")
    parser.add_argument("--min_duration", type=float, default=0.0)
    args = parser.parse_args()
    events = detect_congestion_events(
        args.video_path, model_path=args.model, device=args.device,
        save_video=args.save_video, output_path=args.output,
        min_duration=args.min_duration,
    )
    print(f"Total congestion events found: {len(events)}")
    print(json.dumps(events, indent=2))


if __name__ == "__main__":
    main()
