import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

import argparse
import json

try:
    from classes import BUS, CAR, GREENLIGHT, MOTORCYCLE, REDLIGHT, TRUCK
    from scene import SceneGeometry
    from alignment import align_capture
    from utils import close_finished_events, start_active_events
except ImportError:
    from detections.classes import BUS, CAR, GREENLIGHT, MOTORCYCLE, REDLIGHT, TRUCK
    from detections.scene import SceneGeometry
    from detections.alignment import align_capture
    from detections.utils import close_finished_events, start_active_events


VEHICLE_CLASSES = {CAR, BUS, TRUCK, MOTORCYCLE}
GREEN_SIGNAL_MEMORY_FRAMES = 25


def _line_endpoints(line):
    vector = getattr(line, "vector", line)
    start = getattr(vector, "start", None)
    end = getattr(vector, "end", None)
    if start is None or end is None:
        return None

    def point_xy(point):
        if hasattr(point, "x") and hasattr(point, "y"):
            return float(point.x), float(point.y)
        if len(point) >= 2:
            return float(point[0]), float(point[1])
        return None

    start_xy = point_xy(start)
    end_xy = point_xy(end)
    if start_xy is None or end_xy is None:
        return None
    return np.asarray(start_xy), np.asarray(end_xy)


def _front_rear_points(box, previous_center, previous_motion):
    """Estimate the leading and trailing corners from track motion."""
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    motion = center - previous_center
    if np.linalg.norm(motion) < 1e-6:
        motion = previous_motion
    if np.linalg.norm(motion) < 1e-6:
        return None, None, previous_motion

    direction = motion / np.linalg.norm(motion)
    front = np.array([x2 if direction[0] >= 0 else x1, y2 if direction[1] >= 0 else y1])
    rear = np.array([x1 if direction[0] >= 0 else x2, y1 if direction[1] >= 0 else y2])
    return front, rear, motion


def _tracker_ids(detections):
    ids = getattr(detections, "tracker_id", None)
    return np.arange(len(detections)) if ids is None else np.asarray(ids)


def evaluate_new_event_spatial(detections, scene, state_history):
    """Return vehicles that crossed a stop line while the signal was red."""
    frame_index = state_history.get("frame_index", -1) + 1
    state_history["frame_index"] = frame_index

    class_ids = np.asarray(getattr(detections, "class_id", np.array([], dtype=int)))
    green_seen = bool(np.any(class_ids == GREENLIGHT))
    red_seen = bool(np.any(class_ids == REDLIGHT))
    if green_seen:
        state_history["last_green_frame"] = frame_index
    if red_seen:
        state_history["last_red_frame"] = frame_index

    recent_green = (
        "last_green_frame" in state_history
        and frame_index - state_history["last_green_frame"] <= GREEN_SIGNAL_MEMORY_FRAMES
    )
    state_history["signal_is_red"] = red_seen or not recent_green

    if len(detections) == 0:
        return np.array([]), np.array([])

    event_mask = np.zeros(len(detections), dtype=bool)

    signal_is_red = state_history.get("signal_is_red", False)
    vehicle_mask = np.isin(class_ids, list(VEHICLE_CLASSES))
    boxes = np.asarray(detections.xyxy, dtype=float)
    previous = state_history.setdefault("vehicle_positions", {})
    violations = state_history.setdefault("red_light_violations", {})
    tracker_ids = _tracker_ids(detections)

    for index, tracker_id in enumerate(tracker_ids):
        if not vehicle_mask[index]:
            continue

        key = int(tracker_id)
        box = boxes[index]
        center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
        track = previous.get(key)
        previous[key] = {
            "box": box.copy(),
            "center": center,
            "motion": center - track["center"] if track is not None else np.zeros(2),
        }
        if track is None:
            continue

        front, rear, motion = _front_rear_points(box, track["center"], track["motion"])
        previous[key]["motion"] = motion
        if front is None:
            event_mask[index] = key in violations
            continue

        if key in violations:
            target_side = violations[key]["target_side"]
            line_start, line_end = violations[key]["line"]
            rear_side = np.cross(line_end - line_start, rear - line_start)
            if target_side * rear_side > 0:
                del violations[key]
            else:
                event_mask[index] = True
            continue

        if not signal_is_red:
            continue

        for line in getattr(scene, "stop_lines", {}).values():
            endpoints = _line_endpoints(line)
            if endpoints is None:
                continue
            line_start, line_end = endpoints
            line_vector = line_end - line_start
            previous_front, _, _ = _front_rear_points(
                track["box"],
                track["center"] - track["motion"],
                motion,
            )
            if previous_front is None:
                continue
            previous_side = np.cross(line_vector, previous_front - line_start)
            current_side = np.cross(line_vector, front - line_start)
            if previous_side * current_side < 0:
                target_side = 1 if current_side > 0 else -1
                violations[key] = {
                    "target_side": target_side,
                    "line": (line_start, line_end),
                }
                event_mask[index] = True
                break

    safe_mask = ~event_mask
    return event_mask, safe_mask


def annotate_frame(frame, detections, event_mask, annotators, scene):
    """Draw stop lines, signal detections, and red-light violations."""
    for line in scene.stop_lines.values():
        frame = annotators["line"].annotate(frame, line)

    signals = detections[np.isin(detections.class_id, [REDLIGHT, GREENLIGHT])]
    if len(signals) > 0:
        signal_labels = ["RED LIGHT" if class_id == REDLIGHT else "GREEN LIGHT" for class_id in signals.class_id]
        frame = annotators["signal_box"].annotate(scene=frame, detections=signals)
        frame = annotators["signal_label"].annotate(scene=frame, detections=signals, labels=signal_labels)

    violating = detections[event_mask]
    if len(violating) > 0:
        tracker_ids = violating.tracker_id if violating.tracker_id is not None else []
        labels = [f"#{tracker_id} RED LIGHT" for tracker_id in tracker_ids]
        frame = annotators["box"].annotate(scene=frame, detections=violating)
        frame = annotators["label"].annotate(scene=frame, detections=violating, labels=labels)
    return frame


def detect_red_light_events(
    video_path: str,
    model_path: str = "weights/best.pt",
    device: str = "cpu",
    save_video: bool = False,
    output_path: str = "output_red_light.mp4",
    min_duration: float = 0.0,
) -> list[list]:
    """Detect vehicles that cross a configured stop line during a red phase."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alignment = align_capture(cap, fps)
    if not alignment.valid:
        cap.release()
        print(f"Suppressing red_light for {video_path}: {alignment.reason}")
        return []
    scene = SceneGeometry("scene.json", frame_width=alignment.reference_size[0], frame_height=alignment.reference_size[1])
    model = YOLO(model_path)

    annotation_scene = (
        SceneGeometry("scene.json", frame_width=alignment.reference_size[0],
                      frame_height=alignment.reference_size[1],
                      point_transform=np.linalg.inv(alignment.video_to_reference))
        if save_video else None
    )
    out = None
    annotators = {}
    if save_video:
        fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        annotators = {
            "line": sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=1.0),
            "signal_box": sv.BoxAnnotator(color=sv.Color.YELLOW, thickness=2),
            "signal_label": sv.LabelAnnotator(color=sv.Color.YELLOW, text_scale=0.8, text_thickness=2),
            "box": sv.BoxAnnotator(color=sv.Color.RED, thickness=3),
            "label": sv.LabelAnnotator(color=sv.Color.RED, text_scale=1.0, text_thickness=2),
        }

    active_violations = {}
    state_history = {}
    all_events = []
    frame_idx = 0
    last_t_sec = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t_sec = frame_idx / fps
            last_t_sec = t_sec
            results = model.track(
                frame, device=device, tracker="bytetrack.yaml", persist=True, verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)

            event_mask, _ = evaluate_new_event_spatial(alignment.detections(detections), scene, state_history)
            current_ids = set()
            if len(detections) > 0 and detections.tracker_id is not None:
                current_ids = set(detections.tracker_id[event_mask])

            start_active_events(active_violations, current_ids, t_sec)
            all_events.extend(close_finished_events(
                active_violations, current_ids, t_sec, "red_light", min_duration=min_duration
            ))

            if save_video and out is not None:
                frame = annotate_frame(frame, detections, event_mask, annotators, annotation_scene)
                out.write(frame)
            frame_idx += 1
    finally:
        all_events.extend(close_finished_events(
            active_violations, set(), last_t_sec + (1.0 / fps), "red_light", min_duration=min_duration
        ))
        cap.release()
        if out is not None:
            out.release()

    return all_events


def main():
    parser = argparse.ArgumentParser(description="Detect red-light violations")
    parser.add_argument("video_path", type=str)
    parser.add_argument("--model", type=str, default="weights/best.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output", type=str, default="output_red_light.mp4")
    parser.add_argument("--min_duration", type=float, default=0.0)
    args = parser.parse_args()

    print(f"Processing video: {args.video_path}...")
    events = detect_red_light_events(
        video_path=args.video_path,
        model_path=args.model,
        device=args.device,
        save_video=args.save_video,
        output_path=args.output,
        min_duration=args.min_duration,
    )
    print(f"Total red-light events found: {len(events)}")
    print(json.dumps(events, indent=2))


if __name__ == "__main__":
    main()