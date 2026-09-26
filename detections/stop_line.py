import argparse
import json

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

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
LINE_TOLERANCE_PX = 30.0
STOP_LINE_TIMEOUT_SEC = 1.0


def _line_endpoints(line):
    vector = getattr(line, "vector", line)
    start = getattr(vector, "start", None)
    end = getattr(vector, "end", None)
    if start is None or end is None:
        return None

    def point_xy(point):
        if hasattr(point, "x") and hasattr(point, "y"):
            return np.asarray([float(point.x), float(point.y)])
        if len(point) >= 2:
            return np.asarray([float(point[0]), float(point[1])])
        return None

    start_xy = point_xy(start)
    end_xy = point_xy(end)
    if start_xy is None or end_xy is None:
        return None
    return start_xy, end_xy


def _bottom_centers(detections):
    boxes = np.asarray(detections.xyxy, dtype=float)
    return np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]))


def _tracker_ids(detections):
    ids = getattr(detections, "tracker_id", None)
    return np.arange(len(detections)) if ids is None else np.asarray(ids)


def _distance_to_segment(point, start, end):
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0:
        return float(np.linalg.norm(point - start))
    projection = np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0)
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def evaluate_stop_line_spatial(
    detections,
    scene,
    state_history,
    t_sec=0.0,
    tolerance_px=LINE_TOLERANCE_PX,
    timeout_sec=STOP_LINE_TIMEOUT_SEC,
):
    """Return vehicles resting on a stop line while the signal is red."""
    event_mask = np.zeros(len(detections), dtype=bool)
    class_ids = np.asarray(getattr(detections, "class_id", np.full(len(detections), -1)))

    if np.any(class_ids == GREENLIGHT):
        state_history["signal_is_red"] = False
    elif np.any(class_ids == REDLIGHT):
        state_history["signal_is_red"] = True

    on_line_since = state_history.setdefault("on_line_since", {})
    if not state_history.get("signal_is_red", False) or len(detections) == 0:
        on_line_since.clear()
        return event_mask, ~event_mask

    vehicle_mask = np.isin(class_ids, list(VEHICLE_CLASSES))
    centers = _bottom_centers(detections)
    tracker_ids = _tracker_ids(detections)
    lines = [
        endpoints
        for line in getattr(scene, "stop_lines", {}).values()
        if (endpoints := _line_endpoints(line)) is not None
    ]

    if not lines:
        return event_mask, ~event_mask

    visible_vehicle_ids = set()
    for index, (tracker_id, center) in enumerate(zip(tracker_ids, centers)):
        if not vehicle_mask[index]:
            continue

        tracker_key = int(tracker_id)
        visible_vehicle_ids.add(tracker_key)
        is_on_line = any(
            _distance_to_segment(center, start, end) <= tolerance_px
            for start, end in lines
        )
        if not is_on_line:
            on_line_since.pop(tracker_key, None)
            continue

        if tracker_key not in on_line_since:
            on_line_since[tracker_key] = t_sec
        event_mask[index] = t_sec - on_line_since[tracker_key] >= timeout_sec

    for tracker_key in set(on_line_since) - visible_vehicle_ids:
        on_line_since.pop(tracker_key, None)

    return event_mask, ~event_mask


def annotate_frame(frame, detections, event_mask, annotators, scene):
    """Draw stop lines, signal detections, and vehicles stopped on the line."""
    for line in scene.stop_lines.values():
        frame = annotators["line"].annotate(frame, line)

    signals = detections[np.isin(detections.class_id, [REDLIGHT, GREENLIGHT])]
    if len(signals) > 0:
        signal_labels = [
            "RED LIGHT" if class_id == REDLIGHT else "GREEN LIGHT"
            for class_id in signals.class_id
        ]
        frame = annotators["signal_box"].annotate(scene=frame, detections=signals)
        frame = annotators["signal_label"].annotate(
            scene=frame, detections=signals, labels=signal_labels
        )

    stopped = detections[event_mask]
    if len(stopped) > 0:
        tracker_ids = stopped.tracker_id if stopped.tracker_id is not None else []
        labels = [f"#{tracker_id} STOP LINE" for tracker_id in tracker_ids]
        frame = annotators["box"].annotate(scene=frame, detections=stopped)
        frame = annotators["label"].annotate(scene=frame, detections=stopped, labels=labels)
    return frame


def detect_stop_line_events(
    video_path: str,
    model_path: str = "weights/best.pt",
    device: str = "cpu",
    save_video: bool = False,
    output_path: str = "output_stop_line.mp4",
    min_duration: float = 0.0,
) -> list[list]:
    """Detect vehicles stopped on a configured stop line during a red phase."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alignment = align_capture(cap, fps)
    if not alignment.valid:
        cap.release()
        print(f"Suppressing stop_line for {video_path}: {alignment.reason}")
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        annotators = {
            "line": sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=1.0),
            "signal_box": sv.BoxAnnotator(color=sv.Color.YELLOW, thickness=2),
            "signal_label": sv.LabelAnnotator(color=sv.Color.YELLOW, text_scale=0.8, text_thickness=2),
            "box": sv.BoxAnnotator(color=sv.Color.RED, thickness=3),
            "label": sv.LabelAnnotator(color=sv.Color.RED, text_scale=1.0, text_thickness=2),
        }

    active_events = {}
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
            event_mask, _ = evaluate_stop_line_spatial(
                alignment.detections(detections), scene, state_history, t_sec=t_sec
            )

            current_ids = set()
            if len(detections) > 0 and detections.tracker_id is not None:
                current_ids = set(detections.tracker_id[event_mask])

            start_active_events(active_events, current_ids, t_sec)
            all_events.extend(close_finished_events(
                active_events, current_ids, t_sec, "stop_line", min_duration=min_duration
            ))

            if save_video and out is not None:
                out.write(annotate_frame(frame, detections, event_mask, annotators, annotation_scene))
            frame_idx += 1
    finally:
        all_events.extend(close_finished_events(
            active_events, set(), last_t_sec + (1.0 / fps), "stop_line", min_duration=min_duration
        ))
        cap.release()
        if out is not None:
            out.release()

    return all_events


def main():
    parser = argparse.ArgumentParser(description="Detect stop-line events")
    parser.add_argument("video_path", type=str)
    parser.add_argument("--model", type=str, default="weights/best.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output", type=str, default="output_stop_line.mp4")
    parser.add_argument("--min_duration", type=float, default=0.0)
    args = parser.parse_args()

    print(f"Processing video: {args.video_path}...")
    events = detect_stop_line_events(
        video_path=args.video_path,
        model_path=args.model,
        device=args.device,
        save_video=args.save_video,
        output_path=args.output,
        min_duration=args.min_duration,
    )
    print(f"Total stop-line events found: {len(events)}")
    print(json.dumps(events, indent=2))


if __name__ == "__main__":
    main()