import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

import argparse
import json

try:
    from classes import BUS, CAR, GREENLIGHT, MOTORCYCLE, REDLIGHT, TRUCK
    from scene import SceneGeometry
    from utils import close_finished_events, start_active_events
except ImportError:
    from detections.classes import BUS, CAR, GREENLIGHT, MOTORCYCLE, REDLIGHT, TRUCK
    from detections.scene import SceneGeometry
    from detections.utils import close_finished_events, start_active_events


def evaluate_new_event_spatial(detections, scene, state_history, t_sec=None):
    """Return confirmed red-light crossings, starting at front-line crossing."""
    if t_sec is None:
        t_sec = state_history.get("sample_time", -1 / 25) + 1 / 25
        state_history["sample_time"] = t_sec
    try:
        from traffic_light_events import evaluate_traffic_light_events
    except ImportError:
        from detections.traffic_light_events import evaluate_traffic_light_events
    mask, _ = evaluate_traffic_light_events(detections, scene, state_history, t_sec)
    return mask, ~mask


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
    scene = SceneGeometry("scene.json", frame_width=width, frame_height=height,
                          video_path=video_path)
    model = YOLO(model_path)

    annotation_scene = scene if save_video else None
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

            event_mask, _ = evaluate_new_event_spatial(detections, scene, state_history, t_sec)
            current_ids = set()
            if len(detections) > 0 and detections.tracker_id is not None:
                current_ids = set(detections.tracker_id[event_mask])
                stopped_ids = {
                    int(track_id) for track_id in detections.tracker_id
                    if state_history.get("tracks", {}).get(int(track_id), {}).get("stopped")
                }
                for track_id in stopped_ids:
                    active_violations.pop(track_id, None)

            start_active_events(active_violations, current_ids, t_sec)
            starts = state_history.get("event_starts", {})
            for tracker_id in current_ids:
                active_violations[tracker_id] = min(
                    active_violations[tracker_id],
                    starts.get(("red_light", int(tracker_id)), t_sec),
                )
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
