import argparse
import json

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

try:
    from classes import BUS, CAR, MOTORCYCLE, TRUCK
    from scene import SceneGeometry
    from alignment import align_capture
    from utils import close_finished_events, start_active_events
except ImportError:
    from detections.classes import BUS, CAR, MOTORCYCLE, TRUCK
    from detections.scene import SceneGeometry
    from detections.alignment import align_capture
    from detections.utils import close_finished_events, start_active_events


VEHICLE_CLASSES = {CAR, BUS, TRUCK, MOTORCYCLE}


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


def _box_corners(boxes):
    return np.stack(
        [
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ],
        axis=1,
    )


def _side(line_start, line_end, points):
    return np.cross(line_end - line_start, points - line_start)


def _segments_crossed(old_points, new_points, line_start, line_end):
    old_side = _side(line_start, line_end, old_points)
    new_side = _side(line_start, line_end, new_points)
    for old_point, new_point, did_cross in zip(
        old_points, new_points, old_side * new_side <= 0
    ):
        if not did_cross:
            continue
        movement = new_point - old_point
        denominator = np.cross(movement, line_end - line_start)
        if abs(float(denominator)) < 1e-9:
            continue
        movement_ratio = np.cross(line_start - old_point, line_end - line_start) / denominator
        line_ratio = np.cross(line_start - old_point, movement) / denominator
        if 0.0 <= movement_ratio <= 1.0 and 0.0 <= line_ratio <= 1.0:
            return True
    return False


def _inside_lane(corners, lane):
    polygon = np.asarray(lane.polygon)
    return all(cv2.pointPolygonTest(polygon, tuple(point), False) >= 0 for point in corners)


def _target_lane(corners, scene, source_lane=None, preferred_lane=None):
    lanes = getattr(scene, "lanes", {})
    if preferred_lane in lanes and _inside_lane(corners, lanes[preferred_lane]):
        return preferred_lane
    for name, lane in lanes.items():
        if name != source_lane and _inside_lane(corners, lane):
            return name
    return None


def evaluate_solid_line_crossing_spatial(detections, scene, crossing_states):
    """Return vehicles crossing a solid marking until fully in the new lane."""
    event_mask = np.zeros(len(detections), dtype=bool)
    if len(detections) == 0:
        return event_mask

    class_ids = np.asarray(getattr(detections, "class_id", np.full(len(detections), -1)))
    vehicle_mask = np.isin(class_ids, list(VEHICLE_CLASSES))
    tracker_ids = getattr(detections, "tracker_id", None)
    tracker_ids = np.arange(len(detections)) if tracker_ids is None else np.asarray(tracker_ids)
    corners = _box_corners(np.asarray(detections.xyxy, dtype=float))
    previous = crossing_states.setdefault("previous_corners", {})
    active = crossing_states.setdefault("active", {})
    visible_ids = set()

    for index, tracker_id in enumerate(tracker_ids):
        if not vehicle_mask[index]:
            continue
        key = int(tracker_id)
        visible_ids.add(key)
        current_corners = corners[index]
        old_corners = previous.get(key)
        if old_corners is not None and key not in active:
            for line_name, line in getattr(scene, "solid_lines", {}).items():
                endpoints = _line_endpoints(line)
                if endpoints is None:
                    continue
                line_start, line_end = endpoints
                if not _segments_crossed(old_corners[2:], current_corners[2:], line_start, line_end):
                    continue
                source_lane = next(
                    (name for name, lane in getattr(scene, "lanes", {}).items() if _inside_lane(old_corners, lane)),
                    None,
                )
                active[key] = {"source_lane": source_lane}
                break

        previous[key] = current_corners
        state = active.get(key)
        if state is None:
            continue
        target = _target_lane(current_corners, scene, state["source_lane"])
        if target is not None:
            active.pop(key)
            continue
        event_mask[index] = True

    for key in set(previous) - visible_ids:
        previous.pop(key, None)
        active.pop(key, None)
    return event_mask


def annotate_frame(frame, detections, event_mask, scene, annotators):
    for line in scene.solid_lines.values():
        frame = annotators["line"].annotate(frame, line)

    crossing = detections[event_mask]
    if len(crossing) > 0:
        tracker_ids = crossing.tracker_id if crossing.tracker_id is not None else []
        labels = [f"#{tracker_id} SOLID LINE" for tracker_id in tracker_ids]
        frame = annotators["box"].annotate(scene=frame, detections=crossing)
        frame = annotators["label"].annotate(
            scene=frame, detections=crossing, labels=labels
        )
    return frame


def detect_solid_line_events(
    video_path,
    model_path="weights/best.pt",
    device="cpu",
    save_video=False,
    output_path="output_solid_line.mp4",
    min_duration=0.0,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alignment = align_capture(cap, fps)
    if not alignment.valid:
        cap.release()
        print(f"Suppressing solid_line_crossing for {video_path}: {alignment.reason}")
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
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        annotators = {
            "line": sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=1.0),
            "box": sv.BoxAnnotator(color=sv.Color.RED, thickness=3),
            "label": sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.8, text_thickness=2),
        }
    active_events, state, events = {}, {}, []
    frame_idx = 0
    last_t_sec = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = frame_idx / fps
            last_t_sec = t_sec
            result = model.track(frame, device=device, tracker="bytetrack.yaml", persist=True, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            mask = evaluate_solid_line_crossing_spatial(alignment.detections(detections), scene, state)
            ids = getattr(detections, "tracker_id", None)
            current_ids = set(np.asarray(ids)[mask]) if ids is not None else set()
            start_active_events(active_events, current_ids, t_sec)
            events.extend(close_finished_events(active_events, current_ids, t_sec, "solid_line_crossing", min_duration))
            if save_video and out is not None:
                out.write(annotate_frame(frame, detections, mask, annotation_scene, annotators))
            frame_idx += 1
    finally:
        events.extend(close_finished_events(active_events, set(), last_t_sec + 1.0 / fps, "solid_line_crossing", min_duration))
        cap.release()
        if out is not None:
            out.release()
    return events


def main():
    parser = argparse.ArgumentParser(description="Detect vehicles crossing solid lane markings")
    parser.add_argument("video_path")
    parser.add_argument("--model", default="weights/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output", default="output_solid_line.mp4")
    parser.add_argument("--min_duration", type=float, default=0.0)
    args = parser.parse_args()
    events = detect_solid_line_events(
        args.video_path,
        model_path=args.model,
        device=args.device,
        save_video=args.save_video,
        output_path=args.output,
        min_duration=args.min_duration,
    )
    print(f"Total solid-line events found: {len(events)}")
    print(json.dumps(events, indent=2))


if __name__ == "__main__":
    main()