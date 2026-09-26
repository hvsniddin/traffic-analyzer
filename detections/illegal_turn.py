import argparse
import json

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

try:
    from classes import BUS, CAR, MOTORCYCLE, TRUCK
    from scene import SceneGeometry
    from utils import close_finished_events
except ImportError:
    from detections.classes import BUS, CAR, MOTORCYCLE, TRUCK
    from detections.scene import SceneGeometry
    from detections.utils import close_finished_events


VEHICLE_CLASSES = {CAR, BUS, TRUCK, MOTORCYCLE}


def _tracker_ids(detections):
    ids = getattr(detections, "tracker_id", None)
    return np.arange(len(detections)) if ids is None else np.asarray(ids)


def _center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _in_intersection(point, scene):
    for intersection in getattr(scene, "intersections", {}).values():
        polygon = np.asarray(intersection.polygon, dtype=np.float32)
        if cv2.pointPolygonTest(polygon, tuple(point), False) >= 0:
            return True
    return False


def _in_southeast_crosswalk(point, scene):
    crosswalks = getattr(scene, "crosswalks", {})
    if not crosswalks:
        return False

    crosswalk = crosswalks.get("crosswalk_1")
    if crosswalk is None:
        crosswalk = max(
            crosswalks.values(),
            key=lambda zone: float(np.mean(np.asarray(zone.polygon)[:, 1])),
        )
    polygon = np.asarray(crosswalk.polygon, dtype=np.float32)
    return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0



def evaluate_illegal_turn_spatial(detections, scene, turn_states, t_sec):
    """Return vehicles that enter, leave, and then cross the southeast crosswalk.

    Crossing the crosswalk without first entering the intersection is safe.
    """
    event_mask = np.zeros(len(detections), dtype=bool)
    if len(detections) == 0:
        return event_mask

    class_ids = np.asarray(getattr(detections, "class_id", np.full(len(detections), -1)))
    vehicle_mask = np.isin(class_ids, list(VEHICLE_CLASSES))
    boxes = np.asarray(detections.xyxy, dtype=float)
    tracker_ids = _tracker_ids(detections)
    visible_ids = set(tracker_ids[vehicle_mask])

    for tracker_id in list(turn_states):
        if tracker_id not in visible_ids:
            turn_states.pop(tracker_id)

    for index, tracker_id in enumerate(tracker_ids):
        if not vehicle_mask[index]:
            continue

        key = int(tracker_id)
        current_center = _center(boxes[index])
        state = turn_states.get(key)
        if state is None:
            turn_states[key] = {
                "previous_center": current_center,
                "in_intersection": _in_intersection(current_center, scene),
                "entered_intersection": _in_intersection(current_center, scene),
                "left_intersection": False,
                "illegal": False,
                "start_sec": float(t_sec),
            }
            continue

        in_intersection = _in_intersection(current_center, scene)
        in_southeast_crosswalk = _in_southeast_crosswalk(current_center, scene)
        previously_in_intersection = state["in_intersection"]
        state["in_intersection"] = in_intersection
        state["previous_center"] = current_center
        if in_intersection:
            state["entered_intersection"] = True
            if not previously_in_intersection:
                state["start_sec"] = float(t_sec)
        elif previously_in_intersection and state["entered_intersection"]:
            state["left_intersection"] = True

        if state["left_intersection"] and in_southeast_crosswalk:
            state["illegal"] = True

        event_mask[index] = bool(state["illegal"] and in_southeast_crosswalk)

    return event_mask


def annotate_frame(frame, detections, event_mask, scene, annotators):
    for intersection_annotator in annotators["intersection"]:
        frame = intersection_annotator.annotate(scene=frame)

    violating = detections[event_mask]
    if len(violating) > 0:
        tracker_ids = violating.tracker_id if violating.tracker_id is not None else []
        labels = [f"#{tracker_id} ILLEGAL TURN" for tracker_id in tracker_ids]
        frame = annotators["box"].annotate(scene=frame, detections=violating)
        frame = annotators["label"].annotate(scene=frame, detections=violating, labels=labels)
    return frame


def detect_illegal_turn_events(
    video_path,
    model_path="weights/best.pt",
    device="cpu",
    save_video=False,
    output_path="output_illegal_turn.mp4",
    min_duration=0.0,
):
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
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        annotators = {
            "intersection": [
                sv.PolygonZoneAnnotator(
                    zone=intersection,
                    color=sv.Color.YELLOW,
                    thickness=3,
                )
                for intersection in annotation_scene.intersections.values()
            ],
            "box": sv.BoxAnnotator(color=sv.Color.RED, thickness=3),
            "label": sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.8, text_thickness=2),
        }

    turn_states = {}
    active_events = {}
    events = []
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
            event_mask = evaluate_illegal_turn_spatial(detections, scene, turn_states, t_sec)
            ids = getattr(detections, "tracker_id", None)
            current_ids = set(np.asarray(ids)[event_mask]) if ids is not None else set()
            for tracker_id in current_ids:
                if tracker_id not in active_events:
                    active_events[tracker_id] = turn_states.get(int(tracker_id), {}).get(
                        "start_sec", t_sec
                    )
            events.extend(close_finished_events(active_events, current_ids, t_sec, "illegal_turn", min_duration))

            if save_video and out is not None:
                out.write(annotate_frame(frame, detections, event_mask, annotation_scene, annotators))
            frame_idx += 1
    finally:
        events.extend(close_finished_events(active_events, set(), last_t_sec + 1.0 / fps, "illegal_turn", min_duration))
        cap.release()
        if out is not None:
            out.release()
    return events


def main():
    parser = argparse.ArgumentParser(description="Detect illegal right turns")
    parser.add_argument("video_path")
    parser.add_argument("--model", default="weights/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output", default="output_illegal_turn.mp4")
    parser.add_argument("--min_duration", type=float, default=0.0)
    args = parser.parse_args()
    events = detect_illegal_turn_events(
        args.video_path,
        model_path=args.model,
        device=args.device,
        save_video=args.save_video,
        output_path=args.output,
        min_duration=args.min_duration,
    )
    print(f"Total illegal-turn events found: {len(events)}")
    print(json.dumps(events, indent=2))


if __name__ == "__main__":
    main()
