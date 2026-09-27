import argparse
import json

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

try:
    from classes import BUS, CAR, MOTORCYCLE, PERSON, TRUCK
    from scene import SceneGeometry
except ImportError:
    from detections.classes import BUS, CAR, MOTORCYCLE, PERSON, TRUCK
    from detections.scene import SceneGeometry


VEHICLE_CLASSES = {CAR, BUS, TRUCK, MOTORCYCLE}
MIN_HEADING_MOVEMENT_PX = 2.0


def _bottom_centers(detections):
    boxes = np.asarray(detections.xyxy, dtype=float)
    return np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]))


def _tracker_ids(detections):
    ids = getattr(detections, "tracker_id", None)
    return np.arange(len(detections)) if ids is None else np.asarray(ids)


def _crosswalk_membership(detections, scene):
    return [
        (name, crosswalk.trigger(detections))
        for name, crosswalk in getattr(scene, "crosswalks", {}).items()
    ]


def _in_any_crosswalk(detections, scene):
    mask = np.zeros(len(detections), dtype=bool)
    for _, crosswalk_mask in _crosswalk_membership(detections, scene):
        mask |= crosswalk_mask
    return mask


def _pedestrian_in_vehicle_path(vehicle_box, vehicle_point, heading, pedestrian_box,
                                pedestrian_point):
    """Whether a pedestrian's feet occupy the road ahead of this vehicle.

    Box dimensions give a local image-scale estimate, so the corridor also
    works for vehicles at different depths in the camera image.
    """
    relative = pedestrian_point - vehicle_point
    ahead = float(np.dot(relative, heading))
    sideways = abs(float(heading[0] * relative[1] - heading[1] * relative[0]))
    vehicle_width = min(vehicle_box[2] - vehicle_box[0],
                        vehicle_box[3] - vehicle_box[1])
    pedestrian_width = pedestrian_box[2] - pedestrian_box[0]
    clearance = 0.7 * vehicle_width + 0.5 * pedestrian_width
    return -0.25 * vehicle_width <= ahead <= 3.0 * vehicle_width and sideways <= clearance


def evaluate_failure_to_yield_spatial(detections, scene, crossing_states, t_sec):
    """Return crossing vehicles whose travel corridor contains a pedestrian.

    ``crossing_states`` persists across frames and stores the traversal start
    until the vehicle leaves the crosswalk.
    """
    event_mask = np.zeros(len(detections), dtype=bool)
    if len(detections) == 0:
        return event_mask

    class_ids = np.asarray(detections.class_id)
    vehicle_mask = np.isin(class_ids, list(VEHICLE_CLASSES))
    pedestrian_mask = class_ids == PERSON
    crosswalk_membership = _crosswalk_membership(detections, scene)
    in_crosswalk = np.zeros(len(detections), dtype=bool)
    for _, crosswalk_mask in crosswalk_membership:
        in_crosswalk |= crosswalk_mask
    tracker_ids = _tracker_ids(detections)
    foot_points = _bottom_centers(detections)
    boxes = np.asarray(detections.xyxy, dtype=float)
    visible_vehicle_ids = set(tracker_ids[vehicle_mask])

    for tracker_id in list(crossing_states):
        if tracker_id not in visible_vehicle_ids:
            crossing_states.pop(tracker_id)

    for index, tracker_id in enumerate(tracker_ids):
        if not vehicle_mask[index]:
            continue

        point = foot_points[index]
        state = crossing_states.setdefault(tracker_id, {
            "start_sec": None, "pedestrian_present": False,
            "last_point": point, "heading": None, "was_in_crosswalk": False,
        })
        displacement = point - state["last_point"]
        vehicle_scale = min(boxes[index, 2] - boxes[index, 0],
                            boxes[index, 3] - boxes[index, 1])
        distance = float(np.linalg.norm(displacement))
        if distance >= max(MIN_HEADING_MOVEMENT_PX, 0.05 * vehicle_scale):
            direction = displacement / distance
            previous = state["heading"]
            if previous is not None:
                direction = 0.25 * previous + 0.75 * direction
                direction /= np.linalg.norm(direction)
            state["heading"] = direction
        state["last_point"] = point

        vehicle_crosswalks = {
            name for name, crosswalk_mask in crosswalk_membership if crosswalk_mask[index]
        }
        if not in_crosswalk[index]:
            state["was_in_crosswalk"] = False
            continue
        if not state["was_in_crosswalk"]:
            state["start_sec"] = float(t_sec)
            state["pedestrian_present"] = False
        state["was_in_crosswalk"] = True

        pedestrian_indices = {
            pedestrian_index
            for name, crosswalk_mask in crosswalk_membership
            if name in vehicle_crosswalks
            for pedestrian_index in np.flatnonzero(pedestrian_mask & crosswalk_mask)
        }
        pedestrian_present = state["heading"] is not None and any(
            _pedestrian_in_vehicle_path(
                boxes[index], point, state["heading"],
                boxes[pedestrian_index], foot_points[pedestrian_index],
            )
            for pedestrian_index in pedestrian_indices
        )
        if pedestrian_present:
            state["pedestrian_present"] = True
        event_mask[index] = bool(
            in_crosswalk[index] and state["pedestrian_present"]
        )

    return event_mask


def _close_finished_crossings(crossing_states, current_vehicle_ids, t_sec):
    events = []
    for tracker_id, state in list(crossing_states.items()):
        if tracker_id in current_vehicle_ids:
            continue
        if state["start_sec"] is not None and state["pedestrian_present"] and t_sec > state["start_sec"]:
            events.append([round(state["start_sec"], 2), round(t_sec, 2), "failure_to_yield"])
        state["start_sec"] = None
        state["pedestrian_present"] = False
        state["was_in_crosswalk"] = False
    return events


def annotate_frame(frame, detections, event_mask, scene, annotators):
    for crosswalk_annotator in annotators["crosswalk"]:
        frame = crosswalk_annotator.annotate(scene=frame)

    violating = detections[event_mask]
    if len(violating) > 0:
        tracker_ids = violating.tracker_id if violating.tracker_id is not None else []
        labels = [f"#{tracker_id} FAILURE TO YIELD" for tracker_id in tracker_ids]
        frame = annotators["box"].annotate(scene=frame, detections=violating)
        frame = annotators["label"].annotate(scene=frame, detections=violating, labels=labels)
    return frame


def detect_failure_to_yield_events(
    video_path: str,
    model_path: str = "weights/best.pt",
    device: str = "cpu",
    save_video: bool = False,
    output_path: str = "output_failure_to_yield.mp4",
) -> list[list]:
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
            "crosswalk": [
                sv.PolygonZoneAnnotator(zone=crosswalk, color=sv.Color.YELLOW, thickness=2)
                for crosswalk in annotation_scene.crosswalks.values()
            ],
            "box": sv.BoxAnnotator(color=sv.Color.RED, thickness=3),
            "label": sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.8, text_thickness=2),
        }

    crossing_states = {}
    events = []
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
            aligned = detections
            event_mask = evaluate_failure_to_yield_spatial(
                aligned, scene, crossing_states, t_sec
            )
            class_ids = np.asarray(detections.class_id)
            vehicle_mask = np.isin(class_ids, list(VEHICLE_CLASSES))
            current_vehicle_ids = set(_tracker_ids(detections)[vehicle_mask & _in_any_crosswalk(aligned, scene)])
            events.extend(_close_finished_crossings(crossing_states, current_vehicle_ids, t_sec))

            if save_video and out is not None:
                out.write(annotate_frame(frame, detections, event_mask, annotation_scene, annotators))
            frame_idx += 1
    finally:
        events.extend(_close_finished_crossings(crossing_states, set(), last_t_sec + 1.0 / fps))
        cap.release()
        if out is not None:
            out.release()

    return events


def main():
    parser = argparse.ArgumentParser(description="Detect failure-to-yield events")
    parser.add_argument("video_path")
    parser.add_argument("--model", default="weights/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output", default="output_failure_to_yield.mp4")
    args = parser.parse_args()

    events = detect_failure_to_yield_events(
        args.video_path,
        model_path=args.model,
        device=args.device,
        save_video=args.save_video,
        output_path=args.output,
    )
    print(f"Total failure-to-yield events found: {len(events)}")
    print(json.dumps(events, indent=2))


if __name__ == "__main__":
    main()
