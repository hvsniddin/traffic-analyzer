import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import argparse
import json

try:
    from classes import PERSON
    from utils import close_finished_events, start_active_events
    from scene import SceneGeometry
    from alignment import align_capture
except ImportError:
    from detections.classes import PERSON
    from detections.utils import close_finished_events, start_active_events
    from detections.scene import SceneGeometry
    from detections.alignment import align_capture


def annotate_frame(frame, person_detections, jaywalker_mask, safe_mask, annotators: dict, scene, annotate_only_jaywalkers: bool):
    """Draws bounding boxes, labels, and scene zones onto the current frame."""
    
    # Optional: Draw zones on the frame (can be commented out to speed up rendering)
    if scene.road_zone:
        frame = annotators["road_zone"].annotate(scene=frame)
    for cz in scene.crosswalks.values():
        frame = sv.PolygonZoneAnnotator(zone=cz, color=sv.Color.YELLOW, thickness=2).annotate(scene=frame)

    # Annotate Jaywalkers (RED)
    jaywalkers = person_detections[jaywalker_mask]
    if len(jaywalkers) > 0:
        tracker_ids = jaywalkers.tracker_id if jaywalkers.tracker_id is not None else []
        labels = [f"#{tid} JAYWALKING" for tid in tracker_ids]
        frame = annotators["box_red"].annotate(scene=frame, detections=jaywalkers)
        frame = annotators["lbl_red"].annotate(scene=frame, detections=jaywalkers, labels=labels)

    # Annotate Safe Pedestrians (GREEN)
    if not annotate_only_jaywalkers:
        safe_peds = person_detections[safe_mask]
        if len(safe_peds) > 0:
            tracker_ids = safe_peds.tracker_id if safe_peds.tracker_id is not None else []
            labels = [f"#{tid}" for tid in tracker_ids]
            frame = annotators["box_grn"].annotate(scene=frame, detections=safe_peds)
            frame = annotators["lbl_grn"].annotate(scene=frame, detections=safe_peds, labels=labels)

    return frame


def evaluate_jaywalking_spatial(person_detections, scene) -> tuple[np.ndarray, np.ndarray]:
    """Returns boolean masks for jaywalkers and safe pedestrians."""
    if len(person_detections) == 0:
        return np.array([]), np.array([])

    # In carriageway
    in_road = scene.road_zone.trigger(person_detections) if scene.road_zone else np.zeros(len(person_detections), dtype=bool)

    # In any safe zone[cite: 2]
    in_safe_zone = np.zeros(len(person_detections), dtype=bool)
    for cw in scene.crosswalks.values():
        in_safe_zone |= cw.trigger(person_detections)
    for isl in scene.islands.values():
        in_safe_zone |= isl.trigger(person_detections)
    for wz in scene.waiting_zones.values():
        in_safe_zone |= wz.trigger(person_detections)

    jaywalker_mask = in_road & ~in_safe_zone
    safe_mask = ~jaywalker_mask
    
    return jaywalker_mask, safe_mask


def detect_jaywalking_events(
    video_path: str, 
    model_path: str = "weights/best.pt",
    device: str = "cpu",
    save_video: bool = False, 
    output_path: str = "output_jaywalking.mp4", 
    annotate_only_jaywalkers: bool = True
) -> list[list]:
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alignment = align_capture(cap, fps)
    if not alignment.valid:
        cap.release()
        print(f"Suppressing jaywalking for {video_path}: {alignment.reason}")
        return []
    scene = SceneGeometry("scene.json", frame_width=alignment.reference_size[0], frame_height=alignment.reference_size[1])
    annotation_scene = (
        SceneGeometry("scene.json", frame_width=alignment.reference_size[0],
                      frame_height=alignment.reference_size[1],
                      point_transform=np.linalg.inv(alignment.video_to_reference))
        if save_video else None
    )
    model = YOLO(model_path)

    # Setup Video Writer & Annotators (Only if saving video)
    out = None
    annotators = {}
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        annotators = {
            "box_red": sv.BoxAnnotator(color=sv.Color.RED, thickness=2),
            "lbl_red": sv.LabelAnnotator(color=sv.Color.RED, text_scale=1.2, text_thickness=2),
            "box_grn": sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2),
            "lbl_grn": sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=0.8, text_thickness=1),
            "road_zone": sv.PolygonZoneAnnotator(zone=annotation_scene.road_zone, color=sv.Color.WHITE, thickness=2) if annotation_scene.road_zone else None
        }

    active_jaywalkers = {}
    all_events = []
    frame_idx = 0
    last_t_sec = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps
        last_t_sec = t_sec

        # 1. Inference & Tracking
        results = model.track(frame, device=device, tracker="bytetrack.yaml", persist=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        person_mask = detections.class_id == PERSON # Ensure this matches your pedestrian ID
        person_detections = detections[person_mask]

        current_frame_jaywalker_ids = set()

        if len(person_detections) > 0 and person_detections.tracker_id is not None:
            
            # 2. Evaluate Spatial Logic
            jaywalker_mask, safe_mask = evaluate_jaywalking_spatial(alignment.detections(person_detections), scene)
            
            # Extract IDs of current jaywalkers
            jaywalkers = person_detections[jaywalker_mask]
            if jaywalkers.tracker_id is not None:
                current_frame_jaywalker_ids = set(jaywalkers.tracker_id)

            # 3. Annotate & Save Frame
            if save_video:
                frame = annotate_frame(
                    frame, person_detections, jaywalker_mask, safe_mask, 
                    annotators, annotation_scene, annotate_only_jaywalkers
                )

        # 4. State Management: Open new events and close finished ones
        start_active_events(active_jaywalkers, current_frame_jaywalker_ids, t_sec)
        
        completed = close_finished_events(
            active_jaywalkers, current_frame_jaywalker_ids, t_sec, 
            event_name="jaywalking", min_duration=0.5
        )
        all_events.extend(completed)

        if save_video and out is not None:
            out.write(frame)

        frame_idx += 1

    # 5. Flush lingering events at EOF
    completed_eof = close_finished_events(active_jaywalkers, set(), last_t_sec, "jaywalking", min_duration=0.5)
    all_events.extend(completed_eof)

    cap.release()
    if out:
        out.release()
        
    return all_events




def main():
    parser = argparse.ArgumentParser(description="Detect Jaywalking Events (WIUT Hackathon)")
    
    # Required positional argument
    parser.add_argument("video_path", type=str, help="Path to the input .mp4 video file")
    
    # Optional flags
    parser.add_argument("--save_video", action="store_true", help="Flag to generate and save an annotated video")
    parser.add_argument("--output", type=str, default="output_jaywalking.mp4", help="Path to save the output video (if --save_video is used)")
    parser.add_argument("--show_all_peds", action="store_true", help="Annotate all pedestrians instead of only jaywalkers")
    
    args = parser.parse_args()

    # Convert the positive 'show_all_peds' flag to the negative 'annotate_only_jaywalkers' logic
    annotate_only = not args.show_all_peds

    print(f"Processing video: {args.video_path}...")
    
    # Execute the core function
    events = detect_jaywalking_events(
        video_path=args.video_path,
        save_video=args.save_video,
        output_path=args.output,
        annotate_only_jaywalkers=annotate_only
    )

    # Print the resulting list of lists required for predictions.json
    print("\n--- Detection Complete ---")
    print(f"Total events found: {len(events)}")
    print(json.dumps(events, indent=2))


if __name__ == "__main__":
    main()
