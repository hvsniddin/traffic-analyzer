"""Run solution.detect_events and annotate the responsible objects and areas.

Usage: python annotate_events.py input.mp4 output.mp4
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from solution import detect_events


def annotate_video(video_path: str, output_path: str) -> list[list]:
    source = Path(video_path).resolve()
    target = Path(output_path).resolve()
    if source == target:
        raise ValueError("Output video must differ from input video")

    print("Detecting events...", flush=True)
    events = sorted(detect_events(str(source)), key=lambda event: event[0])
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(fps / 4.0))

    import supervision as sv
    from ultralytics import YOLO

    from detections.scene import SceneGeometry
    from detections.classes import PERSON
    from detections.traffic_light_events import evaluate_traffic_light_events
    from detections.jaywalking import evaluate_jaywalking_spatial
    from detections.failure_to_yield import evaluate_failure_to_yield_spatial
    from detections.solid_line import evaluate_solid_line_crossing_spatial
    from detections.illegal_turn import evaluate_illegal_turn_spatial
    from detections.congestion import evaluate_congestion_spatial

    root = Path(__file__).resolve().parent
    scene = SceneGeometry(str(root / "scene.json"), width, height, video_path=str(source))
    model = YOLO(str(root / "weights" / "best.pt"))
    histories = {
        "traffic_light": {}, "failure_to_yield": {},
        "solid_line_crossing": {}, "illegal_turn": {}, "congestion": {},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot write video: {target}")

    next_event = 0
    active: dict[int, list] = {}
    endings: list[tuple[float, int]] = []
    frame_index = 0
    scale = max(0.65, min(width / 1280, 1.4))
    row_height = max(30, round(34 * scale))
    margin = max(12, round(20 * scale))
    marked_boxes: dict[int, tuple[np.ndarray, set[str]]] = {}
    marked_zones: list[np.ndarray] = []
    try:
        with tqdm(total=frame_count or None, desc="Annotating", unit="frame") as progress:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                t_sec = frame_index / fps
                frame_index += 1

                while next_event < len(events) and events[next_event][0] <= t_sec:
                    event = events[next_event]
                    active[next_event] = event
                    heapq.heappush(endings, (event[1], next_event))
                    next_event += 1
                while endings and endings[0][0] <= t_sec:
                    _, event_index = heapq.heappop(endings)
                    active.pop(event_index, None)

                active_labels = {event[2] for event in active.values()}
                if (frame_index - 1) % stride == 0:
                    result = model.track(
                        frame, persist=True, tracker="bytetrack.yaml", conf=0.3,
                        imgsz=640, verbose=False,
                    )[0]
                    detections = sv.Detections.from_ultralytics(result)
                    ids = detections.tracker_id
                    if ids is None:
                        ids = np.full(len(detections), -1, dtype=int)

                    red_mask, stop_mask = evaluate_traffic_light_events(
                        detections, scene, histories["traffic_light"], t_sec
                    )
                    masks = {
                        "red_light": red_mask,
                        "stop_line": stop_mask,
                        "failure_to_yield": evaluate_failure_to_yield_spatial(
                            detections, scene, histories["failure_to_yield"], t_sec
                        ),
                        "solid_line_crossing": evaluate_solid_line_crossing_spatial(
                            detections, scene, histories["solid_line_crossing"]
                        ),
                        "illegal_turn": evaluate_illegal_turn_spatial(
                            detections, scene, histories["illegal_turn"], t_sec
                        ),
                    }
                    people_indices = np.flatnonzero(detections.class_id == PERSON)
                    people = detections[detections.class_id == PERSON]
                    masks["jaywalking"] = np.zeros(len(detections), dtype=bool)
                    masks["jaywalking"][people_indices] = evaluate_jaywalking_spatial(
                        people, scene
                    )[0]
                    congested, _, congested_zones = evaluate_congestion_spatial(
                        detections, scene, histories["congestion"], t_sec
                    )

                    marked_boxes = {}
                    for label, mask in masks.items():
                        for index in np.flatnonzero(mask):
                            track_id = int(ids[index])
                            if track_id < 0:
                                continue
                            if track_id not in marked_boxes:
                                marked_boxes[track_id] = (detections.xyxy[index], set())
                            marked_boxes[track_id][1].add(label)

                    marked_zones = []
                    if congested:
                        for kind, name in congested_zones:
                            zone = (scene.lanes if kind == "lane" else scene.intersections).get(name)
                            if zone is not None:
                                marked_zones.append(np.asarray(zone.polygon, dtype=np.int32))
                        if not marked_zones and scene.road_zone is not None:
                            marked_zones.append(np.asarray(scene.road_zone.polygon, dtype=np.int32))

                if "congestion" in active_labels and marked_zones:
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, marked_zones, (0, 0, 255))
                    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
                    for polygon in marked_zones:
                        cv2.polylines(frame, [polygon], True, (0, 0, 255), 4)
                        x, y = map(int, np.mean(polygon, axis=0))
                        cv2.putText(
                            frame, "CONGESTION", (max(0, x - 80), max(25, y)),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 3, cv2.LINE_AA,
                        )

                for box, event_labels in marked_boxes.values():
                    visible = sorted(event_labels & active_labels)
                    if not visible:
                        continue
                    x1, y1, x2, y2 = np.rint(box).astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    title = ", ".join(label.replace("_", " ").upper() for label in visible)
                    font_scale = max(0.55, scale * 0.8)
                    (text_width, text_height), baseline = cv2.getTextSize(
                        title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                    )
                    title_x = max(0, min(x1, width - text_width - 10))
                    title_y = max(text_height + 10, y1 - 5)
                    cv2.rectangle(
                        frame, (title_x, title_y - text_height - 8),
                        (title_x + text_width + 8, title_y + baseline), (0, 0, 255), -1,
                    )
                    cv2.putText(
                        frame, title, (title_x + 4, title_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2,
                        cv2.LINE_AA,
                    )

                labels = sorted(label.replace("_", " ").upper() for label in active_labels)
                lines = [f"TIME  {t_sec:.1f}s", *(labels or ["NO EVENT"])]
                panel_width = min(width, max(260, round(450 * scale)))
                panel_height = min(height, margin * 2 + row_height * len(lines))
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                for row, line in enumerate(lines):
                    y = margin + row_height * (row + 1) - 8
                    color = (230, 230, 230) if row == 0 or not labels else (60, 190, 255)
                    cv2.putText(
                        frame, line, (margin, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 2, cv2.LINE_AA,
                    )
                writer.write(frame)
                progress.update(1)
    finally:
        capture.release()
        writer.release()
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Input MP4 video")
    parser.add_argument("output", help="Annotated MP4 video")
    args = parser.parse_args()
    events = annotate_video(args.video, args.output)
    print(f"Saved {args.output} with {len(events)} event segments")
    for start, end, label in events:
        print(f"{start:.2f}–{end:.2f}  {label}")


if __name__ == "__main__":
    main()
