"""Detect and track objects in a video with ByteTrack."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from annotate_video import detect_rfdetr, detect_yolo, load_model


TRACK_COLORS = (
    (255, 80, 80),
    (80, 220, 80),
    (80, 160, 255),
    (220, 100, 220),
    (80, 220, 220),
    (220, 220, 80),
)


@dataclass
class TrackState:
    box: np.ndarray
    label: str
    confidence: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    missing_frames: int = 0
    centers: deque[tuple[int, int]] = field(default_factory=deque)


def box_center(box: np.ndarray) -> np.ndarray:
    return np.array(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), dtype=np.float32)


def shifted_box(box: np.ndarray, shift: np.ndarray) -> np.ndarray:
    return box + np.array((shift[0], shift[1], shift[0], shift[1]), dtype=np.float32)


def detections_to_supervision(
    detections: list[tuple[Any, float, int, str]],
) -> sv.Detections:
    """Convert the annotator's detector output to supervision detections."""
    if not detections:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
        )

    return sv.Detections(
        xyxy=np.asarray([item[0] for item in detections], dtype=np.float32),
        confidence=np.asarray([item[1] for item in detections], dtype=np.float32),
        class_id=np.asarray([item[2] for item in detections], dtype=np.int32),
    )


def draw_tracks(
    frame: np.ndarray,
    tracks: dict[int, TrackState],
) -> np.ndarray:
    annotated = frame.copy()
    for track_id, track in tracks.items():
        color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
        points = list(track.centers)
        if len(points) > 1:
            cv2.polylines(
                annotated,
                [np.asarray(points, dtype=np.int32)],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        track_id = int(track_id)
        x1, y1, x2, y2 = [int(round(float(value))) for value in track.box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{track.label} #{track_id} {track.confidence:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def track_video(
    video_path: Path,
    model_path: Path | None,
    model_type: str,
    output_path: Path,
    threshold: float,
    track_buffer: int,
    match_threshold: float,
    interpolation_frames: int,
    trajectory_length: int,
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.resolve() == output_path.resolve():
        raise ValueError("Output video must be different from the input video")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine video dimensions: {video_path}")
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    model = load_model(model_type, model_path)
    detect = detect_yolo if model_type == "yolo" else detect_rfdetr
    tracker = sv.ByteTrack(
        track_activation_threshold=threshold,
        lost_track_buffer=track_buffer,
        minimum_matching_threshold=match_threshold,
        frame_rate=fps,
    )
    tracks: dict[int, TrackState] = {}
    frame_count = 0
    try:
        with tqdm(
            total=total_frames if total_frames > 0 else None,
            desc="Tracking video",
            unit="frame",
        ) as progress:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                detections = detect(model, frame, threshold)
                supervision_detections = detections_to_supervision(detections)
                tracked = tracker.update_with_detections(supervision_detections)
                labels = {item[2]: item[3] for item in detections}
                visible_ids: set[int] = set()
                for box, confidence, class_id, track_id in zip(
                    tracked.xyxy,
                    tracked.confidence,
                    tracked.class_id,
                    tracked.tracker_id,
                ):
                    if track_id is None:
                        continue
                    track_id = int(track_id)
                    current_box = np.asarray(box, dtype=np.float32)
                    current_center = box_center(current_box)
                    state = tracks.get(track_id)
                    if state is None:
                        state = TrackState(
                            box=current_box,
                            label=labels.get(int(class_id), str(class_id)),
                            confidence=float(confidence),
                            centers=deque(maxlen=trajectory_length),
                        )
                        tracks[track_id] = state
                    else:
                        previous_center = box_center(state.box)
                        elapsed_frames = state.missing_frames + 1
                        state.velocity = (current_center - previous_center) / elapsed_frames
                        state.box = current_box
                        state.label = labels.get(int(class_id), state.label)
                        state.confidence = float(confidence)
                        state.missing_frames = 0
                    state.centers.append(tuple(np.round(current_center).astype(int)))
                    visible_ids.add(track_id)

                for track_id in list(tracks):
                    if track_id in visible_ids:
                        continue
                    state = tracks[track_id]
                    state.missing_frames += 1
                    if state.missing_frames > interpolation_frames:
                        del tracks[track_id]
                        continue
                    state.box = shifted_box(state.box, state.velocity)
                    predicted_center = box_center(state.box)
                    state.centers.append(tuple(np.round(predicted_center).astype(int)))

                writer.write(draw_tracks(frame, tracks))
                frame_count += 1
                progress.update(1)
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    print(f"Processed {frame_count} frames")
    print(f"Saved tracked video: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rfdetr", "yolo"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-threshold", type=float, default=0.8)
    parser.add_argument(
        "--interpolation-frames",
        type=int,
        default=5,
        help="Maximum consecutive missed frames to fill using track motion.",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=30,
        help="Number of recent center points shown for each trajectory.",
    )
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.track_buffer < 1:
        parser.error("--track-buffer must be at least 1")
    if not 0 <= args.match_threshold <= 1:
        parser.error("--match-threshold must be between 0 and 1")
    if args.interpolation_frames < 0:
        parser.error("--interpolation-frames must be 0 or greater")
    if args.trajectory_length < 2:
        parser.error("--trajectory-length must be at least 2")

    output_path = args.output or args.video.with_name(f"{args.video.stem}_tracked.mp4")
    track_video(
        args.video,
        args.model_path,
        args.model,
        output_path,
        args.threshold,
        args.track_buffer,
        args.match_threshold,
        args.interpolation_frames,
        args.trajectory_length,
    )


if __name__ == "__main__":
    main()