"""Annotate one frame from a video with class IDs and names."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from annotate_video import detect_rfdetr, detect_yolo, load_model


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[Any, float, int, str]],
) -> np.ndarray:
    annotated = frame.copy()
    for box, confidence, class_id, class_label in detections:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{class_id}: {class_label} ({confidence:.2f})",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def annotate_frame(
    video_path: Path,
    frame_index: int,
    model_path: Path | None,
    model_type: str,
    output_path: Path,
    threshold: float,
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if frame_index < 0:
        raise ValueError("Frame index must be zero or greater")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
    finally:
        capture.release()

    if not success:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")

    model = load_model(model_type, model_path)
    detect = detect_yolo if model_type == "yolo" else detect_rfdetr
    detections = detect(model, frame, threshold)
    annotated = draw_detections(frame, detections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not write annotated frame: {output_path}")

    print(f"Frame: {frame_index}")
    print(f"Detections: {len(detections)}")
    for _box, confidence, class_id, class_label in detections:
        print(f"  {class_id}: {class_label} ({confidence:.2f})")
    print(f"Saved annotated frame: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rfdetr", "yolo"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    output_path = args.output or args.video.with_name(
        f"{args.video.stem}_frame_{args.frame_index}.jpg"
    )
    annotate_frame(
        args.video,
        args.frame_index,
        args.model_path,
        args.model,
        output_path,
        args.threshold,
    )


if __name__ == "__main__":
    main()
