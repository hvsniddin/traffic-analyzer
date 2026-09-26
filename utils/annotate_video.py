"""Detect objects in a video and save a copy with bounding boxes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


def class_name(class_names: Any, class_id: int) -> str:
    """Return a class name from either the YOLO or RF-DETR name format."""
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_id))
    if class_names is not None and 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    return str(class_id)


def detect_yolo(model: Any, frame: np.ndarray, threshold: float) -> list[tuple[Any, float, int, str]]:
    result = model.predict(source=frame, conf=threshold, verbose=False)[0]
    return [
        (box, float(confidence), int(class_id), class_name(result.names, int(class_id)))
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        )
    ]


def detect_rfdetr(model: Any, frame: np.ndarray, threshold: float) -> list[tuple[Any, float, int, str]]:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb, threshold=threshold)
    return [
        (box, float(confidence), int(class_id), class_name(model.class_names, int(class_id)))
        for box, confidence, class_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
        )
    ]


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[Any, float, int, str]],
) -> np.ndarray:
    annotated = frame.copy()
    for box, confidence, _class_id, label in detections:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label}: {confidence:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def load_model(model_type: str, model_path: Path | None) -> Any:
    default_path = (
        Path.home() / ".roboflow" / "models" / "rf-detr-medium.pth"
        if model_type == "rfdetr"
        else Path("yolo11n.pt")
    )
    selected_path = model_path or default_path
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"Default {model_type} model was not found at {selected_path}. "
            f"Provide it with --model-path PATH."
        )

    if model_type == "yolo":
        from ultralytics import YOLO

        try:
            return YOLO(str(selected_path))
        except Exception as error:
            if model_path is None:
                raise RuntimeError(
                    f"Could not load the default YOLO model at {selected_path}. "
                    "Provide a working model with --model-path PATH."
                ) from error
            raise

    from rfdetr import RFDETRMedium

    try:
        return RFDETRMedium(pretrain_weights=str(selected_path))
    except Exception as error:
        if model_path is None:
            raise RuntimeError(
                f"Could not load the default RF-DETR model at {selected_path}. "
                "Provide a working model with --model-path PATH."
            ) from error
        raise


def annotate_video(
    video_path: Path,
    model_path: Path | None,
    model_type: str,
    output_path: Path,
    threshold: float,
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
        fps = 30.0

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
    frame_count = 0
    try:
        with tqdm(
            total=total_frames if total_frames > 0 else None,
            desc="Processing video",
            unit="frame",
        ) as progress:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                writer.write(draw_detections(frame, detect(model, frame, threshold)))
                frame_count += 1
                progress.update(1)
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    print(f"Processed {frame_count} frames")
    print(f"Saved annotated video: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rfdetr", "yolo"), required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        help=(
            "Optional model file. Defaults to ~/.roboflow/models/"
            "rf-detr-medium.pth for RF-DETR or ./yolo11n.pt for YOLO."
        ),
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    output_path = args.output or args.video.with_name(f"{args.video.stem}_annotated.mp4")
    annotate_video(args.video, args.model_path, args.model, output_path, args.threshold)


if __name__ == "__main__":
    main()