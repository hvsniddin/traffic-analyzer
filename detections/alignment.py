"""Validate a video view and map tracked boxes to the annotated reference view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameAlignment:
    video_to_reference: np.ndarray | None
    reason: str
    reference_size: tuple[int, int]

    @property
    def valid(self) -> bool:
        return self.video_to_reference is not None

    def points(self, points: np.ndarray) -> np.ndarray:
        if not self.valid:
            raise ValueError(f"Scene alignment unavailable: {self.reason}")
        shape = np.asarray(points).shape
        mapped = cv2.perspectiveTransform(
            np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
            self.video_to_reference,
        )
        return mapped.reshape(shape)

    def detections(self, detections):
        """Copy detections, mapping box corners so rule coordinates are reference pixels."""
        if not self.valid:
            raise ValueError(f"Scene alignment unavailable: {self.reason}")
        aligned = deepcopy(detections)
        if len(aligned) == 0:
            return aligned
        boxes = np.asarray(aligned.xyxy, dtype=np.float32)
        corners = boxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]]
        mapped = self.points(corners)
        aligned.xyxy = np.column_stack((
            mapped[:, :, 0].min(axis=1), mapped[:, :, 1].min(axis=1),
            mapped[:, :, 0].max(axis=1), mapped[:, :, 1].max(axis=1),
        )).astype(np.float32)
        return aligned


def _clear_frame(capture: cv2.VideoCapture, fps: float, max_seconds: float = 1.5):
    """Pick a sharp, exposed frame near the start, then rewind for tracking."""
    best, best_score = None, -1.0
    count = max(1, min(8, round(max_seconds * fps)))
    try:
        for index in range(count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(index * max_seconds * fps / count))
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1]))), cv2.COLOR_BGR2GRAY)
            exposure = float(gray.mean())
            if exposure < 15 or exposure > 245:
                continue
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if score > best_score:
                best, best_score = frame, score
    finally:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return best


def _features(frame: np.ndarray):
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / max(width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detector = cv2.SIFT_create(nfeatures=4000)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors, scale


def align_capture(capture: cv2.VideoCapture, fps: float, scene_path: str = "scene.json") -> FrameAlignment:
    """Estimate video-to-reference homography and verify held-out static landmarks."""
    import json

    scene_file = Path(scene_path).resolve()
    config = json.loads(scene_file.read_text(encoding="utf-8"))
    reference_size = tuple(config["reference_size"])

    def failed(reason):
        return FrameAlignment(None, reason, reference_size)

    source = Path(config["source"])
    if not source.is_absolute():
        source = scene_file.parent / source
    reference_cap = cv2.VideoCapture(str(source))
    if not reference_cap.isOpened():
        return failed(f"reference video unavailable: {source}")
    try:
        reference_cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * float(config.get("reference_time_sec", 0)))
        ok, reference = reference_cap.read()
    finally:
        reference_cap.release()
    if not ok:
        return failed("reference frame unavailable")
    if (reference.shape[1], reference.shape[0]) != reference_size:
        return failed("reference frame size differs from scene annotation")
    video_frame = _clear_frame(capture, fps)
    if video_frame is None:
        return failed("no clear frame near video start")

    key_video, desc_video, video_scale = _features(video_frame)
    key_ref, desc_ref, ref_scale = _features(reference)
    if desc_video is None or desc_ref is None or len(desc_video) < 20 or len(desc_ref) < 20:
        return failed("too few stationary features")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_video, desc_ref, k=2)
    good = [a for a, b in pairs if a.distance < 0.7 * b.distance]
    if len(good) < 24:
        return failed(f"too few feature matches: {len(good)}")
    video_pts = np.float32([key_video[m.queryIdx].pt for m in good]) / video_scale
    ref_pts = np.float32([key_ref[m.trainIdx].pt for m in good]) / ref_scale
    # Fit on one subset and validate against landmarks withheld from RANSAC.
    train = np.arange(len(good)) % 5 != 0
    test = ~train
    threshold = 0.006 * np.hypot(*reference_size)
    try:
        matrix, inliers = cv2.findHomography(video_pts[train], ref_pts[train], cv2.RANSAC, threshold)
    except cv2.error:
        return failed("homography estimation failed")
    if matrix is None or inliers is None or int(inliers.sum()) < 16:
        return failed("homography has too few inliers")
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-9:
        return failed("degenerate homography")
    projected = cv2.perspectiveTransform(video_pts[test].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - ref_pts[test], axis=1)
    aligned = errors < threshold
    if int(aligned.sum()) < 5 or float(aligned.mean()) < 0.5 or float(np.median(errors[aligned])) > threshold / 2:
        return failed("held-out landmarks do not align")
    # Reject matches concentrated in a small patch: they cannot validate the scene.
    landmarks = ref_pts[test][aligned]
    span = np.ptp(landmarks, axis=0) / np.asarray(reference_size)
    if np.any(span < 0.25):
        return failed("aligned landmarks cover too little of the reference view")
    corners = np.float32([[0, 0], [video_frame.shape[1], 0],
                          [video_frame.shape[1], video_frame.shape[0]], [0, video_frame.shape[0]]])
    mapped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if not np.isfinite(mapped).all() or cv2.contourArea(mapped) < 0.1 * np.prod(reference_size):
        return failed("homography maps too little of the video")
    return FrameAlignment(matrix, "landmarks aligned", reference_size)
