"""Align a fixed-camera scene map to one video using stationary image features.

The result maps coordinates in ``scene_reference.jpg`` to the current video.
All matching runs at the reference image resolution, once per video.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


MIN_MATCHES = 100
MIN_INLIERS = 70
MIN_INLIER_RATIO = 0.5
MAX_MEDIAN_ERROR_PX = 1.5
MAX_CORNER_SHIFT_FRACTION = 0.08
MAX_SAMPLE_DISAGREEMENT_PX = 3.0
# Inputs smaller than the reference (e.g. the website's 540p previews) are
# upscaled for matching and keep fewer SIFT inliers for the same transform.
# Inlier gates shrink with the pixel-area ratio; the error gates stay strict.
MIN_SMALL_INLIER_RATIO = 0.33
MAX_SMALL_SAMPLE_DISAGREEMENT_PX = 6.0


def _landmarks(width: int, height: int) -> np.ndarray:
    return np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1], [width / 2, height / 2]]
    ).reshape(-1, 1, 2)


def _estimate_one(reference_gray: np.ndarray, target_bgr: np.ndarray) -> tuple[np.ndarray | None, dict]:
    height, width = reference_gray.shape
    area_ratio = min(1.0, (target_bgr.shape[0] * target_bgr.shape[1]) / float(width * height))
    min_inliers = MIN_INLIERS * area_ratio
    min_ratio = MIN_INLIER_RATIO if area_ratio >= 1.0 else MIN_SMALL_INLIER_RATIO
    target = cv2.resize(target_bgr, (width, height), interpolation=cv2.INTER_AREA)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=6000)
    ref_keys, ref_des = sift.detectAndCompute(reference_gray, None)
    tgt_keys, tgt_des = sift.detectAndCompute(target_gray, None)
    if ref_des is None or tgt_des is None:
        return None, {"reason": "no features"}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref_des, tgt_des, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(matches) < MIN_MATCHES * area_ratio:
        return None, {"reason": "too few matches", "matches": len(matches)}
    source = np.float32([ref_keys[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([tgt_keys[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if matrix is None or mask is None:
        return None, {"reason": "homography failed", "matches": len(matches)}
    keep = mask.ravel().astype(bool)
    src_inliers = source[keep]
    tgt_inliers = target[keep]
    projected = cv2.perspectiveTransform(src_inliers, matrix)
    errors = np.linalg.norm(projected[:, 0] - tgt_inliers[:, 0], axis=1)
    original_errors = np.linalg.norm(src_inliers[:, 0] - tgt_inliers[:, 0], axis=1)
    moved = cv2.perspectiveTransform(_landmarks(width, height), matrix)
    corner_shift = np.linalg.norm(moved[:4, 0] - _landmarks(width, height)[:4, 0], axis=1)
    stats = {
        "matches": len(matches),
        "inliers": int(keep.sum()),
        "inlier_ratio": round(float(keep.mean()), 3),
        "original_error_px": round(float(np.median(original_errors)), 2),
        "aligned_error_px": round(float(np.median(errors)), 2),
        "max_corner_shift_px": round(float(corner_shift.max()), 2),
    }
    if (
        stats["inliers"] < min_inliers
        or stats["inlier_ratio"] < min_ratio
        or stats["aligned_error_px"] > MAX_MEDIAN_ERROR_PX
        or stats["max_corner_shift_px"] > MAX_CORNER_SHIFT_FRACTION * max(width, height)
    ):
        return None, {**stats, "reason": "quality gate failed"}
    return matrix, stats


@lru_cache(maxsize=8)
def estimate_video_alignment(video_path: str, reference_path: str | Path) -> tuple[np.ndarray | None, dict]:
    """Return homography at reference-image scale, or None for identity/failure."""
    reference = cv2.imread(str(reference_path))
    if reference is None:
        return None, {"status": "unavailable", "reason": f"missing reference image: {reference_path}"}
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, {"status": "unavailable", "reason": f"cannot open video: {video_path}"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    times = [min(20.0, max(0.0, duration - 1.0))]
    if duration > 60:
        times.append(duration / 2.0)
    estimates = []
    try:
        for second in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(second * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            matrix, stats = _estimate_one(reference_gray, frame)
            if matrix is not None:
                estimates.append((matrix, stats, second))
    finally:
        cap.release()
    if not estimates:
        return None, {"status": "unavailable", "reason": "no reliable frame match"}
    if len(estimates) >= 2:
        height, width = reference_gray.shape
        anchors = _landmarks(width, height)
        first = cv2.perspectiveTransform(anchors, estimates[0][0])
        second = cv2.perspectiveTransform(anchors, estimates[1][0])
        disagreement = float(np.linalg.norm(first[:, 0] - second[:, 0], axis=1).max())
        small = frame.shape[0] < reference_gray.shape[0]
        limit = MAX_SMALL_SAMPLE_DISAGREEMENT_PX if small else MAX_SAMPLE_DISAGREEMENT_PX
        if disagreement > limit:
            return None, {"status": "unavailable", "reason": "inconsistent transforms", "max_disagreement_px": round(disagreement, 2)}
    matrix, stats, second = max(estimates, key=lambda item: item[1]["inliers"])
    if stats["original_error_px"] < 2.0:
        return None, {"status": "identity", "sample_time_sec": round(second, 2), **stats}
    return matrix, {"status": "aligned", "sample_time_sec": round(second, 2), **stats}
