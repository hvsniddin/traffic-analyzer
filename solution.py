"""
Submission entry point, imported by the organizers' run_submission.py:

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

Part A runs one detector + ByteTrack pass at DEFAULT_INFERENCE_FPS and feeds
the tracks to the rule modules in detections/. Part B scores pairwise
time-to-collision with detections/risk.py. See README.md for the approach.
"""
from __future__ import annotations

import os
import random
import time
from typing import Callable

import numpy as np

SEED = 0
DEFAULT_INFERENCE_FPS = 2.0
# The harness scores a video as empty past 3x its duration, so Part A stops
# early and keeps what it has found once this share of the budget is used.
# Model loading and alignment are inside the budget; past the guard only one
# sample and event closing remain (<1 s on GPU, a few seconds on CPU). The
# absolute margin protects short clips where 0.3x is only a second or two.
TIME_BUDGET_FACTOR = float(os.getenv("WIUT_TIME_BUDGET_FACTOR", "2.7"))
MIN_TIME_MARGIN_SEC = 3.0


# Pedestrians drop out of tracking for a few seconds at a time; one person on
# the road is one annotated segment. Tuned on our dev labels (see README).
DEFAULT_MERGE_GAP_SEC = 1.5
MERGE_GAP_SEC = {"jaywalking": 8.0, "stopped_vehicle": 5.0, "wrong_way": 3.0}
# Computed but not reported: on our dev labels these produced only false
# positives, and a predicted class absent from the test set scores 0 in the
# macro mean. congestion fires on red-light queues (37 FP / 1 GT); wrong_way
# fires on normal flow the lane_2 polygon does not capture (43 FP / 0 GT);
# stopped_vehicle had 10 FP / 0 GT. congestion also feeds stopped_vehicle.
SUPPRESSED_CLASSES = {"congestion", "wrong_way", "stopped_vehicle"}
# Final pass after merging: sub-second blips are mostly tracker noise. On dev
# labels this raised Score A 0.104 -> 0.115 and F1@0.7 0.069 -> 0.078; longer
# minimums would drop real 1 s events (red_light, failure_to_yield).
MIN_EVENT_SEC = 1.0

HARNESS_TIME_FACTOR = 3.0
# Part B runs detection at this rate; step() returns the last score in between.
RISK_INFERENCE_FPS = 5.0
# Reading a frame (decode + BGR conversion) is slower than Part A's grab().
HARNESS_READ_SLOWDOWN = 2.0
RISK_SAFETY_MARGIN = 0.85
_PART_A_TIMING: dict = {}


def _seed_everything(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _device() -> str:
    """First GPU when torch sees CUDA (the organizers' T4), otherwise CPU."""
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"

# Official class ids (14). See the task description for definitions and
# start/end conventions. Remove entries you never predict; never add.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

# Anticipation horizon used by the metric (seconds). step() should return
# P(an `accident` starts within the next RISK_HORIZON_SEC seconds).
RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection.

    Args:
        video_path: path to one .mp4 file.

    Returns:
        A list of events, each ``[start_sec, end_sec, label]`` with
        ``0 <= start_sec < end_sec <= duration`` (floats, seconds from the
        first frame) and ``label in CLASSES``. Return ``[]`` if nothing
        happened. Segments of the same class must not overlap.

    Samples frames at DEFAULT_INFERENCE_FPS, detects and tracks road users,
    turns tracks and the aligned scene map into per-track flags per class,
    then merges flags into segments (MERGE_GAP_SEC), drops segments shorter
    than MIN_EVENT_SEC, and leaves out SUPPRESSED_CLASSES.
    """
    return _detect_events(video_path)


def _detect_events(
    video_path: str,
    on_sample: Callable | None = None,
    sample_rate: float | None = None,
    time_budget_factor: float | None = None,
) -> list[list]:
    """Run detection, optionally exposing each sampled frame to the annotator.

    ``sample_rate`` and ``time_budget_factor`` let the demo API trade speed
    for detail; the submission path uses the defaults.
    """
    from pathlib import Path

    import cv2
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
    from detections.near_miss import NearMissDetector, evaluate_near_miss_spatial
    from detections.stopped_vehicle import evaluate_stopped_vehicle_spatial
    from detections.wrong_way import evaluate_wrong_way_spatial

    started = time.perf_counter()
    _seed_everything()
    root = Path(__file__).resolve().parent
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else float("inf")
    # One rate on every device keeps CPU and GPU output identical, so
    # predictions_samples.json can be reproduced anywhere. 2 FPS gives the
    # velocity-based rules samples under 1 s apart and fits 3x even on CPU.
    target_rate = sample_rate or float(os.getenv("WIUT_INFERENCE_FPS", str(DEFAULT_INFERENCE_FPS)))
    stride = max(1, round(fps / target_rate))
    sample_period = stride / fps
    # Below ~1 FPS, seeking to each sample beats decoding every 4K frame.
    seek_samples = target_rate < 1.0 and frame_count > 0
    factor = time_budget_factor or TIME_BUDGET_FACTOR
    # An infinite factor (offline export, demo) disables the guard entirely.
    deadline = started + (factor * duration if factor == float("inf") else
                          min(factor * duration, HARNESS_TIME_FACTOR * duration - MIN_TIME_MARGIN_SEC))
    device = _device()
    print(f"[solution] {Path(video_path).name}: device={device}, {target_rate:g} FPS sampling", flush=True)

    try:
        scene = SceneGeometry(str(root / "scene.json"), width, height, video_path=video_path)
        model = YOLO(str(root / "weights" / "best.pt"))
        histories = {
            "traffic_light": {}, "failure_to_yield": {},
            "solid_line_crossing": {}, "illegal_turn": {}, "congestion": {},
            "stopped_vehicle": {}, "wrong_way": {},
        }
        near_miss = NearMissDetector()
        active: dict[tuple[str, int], tuple[float, float]] = {}
        events: list[list] = []
        frame_index = 0
        last_time = 0.0

        def close_expired(now: float, force: bool = False) -> None:
            for key, (start, last) in list(active.items()):
                if not force and now - last <= max(0.8, 2 * sample_period):
                    continue
                active.pop(key)
                end = min(duration, last + sample_period)
                minimum = 0.25 if key[0] in {"red_light", "solid_line_crossing", "illegal_turn"} else 0.5
                if end - start >= minimum:
                    events.append([start, end, key[0]])

        loop_started = time.perf_counter()
        processing_sec = 0.0
        processed = 0
        while frame_count <= 0 or frame_index < frame_count:
            if time.perf_counter() > deadline:
                # Partial events score better than a video emptied by the harness.
                break
            if seek_samples:
                # Random access avoids decoding every skipped 4K frame on CPU.
                # It also recovers after damaged H.264 packets in C3896.
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                sampled_index = frame_index
                frame_index += stride
                ok, frame = capture.read()
                if not ok:
                    continue
            else:
                # Sequential decoding is faster with a GPU's denser sampling.
                if frame_index % stride:
                    if capture.grab():
                        frame_index += 1
                    else:
                        next_index = ((frame_index // stride) + 1) * stride
                        if frame_count <= 0 or next_index >= frame_count:
                            break
                        capture.set(cv2.CAP_PROP_POS_FRAMES, next_index)
                        frame_index = next_index
                    continue
                sampled_index = frame_index
                frame_index += 1
                ok, frame = capture.read()
                if not ok:
                    next_index = ((frame_index // stride) + 1) * stride
                    if frame_count <= 0 or next_index >= frame_count:
                        break
                    capture.set(cv2.CAP_PROP_POS_FRAMES, next_index)
                    frame_index = next_index
                    continue
            sample_started = time.perf_counter()
            t_sec = sampled_index / fps
            last_time = t_sec
            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml", conf=0.3,
                imgsz=640, device=device, verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            ids = detections.tracker_id
            if ids is None:
                ids = np.full(len(detections), -1, dtype=int)

            masks = {}
            masks["red_light"], masks["stop_line"] = evaluate_traffic_light_events(
                detections, scene, histories["traffic_light"], t_sec
            )
            event_starts = histories["traffic_light"]["event_starts"]
            for track_id in ids[masks["stop_line"]]:
                active.pop(("red_light", int(track_id)), None)
            people = detections[detections.class_id == PERSON]
            people_ids = ids[detections.class_id == PERSON]
            jaywalkers = evaluate_jaywalking_spatial(people, scene)[0]
            masks["failure_to_yield"] = evaluate_failure_to_yield_spatial(
                detections, scene, histories["failure_to_yield"], t_sec
            )
            masks["solid_line_crossing"] = evaluate_solid_line_crossing_spatial(
                detections, scene, histories["solid_line_crossing"]
            )
            masks["illegal_turn"] = evaluate_illegal_turn_spatial(
                detections, scene, histories["illegal_turn"], t_sec
            )
            masks["near_miss"], completed_near_misses = evaluate_near_miss_spatial(
                detections, near_miss, t_sec
            )
            events.extend(completed_near_misses)
            congested, _, congested_zones = evaluate_congestion_spatial(
                detections, scene, histories["congestion"], t_sec
            )
            masks["stopped_vehicle"] = evaluate_stopped_vehicle_spatial(
                detections, scene, histories["stopped_vehicle"], t_sec,
                signal_is_red=histories["traffic_light"].get("signal_is_red", False),
                congested=congested,
            )
            masks["wrong_way"] = evaluate_wrong_way_spatial(
                detections, scene, histories["wrong_way"], t_sec
            )
            rule_starts = {
                **event_starts,
                **histories["stopped_vehicle"]["starts"],
                **histories["wrong_way"]["starts"],
            }
            if on_sample is not None:
                jaywalking_mask = np.zeros(len(detections), dtype=bool)
                jaywalking_mask[np.flatnonzero(detections.class_id == PERSON)] = jaywalkers
                on_sample(
                    sampled_index, detections, ids,
                    {**masks, "jaywalking": jaywalking_mask},
                    congested, congested_zones, scene,
                )
            if congested:
                key = ("congestion", 0)
                start = active.get(key, (t_sec, t_sec))[0]
                active[key] = (start, t_sec)

            for label, mask in masks.items():
                if label == "near_miss":
                    continue
                for track_id in ids[np.asarray(mask, dtype=bool)]:
                    if track_id < 0:
                        continue
                    key = (label, int(track_id))
                    start = active.get(key, (rule_starts.get(key, t_sec), t_sec))[0]
                    active[key] = (start, t_sec)
            for track_id in people_ids[jaywalkers]:
                if track_id < 0:
                    continue
                key = ("jaywalking", int(track_id))
                start = active.get(key, (t_sec, t_sec))[0]
                active[key] = (start, t_sec)
            close_expired(t_sec)
            processing_sec += time.perf_counter() - sample_started
            processed += 1

        close_expired(min(duration, last_time + sample_period), force=True)
        # Timing only (no events) for RiskEstimator's decision to run in budget.
        decode_sec = time.perf_counter() - loop_started - processing_sec
        _PART_A_TIMING.clear()
        _PART_A_TIMING.update({
            "video_id": Path(video_path).name,
            "started": started,
            "duration": duration,
            "sequential": not seek_samples,
            "decode_fps": frame_index / decode_sec if decode_sec > 0 else 0.0,
            "sample_sec": processing_sec / processed if processed else 0.0,
        })
        events.extend(near_miss.finish(min(duration, last_time + sample_period)))
    finally:
        capture.release()

    # Merge nearby detections of the same class, including adjacent tracks.
    merged = []
    for start, end, label in sorted(events, key=lambda item: (item[2], item[0])):
        if label in SUPPRESSED_CLASSES:
            continue
        gap = MERGE_GAP_SEC.get(label, DEFAULT_MERGE_GAP_SEC)
        if merged and merged[-1][2] == label and start - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, label])
    kept = [event for event in merged if event[1] - event[0] >= MIN_EVENT_SEC]
    return sorted(kept, key=lambda item: item[0])


class RiskEstimator:
    """Part B — causal accident anticipation.

    Runs its own detector and tracker on every ``fps / RISK_INFERENCE_FPS``-th
    frame it receives and scores pairwise time-to-collision
    (``detections.risk``). It never opens the video and never reads Part A
    events; from Part A it only takes wall-clock timing, to decide whether
    the harness's full-video decode still fits the 3x budget. If it would
    not, ``reset`` raises: the harness logs it and keeps Part A's events,
    which is worth more than a risk curve on a video scored as empty.
    """

    def reset(self, meta: dict) -> None:
        """Called once before the first frame of each video.

        meta = {"video_id": str, "fps": float, "width": int, "height": int,
                "n_frames": int}
        """
        from pathlib import Path

        from ultralytics import YOLO

        from detections.risk import CollisionRisk

        fps = float(meta.get("fps") or 25.0)
        n_frames = int(meta.get("n_frames") or 0)
        self.stride = max(1, round(fps / RISK_INFERENCE_FPS))
        self.score = 0.0
        self.frame_index = 0
        self.n_frames = n_frames
        mode = os.getenv("WIUT_RISK", "auto")
        if mode == "0":
            raise RuntimeError("Part B disabled by WIUT_RISK=0")

        timing = _PART_A_TIMING if _PART_A_TIMING.get("video_id") == meta.get("video_id") else {}
        now = time.perf_counter()
        if timing:
            self.read_fps = timing["decode_fps"] / HARNESS_READ_SLOWDOWN
            self.budget_end = timing["started"] + RISK_SAFETY_MARGIN * HARNESS_TIME_FACTOR * timing["duration"]
            sample_sec = timing["sample_sec"]
        else:
            self.read_fps, self.budget_end, sample_sec = 0.0, float("inf"), 0.0
        if mode != "1":
            if _device() == "cpu" or not timing.get("sequential") or self.read_fps <= 0:
                raise RuntimeError("Part B skipped: needs a GPU run of Part A on this video first")
            needed = n_frames / self.read_fps + (n_frames / self.stride) * sample_sec
            if now + needed > self.budget_end:
                raise RuntimeError(
                    f"Part B skipped: needs ~{needed:.0f}s, "
                    f"{max(0.0, self.budget_end - now):.0f}s left in the time budget")

        _seed_everything()
        root = Path(__file__).resolve().parent
        self.model = YOLO(str(root / "weights" / "best.pt"))
        self.device = _device()
        self.risk = CollisionRisk()

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within the next RISK_HORIZON_SEC s).

        Args:
            frame: BGR uint8 array of shape (H, W, 3) — OpenCV convention.
            t_sec: timestamp of this frame in seconds.

        Returns:
            A float in [0, 1]. Skipping frames internally and returning the
            previous score is fine; the harness still expects a value for
            every call.
        """
        import supervision as sv

        index = self.frame_index
        self.frame_index += 1
        if index % self.stride:
            return self.score
        # Behind schedule: stop inferring so the remaining decode still fits.
        if self.read_fps > 0:
            remaining = max(0, self.n_frames - index) / self.read_fps
            if time.perf_counter() + remaining > self.budget_end:
                return self.score
        result = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", conf=0.3,
            imgsz=640, device=self.device, verbose=False,
        )[0]
        self.score = min(1.0, max(0.0, self.risk.update(sv.Detections.from_ultralytics(result), t_sec)))
        return self.score
