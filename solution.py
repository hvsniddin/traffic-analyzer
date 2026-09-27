"""
solution.py — the ONLY file a team has to implement.

The organizers' harness (run_submission.py) imports this module and calls:

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B (optional)

Keep the names and signatures exactly as they are. Everything else — models,
tracking, rules, helper modules under src/ — is up to you.

Labels must come from CLASSES. You may REMOVE classes you never predict;
do not add new ids.
"""
from __future__ import annotations

import numpy as np

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
        video_path: path to one .mp4 file. You may open it any way you like
            (OpenCV, decord, PyAV, ffmpeg), read it several times, sample
            frames, run batched models — anything goes.

    Returns:
        A list of events, each ``[start_sec, end_sec, label]`` with
        ``0 <= start_sec < end_sec <= duration`` (floats, seconds from the
        first frame) and ``label in CLASSES``. Return ``[]`` if nothing
        happened. Segments of the same class must not overlap.

    A typical pipeline:
        1. sample frames (every 2nd–5th frame is usually enough),
        2. detect road users (YOLO / RT-DETR) and track them (ByteTrack),
        3. turn trajectories + scene layout (lanes, stop line, crossing)
           into per-frame flags for each class,
        4. merge consecutive flags into segments, drop blips < 0.5 s,
           merge gaps < 1 s,
        5. optionally re-score `accident` / `near_miss` candidates with a
           learned clip classifier.
    """
    import os
    from pathlib import Path

    import cv2
    import supervision as sv
    from ultralytics import YOLO
    import torch

    from detections.scene import SceneGeometry
    from detections.classes import PERSON
    from detections.traffic_light_events import evaluate_traffic_light_events
    from detections.jaywalking import evaluate_jaywalking_spatial
    from detections.failure_to_yield import evaluate_failure_to_yield_spatial
    from detections.solid_line import evaluate_solid_line_crossing_spatial
    from detections.illegal_turn import evaluate_illegal_turn_spatial
    from detections.congestion import evaluate_congestion_spatial

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
    # GPU evaluation can keep short events; CPU inference needs a lower sample
    # rate to fit the organizer's strict 3x wall-clock budget.
    default_rate = 5.0 if torch.cuda.is_available() else 0.5
    target_rate = float(os.getenv("WIUT_INFERENCE_FPS", str(default_rate)))
    stride = max(1, round(fps / target_rate))
    sample_period = stride / fps

    try:
        scene = SceneGeometry(str(root / "scene.json"), width, height, video_path=video_path)
        model = YOLO(str(root / "weights" / "best.pt"))
        histories = {
            "traffic_light": {}, "failure_to_yield": {},
            "solid_line_crossing": {}, "illegal_turn": {}, "congestion": {},
        }
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

        while True:
            # Grabbing skipped frames avoids image decode and model inference.
            if frame_index % stride:
                if not capture.grab():
                    break
                frame_index += 1
                continue
            ok, frame = capture.read()
            if not ok:
                break
            t_sec = frame_index / fps
            last_time = t_sec
            frame_index += 1
            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml", conf=0.3,
                imgsz=640, verbose=False,
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
            congested = evaluate_congestion_spatial(
                detections, scene, histories["congestion"], t_sec
            )[0]
            if congested:
                key = ("congestion", 0)
                start = active.get(key, (t_sec, t_sec))[0]
                active[key] = (start, t_sec)

            for label, mask in masks.items():
                for track_id in ids[np.asarray(mask, dtype=bool)]:
                    if track_id < 0:
                        continue
                    key = (label, int(track_id))
                    start = active.get(key, (event_starts.get(key, t_sec), t_sec))[0]
                    active[key] = (start, t_sec)
            for track_id in people_ids[jaywalkers]:
                if track_id < 0:
                    continue
                key = ("jaywalking", int(track_id))
                start = active.get(key, (t_sec, t_sec))[0]
                active[key] = (start, t_sec)
            close_expired(t_sec)

        close_expired(min(duration, last_time + sample_period), force=True)
    finally:
        capture.release()

    # Merge nearby detections of the same class, including adjacent tracks.
    merged = []
    for start, end, label in sorted(events, key=lambda item: (item[2], item[0])):
        if merged and merged[-1][2] == label and start - merged[-1][1] < 1.0:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, label])
    return sorted(merged, key=lambda item: item[0])


class RiskEstimator:
    """Part B — causal accident anticipation (optional, bonus).

    The harness calls ``reset(meta)`` once per video and then ``step`` for
    EVERY frame, in order. ``step`` must use only the frames it has seen so
    far: do not open the video file inside this class, and do not reuse
    Part A results that were computed with access to future frames.
    """

    def reset(self, meta: dict) -> None:
        """Called once before the first frame of each video.

        meta = {"video_id": str, "fps": float, "width": int, "height": int,
                "n_frames": int}
        """
        # Part B is optional. A constant-zero curve cannot score and decoding
        # the full 4K video again can put Part A over the 3x time budget.
        # The unchanged starter harness catches this and keeps Part A events.
        raise NotImplementedError("Part B risk anticipation is not implemented")

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
        # TODO: replace this stub. A simple strong baseline: track vehicles,
        # estimate time-to-collision between pairs, map min TTC -> risk.
        return 0.0
