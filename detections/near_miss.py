"""Track-based near-miss detection for a fixed traffic camera.

A near miss needs both an evasive manoeuvre and a nearby road user in the
vehicle's recent path. Track positions alone are too noisy to prove contact,
so pairs with sustained overlap are discarded rather than called near misses.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from itertools import combinations

import numpy as np

try:
    from .classes import BICYCLE, BUS, CAR, MOTORCYCLE, PERSON, TRUCK
except ImportError:
    from classes import BICYCLE, BUS, CAR, MOTORCYCLE, PERSON, TRUCK


ROAD_USERS = {BICYCLE, BUS, CAR, MOTORCYCLE, PERSON, TRUCK}
VEHICLES = {BICYCLE, BUS, CAR, MOTORCYCLE, TRUCK}


class NearMissDetector:
    def __init__(self):
        self.tracks = {}
        self.pairs = {}

    @staticmethod
    def _motion(samples, now):
        """Use a half-second displacement to suppress detector box jitter."""
        if len(samples) < 2:
            return None
        old = min(list(samples)[:-1], key=lambda item: abs(now - item[0] - 0.55))
        dt = now - old[0]
        return (samples[-1][1] - old[1]) / dt if dt >= 0.35 else None

    @staticmethod
    def _threat(a, b):
        """Check recent separation and projected closest approach."""
        # Image-space paths can appear to cross while the users occupy
        # different depths (for example, a car below a sidewalk pedestrian).
        if abs(a["point"][1] - b["point"][1]) > 1.2 * max(a["height"], b["height"]):
            return False
        scale = max(10.0, min(a["width"], b["width"]) if b["class"] != PERSON else a["width"])
        relative = b["point"] - a["point"]
        velocity = a["before"] - b["velocity"]
        speed = np.linalg.norm(velocity)
        if speed < 0.25 * scale:
            return False
        approach = float(np.dot(relative, velocity))
        if approach <= 0:
            return False
        t_closest = min(1.25, approach / (speed * speed))
        separation = float(np.linalg.norm(relative))
        closest = float(np.linalg.norm(relative - t_closest * velocity))
        return separation < 3.0 * scale and closest < 0.85 * scale

    def update(self, detections, t_sec):
        """Return (participant mask, completed segments) for one sampled frame."""
        count = len(detections.xyxy)
        mask = np.zeros(count, dtype=bool)
        boxes = np.asarray(detections.xyxy, dtype=float)
        classes = np.asarray(detections.class_id)
        ids = detections.tracker_id
        ids = np.full(count, -1, dtype=int) if ids is None else np.asarray(ids)
        current = {}

        for index, (box, cls, track_id) in enumerate(zip(boxes, classes, ids)):
            if int(cls) not in ROAD_USERS or track_id < 0:
                continue
            track_id = int(track_id)
            width = max(1.0, float(box[2] - box[0]))
            height = max(1.0, float(box[3] - box[1]))
            point = np.array([(box[0] + box[2]) / 2, box[3]], dtype=float)
            history = self.tracks.setdefault(track_id, deque(maxlen=12))
            if history and t_sec - history[-1][0] > 1.0:
                history.clear()
            previous_velocity = self._motion(history, history[-1][0]) if history else None
            history.append((float(t_sec), point, width))
            velocity = self._motion(history, t_sec)
            if velocity is None:
                continue
            before = previous_velocity if previous_velocity is not None else velocity
            old_speed = float(np.linalg.norm(before))
            speed = float(np.linalg.norm(velocity))
            brake = old_speed >= 0.85 * width and speed <= 0.52 * old_speed
            turn = (old_speed >= 0.65 * width and speed >= 0.4 * old_speed
                    and float(np.dot(before, velocity)) < 0.72 * old_speed * speed)
            current[track_id] = {
                "index": index, "point": point, "width": width, "height": height,
                "class": int(cls),
                "velocity": velocity, "before": before,
                "evasive": int(cls) in VEHICLES and (brake or turn),
                "box": box,
            }

        completed = []
        for first_id, second_id in combinations(current, 2):
            first, second = current[first_id], current[second_id]
            pair = (first_id, second_id)
            state = self.pairs.get(pair)
            if state is None and not (first["evasive"] or second["evasive"]):
                continue
            threatened = ((first["evasive"] and self._threat(first, second))
                          or (second["evasive"] and self._threat(second, first)))
            if state is None:
                if not threatened:
                    continue
                self.pairs[pair] = {"start": float(t_sec), "last": float(t_sec), "overlap": 0}
                state = self.pairs[pair]
            distance = float(np.linalg.norm(first["point"] - second["point"]))
            scale = max(first["width"], second["width"])
            if distance < 0.22 * scale:
                state["overlap"] += 1
            elif distance > 1.8 * scale:
                # Both road users have passed the conflict point.
                self.pairs.pop(pair)
                if state["overlap"] < 2 and t_sec > state["start"]:
                    completed.append([round(state["start"], 2), round(t_sec, 2), "near_miss"])
                continue
            state["last"] = float(t_sec)
            mask[[first["index"], second["index"]]] = True

        for pair, state in list(self.pairs.items()):
            if t_sec - state["last"] > 0.8:
                self.pairs.pop(pair)
                if state["overlap"] < 2 and state["last"] > state["start"]:
                    completed.append([round(state["start"], 2),
                                      round(state["last"], 2), "near_miss"])
        for track_id, samples in list(self.tracks.items()):
            if t_sec - samples[-1][0] > 1.0:
                self.tracks.pop(track_id)
        return mask, completed

    def finish(self, end_sec):
        events = []
        for state in self.pairs.values():
            if state["overlap"] < 2 and end_sec > state["start"]:
                events.append([round(state["start"], 2), round(end_sec, 2), "near_miss"])
        self.pairs.clear()
        return events


def evaluate_near_miss_spatial(detections, state_history, t_sec):
    """Evaluate one tracked frame; return participant mask and closed events.

    ``state_history`` is a :class:`NearMissDetector` kept across frames.
    """
    return state_history.update(detections, t_sec)


def detect_near_miss_events(
    video_path: str,
    model_path: str = "weights/best.pt",
    device: str = "cpu",
    save_video: bool = False,
    output_path: str = "output_near_miss.mp4",
    min_duration: float = 0.0,
) -> list[list]:
    """Detect near misses in a video, optionally saving an annotated copy."""
    import cv2
    import supervision as sv
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(fps / 4.0))
    model = YOLO(model_path)
    state = NearMissDetector()
    events = []
    writer = None
    if save_video:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to write video: {output_path}")
    frame_idx = 0
    marked_boxes = np.empty((0, 4), dtype=float)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                t_sec = frame_idx / fps
                result = model.track(
                    frame, device=device, tracker="bytetrack.yaml", persist=True,
                    conf=0.3, imgsz=640, verbose=False,
                )[0]
                detections = sv.Detections.from_ultralytics(result)
                mask, completed = evaluate_near_miss_spatial(detections, state, t_sec)
                events.extend(completed)
                marked_boxes = np.asarray(detections.xyxy[mask], dtype=float).copy()
            if writer is not None:
                for box in marked_boxes:
                    x1, y1, x2, y2 = np.rint(box).astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "NEAR MISS", (x1, max(25, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                writer.write(frame)
            frame_idx += 1
    finally:
        events.extend(state.finish(frame_idx / fps))
        cap.release()
        if writer is not None:
            writer.release()

    merged = []
    for start, end, label in sorted(events, key=lambda event: event[0]):
        if merged and start - merged[-1][1] < 1.5:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end, label])
    return [event for event in merged if event[1] - event[0] >= min_duration]


def main():
    parser = argparse.ArgumentParser(description="Detect near-miss events")
    parser.add_argument("video_path")
    parser.add_argument("--model", default="weights/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output", default="output_near_miss.mp4")
    parser.add_argument("--min_duration", type=float, default=0.0)
    args = parser.parse_args()
    events = detect_near_miss_events(
        args.video_path, model_path=args.model, device=args.device,
        save_video=args.save_video, output_path=args.output,
        min_duration=args.min_duration,
    )
    print(f"Total near-miss events found: {len(events)}")
    print(json.dumps(events, indent=2))


if __name__ == "__main__":
    main()
