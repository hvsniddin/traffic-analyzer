"""Causal collision risk from tracked road users (Part B).

Each pair of road users that is closing in on each other gets a risk from
its projected closest approach and time to reach it (constant-velocity
time-to-collision in image space). Sharp braking or swerving by one of the
pair raises it further. The frame score is the worst pair, smoothed over
the last second so single-frame tracker glitches cannot raise an alarm.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np

try:
    from .classes import BICYCLE, BUS, CAR, MOTORCYCLE, PERSON, TRUCK
except ImportError:
    from classes import BICYCLE, BUS, CAR, MOTORCYCLE, PERSON, TRUCK

ROAD_USERS = {BICYCLE, BUS, CAR, MOTORCYCLE, PERSON, TRUCK}
VEHICLES = {BICYCLE, BUS, CAR, MOTORCYCLE, TRUCK}
HORIZON_SEC = 5.0
VELOCITY_WINDOW_SEC = 0.6
# Closest approach below this share of the larger width counts as a conflict.
# Values were set on 45 s of C3905 so normal traffic stays well under 0.5.
CONFLICT_RATIO = 0.3
# Relative speed (widths per second) below which a pair is queuing or passing
# a stopped vehicle in the next lane rather than on a collision course.
MIN_CLOSING_SPEED = 2.0
SMOOTH_SEC = 1.0
EVASIVE_BONUS = 0.25
# Two moving users on near-parallel headings are in adjacent lanes or
# following each other; only crossing paths count unless one brakes hard.
PARALLEL_COS = 0.85
MOVING_SPEED = 0.3
# Boxes this close already overlap (queues, occlusion); not a future contact.
MIN_SEPARATION_RATIO = 0.3


class CollisionRisk:
    def __init__(self):
        self.tracks: dict[int, deque] = {}
        self.recent: deque = deque()

    def _velocity(self, samples):
        latest_t, latest_point = samples[-1]
        old = min(list(samples)[:-1], key=lambda s: abs(latest_t - s[0] - VELOCITY_WINDOW_SEC))
        dt = latest_t - old[0]
        return (latest_point - old[1]) / dt if dt >= 0.3 else None

    def update(self, detections, t_sec: float) -> float:
        boxes = np.asarray(detections.xyxy, dtype=float)
        classes = np.asarray(detections.class_id)
        ids = detections.tracker_id
        users = []
        if ids is not None:
            for box, cls, track_id in zip(boxes, classes, ids):
                if int(cls) not in ROAD_USERS or track_id < 0:
                    continue
                point = np.array([(box[0] + box[2]) / 2, box[3]], dtype=float)
                history = self.tracks.setdefault(int(track_id), deque(maxlen=16))
                if history and t_sec - history[-1][0] > 1.0:
                    history.clear()
                previous = self._velocity(history) if len(history) >= 2 else None
                history.append((float(t_sec), point))
                velocity = self._velocity(history) if len(history) >= 2 else None
                if velocity is None:
                    continue
                width = max(1.0, float(box[2] - box[0]))
                height = max(1.0, float(box[3] - box[1]))
                evasive = False
                if previous is not None and int(cls) in VEHICLES:
                    old_speed = float(np.linalg.norm(previous))
                    speed = float(np.linalg.norm(velocity))
                    evasive = old_speed >= 0.85 * width and (
                        speed <= 0.5 * old_speed
                        or float(np.dot(previous, velocity)) < 0.7 * old_speed * speed)
                users.append({"cls": int(cls), "point": point, "velocity": velocity,
                              "width": width, "height": height, "evasive": evasive})

        raw = 0.0
        for a, b in combinations(users, 2):
            if a["cls"] not in VEHICLES and b["cls"] not in VEHICLES:
                continue
            scale = max(a["width"], b["width"])
            relative = b["point"] - a["point"]
            if float(np.linalg.norm(relative)) < MIN_SEPARATION_RATIO * scale:
                continue
            speed_a = float(np.linalg.norm(a["velocity"]))
            speed_b = float(np.linalg.norm(b["velocity"]))
            evasive = a["evasive"] or b["evasive"]
            if (not evasive and speed_a >= MOVING_SPEED * a["width"]
                    and speed_b >= MOVING_SPEED * b["width"]
                    and abs(float(np.dot(a["velocity"], b["velocity"]))) >= PARALLEL_COS * speed_a * speed_b):
                continue
            closing = b["velocity"] - a["velocity"]
            speed_sq = float(np.dot(closing, closing))
            if speed_sq < (MIN_CLOSING_SPEED * scale) ** 2:
                continue
            t_closest = -float(np.dot(relative, closing)) / speed_sq
            if not 0.0 < t_closest < HORIZON_SEC:
                continue
            closest = float(np.linalg.norm(relative + t_closest * closing))
            if closest >= CONFLICT_RATIO * scale:
                continue
            pair = (1.0 - t_closest / HORIZON_SEC) * (1.0 - closest / (CONFLICT_RATIO * scale))
            if evasive:
                pair += EVASIVE_BONUS
            raw = max(raw, min(1.0, pair))

        self.recent.append((t_sec, raw))
        while self.recent and t_sec - self.recent[0][0] > SMOOTH_SEC:
            self.recent.popleft()
        for track_id, samples in list(self.tracks.items()):
            if t_sec - samples[-1][0] > 1.0:
                del self.tracks[track_id]
        # The mean over one second keeps a lone spike below the 0.5 alarm line.
        return float(np.mean([score for _, score in self.recent]))
