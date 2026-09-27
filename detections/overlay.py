"""Collect per-sample boxes, event flags, scene geometry, and risk for the website.

``OverlayRecorder`` is passed to ``solution._detect_events`` as ``on_sample``.
Coordinates are stored as 0-1 fractions of the frame so the browser can draw
them over any rendition of the video. The risk curve feeds the same tracker
output, in time order, through the causal ``CollisionRisk`` model.
"""

from __future__ import annotations

import numpy as np

try:
    from .risk import CollisionRisk
except ImportError:
    from risk import CollisionRisk

SCENE_LAYERS = {
    "lane": "lanes", "crosswalk": "crosswalks", "intersection": "intersections",
    "island": "islands", "waiting_zone": "waiting_zones",
}
LINE_LAYERS = {"stop_line": "stop_lines", "solid_line": "solid_lines"}


def scene_shapes(scene, width: int, height: int) -> list[dict]:
    """Aligned scene geometry in 0-1 coordinates."""
    size = np.array([width, height], dtype=float)
    shapes = []
    for kind, attr in SCENE_LAYERS.items():
        for name, zone in getattr(scene, attr, {}).items():
            points = np.asarray(zone.polygon, dtype=float) / size
            shapes.append({"kind": kind, "name": name, "points": points.round(4).tolist()})
    for kind, attr in LINE_LAYERS.items():
        for name, line in getattr(scene, attr, {}).items():
            vector = line.vector
            points = np.array([[vector.start.x, vector.start.y], [vector.end.x, vector.end.y]]) / size
            shapes.append({"kind": kind, "name": name, "points": points.round(4).tolist()})
    return shapes


class OverlayRecorder:
    def __init__(self, fps: float, width: int, height: int, on_progress=None):
        self.fps = fps
        self.width = width
        self.height = height
        self.on_progress = on_progress
        self.scene: list[dict] = []
        self.frames: list[dict] = []
        self.risk: list[list[float]] = []
        self._risk_model = CollisionRisk()
        self._size = np.array([width, height, width, height], dtype=float)

    def __call__(self, frame_index, detections, ids, masks, congested, congested_zones, scene):
        t_sec = frame_index / self.fps
        if not self.scene:
            self.scene = scene_shapes(scene, self.width, self.height)
        self.risk.append([round(t_sec, 3), round(self._risk_model.update(detections, t_sec), 4)])
        objects = []
        for index, box in enumerate(np.asarray(detections.xyxy, dtype=float) / self._size):
            labels = [label for label, mask in masks.items() if len(mask) and mask[index]]
            objects.append([*box.round(4).tolist(), int(detections.class_id[index]), int(ids[index]), labels])
        self.frames.append({"t": round(t_sec, 3), "objects": objects})
        if self.on_progress is not None:
            self.on_progress(frame_index)

    def overlay(self) -> dict:
        return {"scene": self.scene, "frames": self.frames}
