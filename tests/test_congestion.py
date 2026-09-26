import unittest
from types import SimpleNamespace

import numpy as np

from detections.congestion import (
    annotate_frame,
    evaluate_congestion_spatial,
    update_congestion_event,
)


class CongestionTests(unittest.TestCase):
    def setUp(self):
        self.scene = SimpleNamespace(
            frame_width=100,
            frame_height=100,
            road_zone=SimpleNamespace(
                polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
            ),
            lanes={
                "left": SimpleNamespace(polygon=np.array([[0, 0], [50, 0], [50, 100], [0, 100]])),
                "right": SimpleNamespace(polygon=np.array([[50, 0], [100, 0], [100, 100], [50, 100]])),
            },
            lane_directions={"left": np.array([0.0, 1.0]), "right": np.array([0.0, 1.0])},
            intersections={
                "junction": SimpleNamespace(
                    polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
                )
            },
        )

    def detections(self, left_x=20, right_x=70, include_right=True):
        boxes = [[left_x - 5, 40, left_x + 5, 60]]
        if include_right:
            boxes.append([right_x - 5, 40, right_x + 5, 60])
        return SimpleNamespace(
            tracker_id=np.arange(1, len(boxes) + 1),
            class_id=np.full(len(boxes), 2),
            xyxy=np.asarray(boxes, dtype=float),
        )

    def test_all_lanes_must_be_slow_and_occupied(self):
        history = {}
        congested = False
        for t in (0.0, 0.5, 1.0, 1.5):
            congested, _, zones = evaluate_congestion_spatial(
                self.detections(), self.scene, history, t
            )
        self.assertTrue(congested)
        self.assertIn(("lane", "left"), zones)
        self.scene.intersections = {}
        congested, _, _ = evaluate_congestion_spatial(
            self.detections(include_right=False), self.scene, history, 2.0
        )
        self.assertFalse(congested)

    def test_moving_lane_prevents_congestion(self):
        history = {}
        for t, x in ((0.0, 70), (0.5, 78), (1.0, 86), (1.5, 94)):
            congested, _, _ = evaluate_congestion_spatial(
                self.detections(right_x=x), self.scene, history, t
            )
        self.assertFalse(congested)

    def test_event_starts_at_queue_stop_and_ends_when_it_clears(self):
        state = {"candidate_since": None, "active_start": None, "clear_since": None}
        events = []
        for t in range(7):
            events.extend(update_congestion_event(state, 1 <= t < 5, float(t)))
        self.assertEqual(events, [[0.2, 5.0, "congestion"]])

    def test_broad_queue_beyond_annotated_lanes(self):
        self.scene.lanes = {}
        self.scene.lane_directions = {}
        self.scene.intersections = {}
        boxes = np.array([[x - 3, y - 10, x + 3, y] for x, y in
                          ((10, 25), (25, 35), (40, 45), (55, 55), (70, 65), (85, 75))],
                         dtype=float)
        detections = SimpleNamespace(
            tracker_id=np.arange(1, 7), class_id=np.full(6, 2), xyxy=boxes
        )
        history = {}
        for t in (0.0, 0.5, 1.0):
            congested, _, _ = evaluate_congestion_spatial(
                detections, self.scene, history, t
            )
        self.assertTrue(congested)

    def test_intersection_congestion_without_lanes(self):
        self.scene.lanes = {}
        self.scene.lane_directions = {}
        boxes = np.array([[x - 3, 40, x + 3, 60] for x in (20, 50, 80)], dtype=float)
        detections = SimpleNamespace(
            tracker_id=np.arange(1, 4), class_id=np.full(3, 2), xyxy=boxes
        )
        history = {}
        for t in (0.0, 0.5, 1.0):
            congested, _, zones = evaluate_congestion_spatial(
                detections, self.scene, history, t
            )
        self.assertTrue(congested)
        self.assertEqual(zones, {("intersection", "junction")})

    def test_annotation_colors_only_congested_zone_red(self):
        scene = SimpleNamespace(
            lanes={"lane": SimpleNamespace(
                polygon=np.array([[5, 5], [35, 5], [35, 35], [5, 35]])
            )},
            intersections={"junction": SimpleNamespace(
                polygon=np.array([[60, 60], [90, 60], [90, 90], [60, 90]])
            )},
        )
        frame = annotate_frame(
            np.zeros((100, 100, 3), dtype=np.uint8),
            SimpleNamespace(tracker_id=None), set(), {("intersection", "junction")},
            scene, {},
        )
        self.assertGreater(frame[5, 20, 1], frame[5, 20, 2])
        self.assertGreater(frame[60, 75, 2], frame[60, 75, 1])
        self.assertGreater(frame[80, 50, 2], 0)  # Red label background.


if __name__ == "__main__":
    unittest.main()
