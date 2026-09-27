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
        boxes = [[left_x - 3, y - 10, left_x + 3, y]
                 for y in (20, 35, 50, 65, 80)]
        if include_right:
            boxes.extend([[right_x - 3, y - 10, right_x + 3, y]
                          for y in (20, 35, 50, 65, 80)])
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

    def test_each_lane_in_direction_group_needs_five_vehicles(self):
        self.scene.intersections = {}
        self.scene.road_zone = None
        history = {}
        full = self.detections()
        keep = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
        detections = SimpleNamespace(
            tracker_id=full.tracker_id[keep],
            class_id=full.class_id[keep],
            xyxy=full.xyxy[keep],
        )
        for t in (0.0, 0.5, 1.0):
            congested, _, zones = evaluate_congestion_spatial(
                detections, self.scene, history, t
            )
        self.assertFalse(congested)
        self.assertFalse(zones)

    def test_lane_2_uses_same_five_vehicle_minimum_as_other_lanes(self):
        self.scene.lanes = {"lane_2": self.scene.lanes["left"]}
        self.scene.lane_directions = {"lane_2": np.array([0.0, 1.0])}
        self.scene.intersections = {}
        history = {}

        def detections(car_count):
            boxes = [[17, 10 + i * 10, 23, 20 + i * 10]
                     for i in range(car_count + 1)]
            return SimpleNamespace(
                tracker_id=np.arange(1, len(boxes) + 1),
                class_id=np.array([1] + [2] * car_count),
                xyxy=np.asarray(boxes, dtype=float),
            )

        for t in (0.0, 0.5, 1.0):
            congested, _, zones = evaluate_congestion_spatial(
                detections(3), self.scene, history, t
            )
        self.assertFalse(congested)
        self.assertNotIn(("lane", "lane_2"), zones)

        for t in (1.5, 2.0, 2.5):
            congested, _, zones = evaluate_congestion_spatial(
                detections(4), self.scene, history, t
            )
        self.assertTrue(congested)
        self.assertIn(("lane", "lane_2"), zones)

    def test_five_spread_out_slow_vehicles_are_not_a_lane_queue(self):
        self.scene.lanes = {"lane_2": self.scene.lanes["left"]}
        self.scene.lane_directions = {"lane_2": np.array([0.0, 1.0])}
        self.scene.intersections = {}
        boxes = np.asarray([[17, y - 5, 23, y] for y in (10, 28, 46, 64, 82)], dtype=float)
        detections = SimpleNamespace(
            tracker_id=np.arange(1, 6), class_id=np.full(5, 2), xyxy=boxes
        )
        history = {}
        for t in (0.0, 0.5, 1.0):
            congested, _, zones = evaluate_congestion_spatial(
                detections, self.scene, history, t
            )
        self.assertFalse(congested)
        self.assertNotIn(("lane", "lane_2"), zones)

    def test_side_by_side_slow_vehicles_are_not_a_lane_queue(self):
        self.scene.lanes = {"lane_2": self.scene.lanes["left"]}
        self.scene.lane_directions = {"lane_2": np.array([0.0, 1.0])}
        self.scene.intersections = {}
        boxes = np.asarray([[x - 3, 40, x + 3, 50] for x in (5, 14, 23, 32, 41)], dtype=float)
        detections = SimpleNamespace(
            tracker_id=np.arange(1, 6), class_id=np.full(5, 2), xyxy=boxes
        )
        history = {}
        for t in (0.0, 0.5, 1.0):
            congested, _, zones = evaluate_congestion_spatial(
                detections, self.scene, history, t
            )
        self.assertFalse(congested)
        self.assertNotIn(("lane", "lane_2"), zones)

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
                          ((10, 25), (30, 35), (50, 45), (70, 55), (90, 65))],
                         dtype=float)
        detections = SimpleNamespace(
            tracker_id=np.arange(1, 6), class_id=np.full(5, 2), xyxy=boxes
        )
        history = {}
        for t in (0.0, 0.5, 1.0):
            congested, _, _ = evaluate_congestion_spatial(
                detections, self.scene, history, t
            )
        self.assertTrue(congested)

    def test_broad_road_spread_does_not_bypass_queue_check(self):
        self.scene.lanes = {}
        self.scene.lane_directions = {}
        self.scene.intersections = {}
        boxes = np.array([[x - 3, y - 4, x + 3, y] for x, y in
                          ((10, 25), (30, 35), (50, 45), (70, 55), (90, 65))],
                         dtype=float)
        detections = SimpleNamespace(
            tracker_id=np.arange(1, 6), class_id=np.full(5, 2), xyxy=boxes
        )
        history = {}
        for t in (0.0, 0.5, 1.0):
            congested, _, _ = evaluate_congestion_spatial(
                detections, self.scene, history, t
            )
        self.assertFalse(congested)

    def test_intersection_congestion_without_lanes(self):
        self.scene.lanes = {}
        self.scene.lane_directions = {}
        boxes = np.array([[x - 3, 40, x + 3, 60] for x in (20, 35, 50, 65, 80)], dtype=float)
        detections = SimpleNamespace(
            tracker_id=np.arange(1, 6), class_id=np.full(5, 2), xyxy=boxes
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
