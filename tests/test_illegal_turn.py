import unittest
from types import SimpleNamespace

import numpy as np

from detections.illegal_turn import evaluate_illegal_turn_spatial


class Detections(SimpleNamespace):
    def __len__(self):
        return len(self.xyxy)


def rectangle(x1, y1, x2, y2):
    return SimpleNamespace(polygon=np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
    ], dtype=np.float32))


class IllegalTurnTests(unittest.TestCase):
    def setUp(self):
        self.scene = SimpleNamespace(
            lanes={
                "lane_1": rectangle(0, 10, 20, 30),
                "lane_2": rectangle(25, 10, 45, 30),
                "lane_3": rectangle(50, 10, 70, 30),
            },
            intersections={"intersection_1": rectangle(0, 40, 100, 70)},
            crosswalks={"crosswalk_1": rectangle(0, 80, 100, 100)},
        )

    def run_track(self, points):
        states = {}
        masks = []
        for t, (x, y) in enumerate(points):
            detections = Detections(
                xyxy=np.array([[x - 1, y - 1, x + 1, y + 1]], dtype=float),
                class_id=np.array([2]),
                tracker_id=np.array([7]),
            )
            masks.append(bool(evaluate_illegal_turn_spatial(
                detections, self.scene, states, float(t)
            )[0]))
        return masks

    def test_northeast_lane_turn_triggers_after_intersection(self):
        for x in (10, 60):
            with self.subTest(x=x):
                self.assertEqual(self.run_track([(x, 20), (x, 50), (x, 90)]),
                                 [False, False, True])

    def test_lane_2_and_other_approaches_do_not_trigger(self):
        for origin in ((35, 20), (90, 20), (90, 50)):
            with self.subTest(origin=origin):
                self.assertEqual(self.run_track([origin, (35, 50), (35, 90)]),
                                 [False, False, False])

    def test_first_origin_lane_is_retained(self):
        self.assertEqual(self.run_track([(35, 20), (10, 20), (10, 50), (10, 90)]),
                         [False, False, False, False])

    def test_lane_2_takes_priority_at_shared_lane_edges(self):
        self.scene.lanes["lane_2"] = rectangle(20, 10, 45, 30)
        self.assertEqual(self.run_track([(20, 20), (20, 50), (20, 90)]),
                         [False, False, False])


if __name__ == "__main__":
    unittest.main()
