import unittest
from types import SimpleNamespace

import numpy as np

from detections.classes import CAR, PERSON
from detections.failure_to_yield import (
    _close_finished_crossings,
    evaluate_failure_to_yield_spatial,
)


class Detections(SimpleNamespace):
    def __len__(self):
        return len(self.xyxy)


class RectangularCrosswalk:
    def trigger(self, detections):
        boxes = detections.xyxy
        x = (boxes[:, 0] + boxes[:, 2]) / 2
        y = boxes[:, 3]
        return (0 <= x) & (x <= 200) & (0 <= y) & (y <= 100)


class FailureToYieldTests(unittest.TestCase):
    def setUp(self):
        self.scene = SimpleNamespace(crosswalks={"crossing": RectangularCrosswalk()})
        self.states = {}

    def frame(self, t, car, pedestrians=()):
        boxes = [[car[0] - 12, car[1] - 20, car[0] + 12, car[1]]]
        for x, y in pedestrians:
            boxes.append([x - 4, y - 16, x + 4, y])
        detections = Detections(
            xyxy=np.asarray(boxes, dtype=float),
            class_id=np.asarray([CAR] + [PERSON] * len(pedestrians)),
            tracker_id=np.asarray([1] + list(range(2, 2 + len(pedestrians)))),
        )
        return bool(evaluate_failure_to_yield_spatial(
            detections, self.scene, self.states, t,
        )[0])

    def test_pedestrian_ahead_in_vehicle_path_triggers(self):
        self.assertFalse(self.frame(0, (40, 50), [(80, 50)]))
        self.assertTrue(self.frame(0.25, (60, 50), [(80, 50)]))
        self.assertTrue(self.frame(0.5, (90, 50), [(80, 50)]))

    def test_pedestrian_already_past_car_does_not_trigger(self):
        self.frame(0, (40, 50), [(20, 50)])
        self.assertFalse(self.frame(0.25, (60, 50), [(20, 50)]))

    def test_pedestrian_beyond_vehicle_corridor_does_not_trigger(self):
        self.frame(0, (40, 50), [(80, 80)])
        self.assertFalse(self.frame(0.25, (60, 50), [(80, 80)]))

    def test_turn_uses_recent_direction(self):
        self.frame(0, (50, 25))
        self.frame(0.25, (65, 25))
        self.assertTrue(self.frame(0.5, (65, 45), [(65, 70)]))

    def test_only_conflicting_traversal_produces_event(self):
        self.frame(0, (40, 50), [(80, 80)])
        self.frame(0.25, (60, 50), [(80, 80)])
        self.assertEqual(_close_finished_crossings(self.states, set(), 0.5), [])
        self.frame(1, (40, 50), [(80, 50)])
        self.assertTrue(self.frame(1.25, (60, 50), [(80, 50)]))
        self.assertEqual(_close_finished_crossings(self.states, set(), 1.5),
                         [[1.0, 1.5, "failure_to_yield"]])


if __name__ == "__main__":
    unittest.main()
