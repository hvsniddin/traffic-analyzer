import unittest
from types import SimpleNamespace

import numpy as np

from detections.classes import CAR, PERSON
from detections.near_miss import NearMissDetector


def frame(car_x, pedestrian_x=75, pedestrian_y=50):
    return SimpleNamespace(
        xyxy=np.asarray([[car_x - 10, 30, car_x + 10, 50],
                         [pedestrian_x - 4, pedestrian_y - 16,
                          pedestrian_x + 4, pedestrian_y]], dtype=float),
        class_id=np.asarray([CAR, PERSON]),
        tracker_id=np.asarray([1, 2]),
    )


class NearMissTests(unittest.TestCase):
    def test_braking_for_pedestrian_then_clearing(self):
        detector = NearMissDetector()
        events = []
        for index, x in enumerate([0, 20, 40, 55, 58, 61, 95, 120]):
            _, ended = detector.update(frame(x), index * 0.25)
            events.extend(ended)
        self.assertEqual(events, [[1.0, 1.75, "near_miss"]])

    def test_braking_without_conflicting_road_user_is_not_event(self):
        detector = NearMissDetector()
        for index, x in enumerate([0, 20, 40, 55, 58, 61, 95, 120]):
            _, events = detector.update(frame(x, 300), index * 0.25)
            self.assertEqual(events, [])
        self.assertEqual(detector.finish(2.0), [])

    def test_constant_speed_close_pass_is_not_event(self):
        detector = NearMissDetector()
        for index, x in enumerate([0, 20, 40, 60, 80, 100, 120]):
            _, events = detector.update(frame(x), index * 0.25)
            self.assertEqual(events, [])

    def test_image_path_crossing_at_different_depth_is_not_event(self):
        detector = NearMissDetector()
        for index, x in enumerate([0, 20, 40, 55, 58, 61, 95, 120]):
            _, events = detector.update(frame(x, pedestrian_y=90), index * 0.25)
            self.assertEqual(events, [])
        self.assertEqual(detector.finish(2.0), [])


if __name__ == "__main__":
    unittest.main()
