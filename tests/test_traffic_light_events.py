import unittest
from types import SimpleNamespace

import numpy as np

from detections.traffic_light_events import evaluate_traffic_light_events
from detections.classes import CAR, GREENLIGHT, REDLIGHT


class TrafficLightEventTests(unittest.TestCase):
    def setUp(self):
        self.scene = SimpleNamespace(
            stop_lines={"line": SimpleNamespace(vector=SimpleNamespace(
                start=SimpleNamespace(x=0, y=50), end=SimpleNamespace(x=100, y=50)))},
            intersections={"junction": SimpleNamespace(polygon=np.array(
                [[0, 50], [100, 50], [100, 90], [0, 90]]))},
        )
        self.history = {}

    def frame(self, t, top, signal=REDLIGHT, x=50, height=10):
        boxes = np.empty((0, 4), dtype=float)
        classes = []
        ids = []
        if top is not None:
            boxes = np.array([[x - 10, top, x + 10, top + height]], dtype=float)
            classes.append(CAR)
            ids.append(1)
        if signal is not None:
            boxes = np.vstack((boxes, [0, 0, 2, 2]))
            classes.append(signal)
            ids.append(2)
        class Detections:
            def __len__(self):
                return len(self.xyxy)
        result = Detections()
        result.xyxy = boxes
        result.class_id = np.array(classes)
        result.tracker_id = np.array(ids)
        red, stop = evaluate_traffic_light_events(result, self.scene, self.history, t)
        return (bool(red[0]), bool(stop[0])) if top is not None else (False, False)

    def test_full_crossing_starts_at_front_and_ends_after_intersection(self):
        self.assertEqual(self.frame(0, 25), (False, False))
        self.assertEqual(self.frame(1, 45), (False, False))
        self.assertEqual(self.frame(2, 55), (True, False))
        self.assertAlmostEqual(self.history["event_starts"][("red_light", 1)], 0.75)
        self.assertEqual(self.frame(3, 75, GREENLIGHT), (True, False))
        self.assertEqual(self.frame(4, 95, GREENLIGHT), (False, False))

    def test_stopping_on_line_is_only_stop_line_until_green(self):
        self.frame(0, 25)
        self.assertEqual(self.frame(1, 45), (False, False))
        self.assertEqual(self.frame(1.5, 45), (False, True))
        self.assertEqual(self.frame(2, 45), (False, True))
        self.assertEqual(self.history["event_starts"][("stop_line", 1)], 1)
        self.assertEqual(self.frame(3, 55), (False, True))
        self.assertEqual(self.frame(4, 55, GREENLIGHT), (False, False))

    def test_stop_after_full_crossing_cancels_red_light(self):
        self.frame(0, 25)
        self.frame(1, 45)
        self.assertEqual(self.frame(2, 55), (True, False))
        self.assertEqual(self.frame(2.5, 55), (False, True))
        self.assertEqual(self.frame(3, 55), (False, True))

    def test_box_spans_line_but_road_contact_is_past_it(self):
        self.assertEqual(self.frame(0, 45, height=50), (False, False))
        self.assertEqual(self.frame(1, 45, height=50), (False, False))
        self.assertEqual(self.frame(2, 45, height=50), (False, False))

    def test_unseen_signal_counts_as_red_for_a_crossing_on_the_segment(self):
        self.assertEqual(self.frame(0, 25, None), (False, False))
        self.assertEqual(self.frame(1, 45, None), (False, False))
        self.assertEqual(self.frame(2, 55, None), (True, False))

    def test_crossing_beyond_line_end_is_not_a_red_light_violation(self):
        self.assertEqual(self.frame(0, 25, None, x=130), (False, False))
        self.assertEqual(self.frame(1, 45, None, x=130), (False, False))
        self.assertEqual(self.frame(2, 55, None, x=130), (False, False))

    def test_traffic_heading_toward_lane_2_outside_segment_is_excluded(self):
        self.assertEqual(self.frame(0, 65, None, x=130), (False, False))
        self.assertEqual(self.frame(1, 45, None, x=120), (False, False))
        self.assertEqual(self.frame(2, 35, None, x=110), (False, False))

    def test_recent_green_exempts_crossing_but_expires(self):
        self.frame(0, 25, GREENLIGHT)
        self.frame(1, 45, None)
        self.assertEqual(self.frame(2, 55, None), (False, False))
        for frame in range(3, 28):
            self.frame(frame, None, None)
        self.frame(28, 25, None)
        self.frame(29, 45, None)
        self.assertEqual(self.frame(30, 55, None), (True, False))



if __name__ == "__main__":
    unittest.main()
