import unittest
from types import SimpleNamespace

import numpy as np

from detections.solid_line import evaluate_solid_line_crossing_spatial


class Detections(SimpleNamespace):
    def __len__(self):
        return len(self.xyxy)


class SolidLineTests(unittest.TestCase):
    def setUp(self):
        self.scene = SimpleNamespace(
            solid_lines={"marking": SimpleNamespace(start=(50, 0), end=(50, 100))},
            lanes={},
        )
        self.state = {}

    def detect(self, center_x, bottom_y=50):
        detections = Detections(
            xyxy=np.array([[center_x - 10, bottom_y - 20,
                            center_x + 10, bottom_y]], dtype=float),
            class_id=np.array([2]),
            tracker_id=np.array([1]),
        )
        return bool(evaluate_solid_line_crossing_spatial(
            detections, self.scene, self.state
        )[0])

    def test_box_corner_touch_does_not_start_event(self):
        self.assertFalse(self.detect(30))
        self.assertFalse(self.detect(42))  # Right corner crossed x=50.
        self.assertFalse(self.detect(47))

    def test_bottom_midpoint_crossing_starts_event(self):
        self.assertFalse(self.detect(42))
        self.assertFalse(self.detect(50))  # Too close to the line to confirm.
        self.assertTrue(self.detect(58))

    def test_near_line_jitter_does_not_start_event(self):
        self.assertFalse(self.detect(49.5))
        self.assertFalse(self.detect(50.5))
        self.assertFalse(self.detect(49.5))

    def test_crossing_line_extension_does_not_start_event(self):
        self.assertFalse(self.detect(40, bottom_y=150))
        self.assertFalse(self.detect(60, bottom_y=150))


if __name__ == "__main__":
    unittest.main()
