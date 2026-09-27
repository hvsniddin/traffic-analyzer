import unittest
from types import SimpleNamespace

import numpy as np

from detections.wrong_way import evaluate_wrong_way_spatial


class Detections(SimpleNamespace):
    def __len__(self):
        return len(self.xyxy)


def rectangle(x1, y1, x2, y2):
    return SimpleNamespace(polygon=np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
    ], dtype=np.float32))


class WrongWayTests(unittest.TestCase):
    def setUp(self):
        self.scene = SimpleNamespace(
            lanes={"lane_1": rectangle(0, 0, 1000, 100)},
            lane_directions={"lane_1": np.array([1.0, 0.0])},
        )

    def run_track(self, velocity):
        history, masks = {}, []
        for step in range(0, 16):
            t = step * 0.2
            x = 500 + velocity * t
            detections = Detections(
                xyxy=np.array([[x - 10, 40, x + 10, 50]], dtype=float),
                class_id=np.array([2]),
                tracker_id=np.array([5]),
            )
            masks.append(bool(evaluate_wrong_way_spatial(detections, self.scene, history, t)[0]))
        return masks, history

    def test_opposing_motion_is_flagged_with_early_start(self):
        masks, history = self.run_track(-60)
        self.assertTrue(masks[-1])
        self.assertFalse(masks[3])
        self.assertLess(history["starts"][("wrong_way", 5)], 0.5)

    def test_correct_direction_is_not_flagged(self):
        masks, _ = self.run_track(60)
        self.assertFalse(any(masks))

    def test_jitter_is_not_flagged(self):
        masks, _ = self.run_track(-2)
        self.assertFalse(any(masks))


if __name__ == "__main__":
    unittest.main()
