import unittest
from types import SimpleNamespace

import numpy as np

from detections.stopped_vehicle import evaluate_stopped_vehicle_spatial


class Detections(SimpleNamespace):
    def __len__(self):
        return len(self.xyxy)


def rectangle(x1, y1, x2, y2):
    return SimpleNamespace(polygon=np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
    ], dtype=np.float32))


class StoppedVehicleTests(unittest.TestCase):
    def setUp(self):
        self.scene = SimpleNamespace(
            lanes={"lane_1": rectangle(0, 0, 100, 100), "lane_2": rectangle(200, 0, 300, 100)},
            intersections={},
        )

    def run_track(self, x, times, red=lambda t: False):
        history, masks = {}, []
        for t in times:
            detections = Detections(
                xyxy=np.array([[x - 10, 40, x + 10, 50]], dtype=float),
                class_id=np.array([2]),
                tracker_id=np.array([3]),
            )
            masks.append(bool(evaluate_stopped_vehicle_spatial(
                detections, self.scene, history, t, signal_is_red=red(t))[0]))
        return masks, history

    def test_free_lane_stop_reported_after_ten_seconds(self):
        masks, history = self.run_track(250, range(0, 13))
        self.assertEqual(masks, [False] * 10 + [True] * 3)
        self.assertEqual(history["starts"][("stopped_vehicle", 3)], 0)

    def test_signal_queue_is_ignored(self):
        masks, _ = self.run_track(50, range(0, 15), red=lambda t: t < 5)
        self.assertFalse(any(masks))

    def test_green_stop_in_approach_lane_counts(self):
        masks, _ = self.run_track(50, range(0, 12))
        self.assertTrue(masks[-1])

    def test_moving_vehicle_is_not_stopped(self):
        history, flagged = {}, False
        for t in range(0, 15):
            detections = Detections(
                xyxy=np.array([[200 + 5 * t, 40, 220 + 5 * t, 50]], dtype=float),
                class_id=np.array([2]),
                tracker_id=np.array([3]),
            )
            flagged |= bool(evaluate_stopped_vehicle_spatial(detections, self.scene, history, t)[0])
        self.assertFalse(flagged)

    def test_vehicle_waiting_behind_another_stopped_one_is_queued(self):
        history, flagged = {}, False
        for t in range(0, 15):
            detections = Detections(
                xyxy=np.array([[240, 40, 260, 50], [265, 40, 285, 50]], dtype=float),
                class_id=np.array([2, 1]),
                tracker_id=np.array([3, 4]),
            )
            flagged |= bool(evaluate_stopped_vehicle_spatial(detections, self.scene, history, t).any())
        self.assertFalse(flagged)


if __name__ == "__main__":
    unittest.main()
