import unittest
from types import SimpleNamespace

import numpy as np

from detections.risk import CollisionRisk


def frame(boxes):
    return SimpleNamespace(
        xyxy=np.array([b[:4] for b in boxes], dtype=float),
        class_id=np.array([2] * len(boxes)),
        tracker_id=np.array([b[4] for b in boxes]),
    )


def car(x, y, track_id):
    return [x - 20, y - 20, x + 20, y, track_id]


class CollisionRiskTests(unittest.TestCase):
    def run_pair(self, first, second, steps=12, dt=0.2):
        risk, scores = CollisionRisk(), []
        for step in range(steps):
            t = step * dt
            scores.append(risk.update(frame([car(*first(t), 1), car(*second(t), 2)]), t))
        return scores

    def test_crossing_paths_raise_risk(self):
        # Both head for (400, 300) at 200 px/s and meet at t = 2.5 s.
        scores = self.run_pair(lambda t: (400, -200 + 200 * t), lambda t: (-100 + 200 * t, 300))
        self.assertGreater(max(scores), 0.2)

    def test_parallel_lanes_stay_quiet(self):
        scores = self.run_pair(lambda t: (100 + 200 * t, 300), lambda t: (100 + 120 * t, 305))
        self.assertEqual(max(scores), 0.0)

    def test_queued_neighbours_stay_quiet(self):
        scores = self.run_pair(lambda t: (100, 300), lambda t: (160 + 5 * t, 300))
        self.assertEqual(max(scores), 0.0)


if __name__ == "__main__":
    unittest.main()
