import unittest
from types import SimpleNamespace

import numpy as np
import supervision as sv

from detections.jaywalking import evaluate_jaywalking_spatial


class JaywalkingClearanceTests(unittest.TestCase):
    def test_crosswalk_clearance_uses_box_bottom_center(self):
        scene = SimpleNamespace(
            road_zone=sv.PolygonZone(polygon=np.array([[0, 0], [199, 0], [199, 199], [0, 199]])),
            crosswalks={
                "crosswalk": sv.PolygonZone(
                    polygon=np.array([[80, 30], [120, 30], [120, 170], [80, 170]])
                )
            },
            islands={},
            waiting_zones={},
        )
        # All boxes are 68 px high, giving an estimated 20 px clearance.
        boxes = np.array([
            [40, 32, 60, 100],   # foot x=50, 30 px outside
            [52, 32, 72, 100],   # foot x=62, 18 px outside
            [75, 32, 95, 100],   # foot x=85, inside
            [110, 32, 130, 100], # foot x=120, boundary
            [128, 32, 148, 100], # foot x=138, 18 px outside
            [150, 32, 170, 100], # foot x=160, 40 px outside
        ], dtype=float)
        detections = sv.Detections(
            xyxy=boxes,
            confidence=np.ones(len(boxes)),
            class_id=np.zeros(len(boxes), dtype=int),
        )

        jaywalking, safe = evaluate_jaywalking_spatial(detections, scene)

        np.testing.assert_array_equal(jaywalking, [True, False, False, False, False, True])
        np.testing.assert_array_equal(safe, ~jaywalking)


if __name__ == "__main__":
    unittest.main()
