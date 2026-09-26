import json
import numpy as np
import supervision as sv

class SceneGeometry:
    def __init__(self, json_path: str, frame_width: int = 3840, frame_height: int = 2160):
        self.frame_width = frame_width
        self.frame_height = frame_height

        with open(json_path, "r") as f:
            self.raw_config = json.load(f)

        self.road_zone = None
        self.lanes = {}           # name -> sv.PolygonZone
        self.lane_directions = {} # name -> np.ndarray ([dx, dy] normalized direction vector)
        self.crosswalks = {}     # name -> sv.PolygonZone
        self.stop_lines = {}     # name -> sv.LineZone
        self.islands = {}        # name -> sv.PolygonZone
        self.waiting_zones = {}   # name -> sv.PolygonZone

        self._build_geometry()

    def _denormalize(self, points: list[list[float]]) -> np.ndarray:
        """Converts [x_norm, y_norm] to [x_pixel, y_pixel] integer array."""
        pts = np.array(points, dtype=np.float32)
        pts[:, 0] *= self.frame_width
        pts[:, 1] *= self.frame_height
        return pts.astype(np.int32)

    def _build_geometry(self):
        # 1. Road Boundary
        if "road" in self.raw_config and "points" in self.raw_config["road"]:
            road_pts = self._denormalize(self.raw_config["road"]["points"])
            self.road_zone = sv.PolygonZone(polygon=road_pts)

        # 2. Lanes & Flow Directions
        for lane in self.raw_config.get("lanes", []):
            name = lane["name"]
            lane_pts = self._denormalize(lane["points"])
            self.lanes[name] = sv.PolygonZone(polygon=lane_pts)

            # Extract allowed travel direction vector [dx, dy]
            if "direction" in lane and len(lane["direction"]) == 2:
                dir_pts = self._denormalize(lane["direction"])
                v = dir_pts[1] - dir_pts[0]  # vector from start to end
                norm = np.linalg.norm(v)
                if norm > 0:
                    self.lane_directions[name] = v / norm  # Unit vector

        # 3. Crosswalks
        for cw in self.raw_config.get("crosswalks", []):
            cw_pts = self._denormalize(cw["points"])
            self.crosswalks[cw["name"]] = sv.PolygonZone(polygon=cw_pts)

        # 4. Stop Lines
        for sl in self.raw_config.get("stop_lines", []):
            sl_pts = self._denormalize(sl["points"])
            p1 = sv.Point(x=sl_pts[0][0], y=sl_pts[0][1])
            p2 = sv.Point(x=sl_pts[1][0], y=sl_pts[1][1])
            self.stop_lines[sl["name"]] = sv.LineZone(start=p1, end=p2)

        # 5. Pedestrian Islands
        for isl in self.raw_config.get("islands", []):
            isl_pts = self._denormalize(isl["points"])
            self.islands[isl["name"]] = sv.PolygonZone(polygon=isl_pts)

        # 6. Waiting Zones
        for wz in self.raw_config.get("waiting_zones", []):
            wz_pts = self._denormalize(wz["points"])
            self.waiting_zones[wz["name"]] = sv.PolygonZone(polygon=wz_pts)
