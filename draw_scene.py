"""Click once on a fixed-camera frame to map road geometry.

Only this editor needs a desktop display. The saved normalized coordinates can
be used by inference at any frame resolution from the same camera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MODES = {
    ord("1"): ("road", 3),
    ord("2"): ("lane", 3),
    ord("3"): ("crosswalk", 3),
    ord("4"): ("stop_line", 2),
    ord("5"): ("signal", 2),
    ord("6"): ("queue_zone", 3),
    ord("7"): ("lane_direction", 2),
    ord("8"): ("island", 3),
    ord("9"): ("waiting_zone", 3),
    ord("a"): ("solid_line", 2),
    ord("b"): ("intersection", 3),
}
COLORS = {
    "road": (80, 180, 80),
    "lane": (255, 160, 30),
    "crosswalk": (40, 220, 220),
    "stop_line": (40, 40, 255),
    "signal": (220, 50, 220),
    "queue_zone": (160, 80, 220),
    "lane_direction": (255, 255, 255),
    "island": (50, 50, 150),
    "waiting_zone": (180, 120, 50),
    "solid_line": (0, 165, 255),
    "intersection": (255, 80, 80),
}


def load_frame(video: Path | None, image: Path | None, time_sec: float) -> np.ndarray:
    if image is not None:
        frame = cv2.imread(str(image))
    else:
        assert video is not None
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame at {time_sec}s from {video}")
    if frame is None:
        raise RuntimeError(f"Could not read image: {image}")
    return frame


class Editor:
    def __init__(self, frame: np.ndarray, display_width: int) -> None:
        self.original_height, self.original_width = frame.shape[:2]
        display_height = round(self.original_height * display_width / self.original_width)
        self.background = cv2.resize(frame, (display_width, display_height))
        self.width, self.height = display_width, display_height
        self.shapes: list[dict] = []
        self.pending: list[tuple[int, int]] = []
        self.mode = "road"
        self.message = "Choose a shape, click points, press Enter."
        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0

    def _clamp_view(self) -> None:
        self.view_x = max(0.0, min(self.view_x, self.width - self.width / self.zoom))
        self.view_y = max(0.0, min(self.view_y, self.height - self.height / self.zoom))

    def set_zoom(self, new_zoom: float) -> None:
        center_x = self.view_x + self.width / (2 * self.zoom)
        center_y = self.view_y + self.height / (2 * self.zoom)
        self.zoom = max(1.0, min(6.0, new_zoom))
        self.view_x = center_x - self.width / (2 * self.zoom)
        self.view_y = center_y - self.height / (2 * self.zoom)
        self._clamp_view()
        self.message = f"Zoom {self.zoom:.1f}x. Pan with I/J/K/L."

    def pan(self, dx: float, dy: float) -> None:
        self.view_x += dx * self.width / self.zoom
        self.view_y += dy * self.height / self.zoom
        self._clamp_view()

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if not (0 <= x < self.width and 0 <= y < self.height):
                return  # The help bar below the frame is not part of the scene.
            needed = 2 if self.mode in {"stop_line", "signal", "lane_direction", "solid_line"} else None
            if needed and len(self.pending) >= needed:
                self.message = "Press Enter to finish this shape first."
            else:
                base_x = min(self.width - 1, round(self.view_x + x / self.zoom))
                base_y = min(self.height - 1, round(self.view_y + y / self.zoom))
                self.pending.append((base_x, base_y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.pending:
            self.pending.pop()

    def commit(self) -> None:
        minimum = 2 if self.mode in {"stop_line", "signal", "lane_direction", "solid_line"} else 3
        if len(self.pending) < minimum:
            self.message = f"{self.mode} needs at least {minimum} points."
            return
        if self.mode == "lane_direction":
            tail = self.pending[0]
            lanes = [s for s in self.shapes if s["type"] == "lane" and
                     cv2.pointPolygonTest(np.asarray(s["points"], dtype=np.float32), tail, False) >= 0]
            if len(lanes) != 1:
                self.message = ("Arrow tail must be inside exactly one lane; "
                                f"found {len(lanes)}. Press Backspace to adjust.")
                return
            lane = lanes[0]
            lane["direction"] = self.pending.copy()
            self.message = f"Set direction for {lane['name']}."
        else:
            if self.mode == "road":
                self.shapes = [s for s in self.shapes if s["type"] != "road"]
            number = 1 + sum(s["type"] == self.mode for s in self.shapes)
            name = "road" if self.mode == "road" else f"{self.mode}_{number}"
            self.shapes.append({"type": self.mode, "name": name, "points": self.pending.copy()})
            self.message = f"Added {name}."
        self.pending.clear()

    def full_canvas(self) -> np.ndarray:
        canvas = self.background.copy()
        for shape in self.shapes:
            points = np.asarray(shape["points"], dtype=np.int32)
            kind = shape["type"]
            if kind == "signal":
                cv2.rectangle(canvas, tuple(points[0]), tuple(points[1]), COLORS[kind], 2)
            else:
                cv2.polylines(canvas, [points], kind not in {"stop_line", "solid_line"}, COLORS[kind], 2)
            x, y = map(int, points[0])
            cv2.putText(canvas, shape["name"], (x + 4, max(18, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[kind], 2)
            if kind == "lane" and shape.get("direction"):
                a, b = shape["direction"]
                cv2.arrowedLine(canvas, a, b, COLORS["lane_direction"], 3, tipLength=0.25)
        if self.pending:
            points = np.asarray(self.pending, dtype=np.int32)
            for point in points:
                cv2.circle(canvas, tuple(point), 4, COLORS[self.mode], -1)
            if len(points) > 1:
                cv2.polylines(canvas, [points], False, COLORS[self.mode], 2)
        return canvas

    def render(self) -> np.ndarray:
        canvas = self.full_canvas()
        if self.zoom > 1.0:
            x0, y0 = round(self.view_x), round(self.view_y)
            x1 = min(self.width, round(self.view_x + self.width / self.zoom))
            y1 = min(self.height, round(self.view_y + self.height / self.zoom))
            canvas = cv2.resize(canvas[y0:y1, x0:x1], (self.width, self.height))
        bar = np.zeros((58, self.width, 3), dtype=np.uint8)
        cv2.putText(bar, "1 road  2 lane  3 crossing  4 stop  5 signal  6 queue  7 arrow  8 island  9 waiting  A solid  B intersection",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(bar, f"{self.mode}: {self.message}  +/- zoom  I/J/K/L pan  0 reset  Enter finish  U undo  S save",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
        return np.vstack([canvas, bar])

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("coordinates") != "normalized_xy_0_to_1":
            raise ValueError(f"Unsupported coordinate format in {path}")
        source_width, source_height = data["reference_size"]
        if abs(source_width / source_height - self.original_width / self.original_height) > 0.001:
            raise ValueError(f"Scene in {path} has a different aspect ratio")

        def pixels(points: list[list[float]]) -> list[tuple[int, int]]:
            return [(min(self.width - 1, round(x * self.width)),
                     min(self.height - 1, round(y * self.height))) for x, y in points]

        self.shapes.clear()
        groups = [("road", [data["road"]] if data.get("road") else [])]
        groups += [(kind, data.get(plural, [])) for kind, plural in (
            ("lane", "lanes"), ("crosswalk", "crosswalks"),
            ("stop_line", "stop_lines"), ("signal", "signals"),
            ("queue_zone", "queue_zones"), ("island", "islands"),
            ("waiting_zone", "waiting_zones"), ("solid_line", "solid_lines"),
            ("intersection", "intersections"))]
        # The first editor version called islands "exclusions". Read those files too.
        groups.append(("island", data.get("exclusions", [])))
        for kind, records in groups:
            for record in records:
                shape = {"type": kind, "name": record["name"], "points": pixels(record["points"])}
                if kind == "lane" and record.get("direction"):
                    shape["direction"] = pixels(record["direction"])
                if kind == "signal":
                    shape["controls_lanes"] = record.get("controls_lanes", [])
                self.shapes.append(shape)
        self.message = f"Loaded {len(self.shapes)} shapes from {path}."

    def save(self, out: Path, source: str, time_sec: float) -> None:
        if self.pending:
            raise ValueError("Finish or discard the unfinished shape before saving.")

        def norm(points: list[tuple[int, int]]) -> list[list[float]]:
            return [[round(x / self.width, 6), round(y / self.height, 6)] for x, y in points]

        data = {
            "source": source,
            "reference_time_sec": time_sec,
            "reference_size": [self.original_width, self.original_height],
            "coordinates": "normalized_xy_0_to_1",
            "road": None,
            "lanes": [],
            "crosswalks": [],
            "stop_lines": [],
            "signals": [],
            "queue_zones": [],
            "islands": [],
            "waiting_zones": [],
                 "solid_lines": [],
                 "intersections": [],
        }
        plurals = {"lane": "lanes", "crosswalk": "crosswalks", "stop_line": "stop_lines",
                   "signal": "signals", "queue_zone": "queue_zones", "island": "islands",
                     "waiting_zone": "waiting_zones", "solid_line": "solid_lines",
                     "intersection": "intersections"}
        for shape in self.shapes:
            record = {"name": shape["name"], "points": norm(shape["points"])}
            if shape["type"] == "lane":
                record["direction"] = norm(shape["direction"]) if shape.get("direction") else None
            if shape["type"] == "signal":
                record["controls_lanes"] = shape.get("controls_lanes", [])
            if shape["type"] == "road":
                data["road"] = record
            else:
                data[plurals[shape["type"]]].append(record)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        preview = out.with_name(out.stem + "_preview.png")
        cv2.imwrite(str(preview), self.full_canvas())
        print(f"Saved {out} and {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--image", type=Path)
    parser.add_argument("--time", type=float, default=20, help="Seconds into the video (default: 20)")
    parser.add_argument("--scene", type=Path,
                        help="Existing scene JSON to load and update")
    parser.add_argument("--out", type=Path, default=Path("config/scene.json"))
    parser.add_argument("--display-width", type=int, default=1280)
    args = parser.parse_args()
    if args.time < 0 or args.display_width < 400:
        parser.error("--time must be nonnegative and --display-width must be at least 400")
    frame = load_frame(args.video, args.image, args.time)
    editor = Editor(frame, args.display_width)
    scene_path = args.scene or (args.out if args.out.exists() else None)
    if scene_path is not None:
        if not scene_path.exists():
            parser.error(f"Existing scene does not exist: {scene_path}")
        editor.load(scene_path)
        print(f"Editing existing scene: {scene_path}")
    else:
        print(f"Creating new scene: {args.out}")
    window = "Fixed camera scene editor"
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, editor.on_mouse)
    except cv2.error as exc:
        raise RuntimeError("OpenCV could not open a window. Run this on a desktop with opencv-python (GUI build).") from exc
    print("Left click: add point. Right click or Backspace: remove last point.")
    print("Keys: 1 road, 2 lane, 3 crosswalk, 4 stop line, 5 signal, 6 queue zone, 7 lane direction, 8 island, 9 waiting zone, a solid line, b intersection.")
    print("For key 7, click the arrow tail inside the intended lane, then click its head.")
    print("Enter: finish shape. U: undo previous shape. +/-: zoom. I/J/K/L: pan. 0: reset view.")
    print("S: save JSON and preview. Esc: exit. Existing --out JSON is loaded automatically.")
    try:
        while True:
            cv2.imshow(window, editor.render())
            key = cv2.waitKey(30) & 0xFF
            if key in MODES:
                if editor.pending:
                    editor.message = "Finish or discard current points first."
                else:
                    editor.mode = MODES[key][0]
                    editor.message = "Click points, then Enter."
            elif key in (10, 13):
                editor.commit()
            elif key in (8, 127):
                if editor.pending:
                    editor.pending.pop()
            elif key == ord("u"):
                if editor.pending:
                    editor.pending.clear()
                elif editor.shapes:
                    removed = editor.shapes.pop()
                    editor.message = f"Removed {removed['name']}."
            elif key == ord("s"):
                try:
                    editor.save(args.out, str(args.video or args.image), args.time if args.video else 0)
                    editor.message = f"Saved {args.out}."
                except ValueError as exc:
                    editor.message = str(exc)
            elif key in (ord("+"), ord("=")):
                editor.set_zoom(editor.zoom * 1.5)
            elif key == ord("-"):
                editor.set_zoom(editor.zoom / 1.5)
            elif key == ord("0"):
                editor.zoom, editor.view_x, editor.view_y = 1.0, 0.0, 0.0
                editor.message = "View reset."
            elif key == ord("i"):
                editor.pan(0, -0.15)
            elif key == ord("j"):
                editor.pan(-0.15, 0)
            elif key == ord("k"):
                editor.pan(0, 0.15)
            elif key == ord("l"):
                editor.pan(0.15, 0)
            elif key == 27:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
