"""Build the static sample data the website reads (public/samples/*.json + images).

Inputs, all optional except the videos:
  --videos       folder with the original sample MP4s (metadata via ffprobe)
  --frames       folder with <id>/%05d.jpg frames at 1 fps (EDA and detector counts)
  --labels       our dev-label CSVs (video,start_sec,end_sec,label[,notes])
  --predictions  predictions_samples.json written by run_submission.py
  --weights      detector weights for object counts (Ultralytics YOLO)

Every section that has no input is written as null, and the site shows it as
pending instead of inventing numbers. The 540p preview.mp4 per sample must
already exist in public/samples/<id>/ (see web/README.md for the ffmpeg command).

Example (from the traffic-analyzer folder):
  python web/scripts/build_sample_data.py --videos ../dataset --frames <frames_dir> \
      --labels ../annotations/events.csv ../annotations/events_codex.csv \
      --weights ../weights/fixed_names.pt --predictions predictions_samples.json
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]
# fixed_names.pt ids -> website series. Traffic lights are reported separately.
DETECTOR_NAMES = {0: "bicycle", 1: "bus", 2: "car", 4: "motorcycle", 5: "person", 7: "truck"}
LIGHT_NAMES = {3: "green", 6: "red"}
SERIES = ["car", "bus", "truck", "motorcycle", "bicycle", "person"]

WEB_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = WEB_DIR / "public" / "samples"


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name,bit_rate:format=duration,size",
         "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    stream, fmt = info["streams"][0], info["format"]
    num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
    return {
        "file": video.name,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(num / den, 3),
        "duration_sec": round(float(fmt["duration"]), 2),
        "n_frames": int(stream.get("nb_frames") or 0),
        "codec": stream["codec_name"],
        "bitrate_mbps": round(int(stream.get("bit_rate") or 0) / 1e6, 1),
        "size_gb": round(int(fmt["size"]) / 1024**3, 2),
    }


def merge_same_class(events: list[list]) -> list[list]:
    """The official format forbids overlapping segments of one class; merge them."""
    merged: list[list] = []
    for label in CLASSES:
        spans = sorted((s, e) for s, e, lab in events if lab == label)
        for start, end in spans:
            if merged and merged[-1][2] == label and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end, label])
    return sorted(merged, key=lambda ev: (ev[0], ev[1]))


def load_labels(paths: list[Path]) -> dict[str, dict]:
    by_video: dict[str, dict] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                vid = Path(row["video"].strip()).stem.upper()
                label = row["label"].strip()
                try:
                    start, end = float(row["start_sec"]), float(row["end_sec"])
                except ValueError:
                    print(f"skip unfinished label row in {path.name}: {row}")
                    continue
                if label not in CLASSES or start >= end:
                    continue
                entry = by_video.setdefault(vid, {"sources": set(), "events": []})
                entry["sources"].add(path.name)
                entry["events"].append([start, end, label])
    return {
        vid: {
            "source": " + ".join(sorted(entry["sources"])),
            "note": "Draft human labels by the team; same-class overlaps merged. Boundaries are approximate.",
            "events": merge_same_class(entry["events"]),
        }
        for vid, entry in by_video.items()
    }


def load_predictions(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for name, video in data["videos"].items():
        risk = video.get("risk") or []
        has_risk = any(score > 0 for _, score in risk)
        result[Path(name).stem.upper()] = {
            "source": path.name,
            "note": None,
            "events": video["events"],
            # An all-zero curve means Part B is off; show nothing rather than a flat line.
            "risk": [[round(t, 2), round(s, 3)] for t, s in risk[:: max(1, len(risk) // 1500)]] if has_risk else None,
        }
    return result


def save_jpg(path: Path, image: np.ndarray, width: int = 960) -> None:
    h, w = image.shape[:2]
    resized = cv2.resize(image, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), resized, [cv2.IMWRITE_JPEG_QUALITY, 82])


def dim(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor((gray * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def blue_ramp(norm: np.ndarray) -> np.ndarray:
    """Single-hue sequential ramp (light -> dark blue), BGR float image."""
    light, dark = np.array([251, 226, 205.0]), np.array([107, 54, 13.0])  # #cde2fb -> #0d366b
    return light + (dark - light) * norm[..., None]


def motion_eda(preview: Path, out_dir: Path, poster: np.ndarray):
    """Frame differencing and Farneback flow on the 540p preview."""
    cap = cv2.VideoCapture(str(preview))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    size = (480, 270)
    heat = np.zeros(size[::-1], np.float32)
    flow_sum = np.zeros((*size[::-1], 2), np.float32)
    flow_n = np.zeros(size[::-1], np.float32)
    per_sec: dict[int, list[float]] = {}
    prev = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % 3 == 0:  # ~10 fps is plenty for motion statistics
            gray = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if prev is not None:
                moving = cv2.absdiff(gray, prev) > 18
                heat += moving
                per_sec.setdefault(int(index / fps), []).append(float(moving.mean()))
                if index % 15 == 0:
                    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag = np.linalg.norm(flow, axis=2)
                    mask = mag > 1.0
                    flow_sum[mask] += flow[mask]
                    flow_n += mask
            prev = gray
        index += 1
    cap.release()
    if not per_sec:
        return [], None, None

    motion = [[float(t), round(float(np.mean(v)), 4)] for t, v in sorted(per_sec.items())]

    # Motion heatmap: share of sampled frames with change at each pixel.
    norm = np.clip(heat / max(np.percentile(heat, 99.5), 1), 0, 1)
    base = cv2.resize(dim(poster), size).astype(np.float32)
    alpha = np.clip(norm * 1.4, 0, 0.9)[..., None]
    overlay = base * (1 - alpha) + blue_ramp(norm) * alpha
    heat_img = cv2.resize(overlay.astype(np.uint8), (960, 540), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out_dir / "motion_heatmap.jpg"), heat_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Dominant motion direction per 15 px cell = lane directions.
    flow_img = cv2.resize(dim(poster), (960, 540))
    cell = 15
    scale = 960 / size[0]
    for y in range(cell // 2, size[1], cell):
        for x in range(cell // 2, size[0], cell):
            block = (slice(max(0, y - cell // 2), y + cell // 2), slice(max(0, x - cell // 2), x + cell // 2))
            count = flow_n[block].sum()
            if count < 8:
                continue
            vec = flow_sum[block].sum(axis=(0, 1)) / count
            norm_v = np.linalg.norm(vec)
            if norm_v < 1.0:
                continue
            direction = vec / norm_v
            start = np.array([x, y]) * scale
            end = start + direction * cell * scale * 0.8
            cv2.arrowedLine(flow_img, tuple(start.astype(int)), tuple(end.astype(int)),
                            (122, 175, 27), 2, cv2.LINE_AA, tipLength=0.35)  # #1baf7a
    cv2.imwrite(str(out_dir / "flow.jpg"), flow_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return motion, "motion_heatmap.jpg", "flow.jpg"


def detector_eda(frames: list[Path], step: int, model, imgsz: int, conf: float, cache: Path):
    if cache.exists():
        return json.loads(cache.read_text())
    rows = []
    for index, frame_path in enumerate(frames):
        if index % step:
            continue
        result = model.predict(str(frame_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        boxes = result.boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        h, w = result.orig_shape
        rows.append({
            "t": float(index),  # frames are extracted at 1 fps starting at t = 0
            "dets": [[int(c), round(float((x1 + x2) / 2 / w), 4), round(float(y2 / h), 4)]
                     for c, (x1, _, x2, y2) in zip(classes, xyxy)],
        })
    cache.write_text(json.dumps(rows))
    return rows


def positions_image(rows: list[dict], poster: np.ndarray, out: Path) -> None:
    img = cv2.resize(dim(poster), (960, 540))
    colors = {"person": (214, 120, 42), "vehicle": (52, 104, 235)}  # #2a78d6 people, #eb6834 vehicles
    for row in rows:
        for cls, x, y in row["dets"]:
            name = DETECTOR_NAMES.get(cls)
            if not name:
                continue
            color = colors["person"] if name in ("person", "bicycle") else colors["vehicle"]
            cv2.circle(img, (int(x * 960), int(y * 540)), 2, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def build(sample_id: str, args, labels: dict, predictions: dict, model) -> dict:
    out_dir = OUT_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = next(p for p in args.videos.iterdir() if p.stem.upper() == sample_id)
    meta = probe(video_path)

    frames = sorted((args.frames / sample_id).glob("*.jpg")) if args.frames else []
    preview = out_dir / "preview.mp4"
    poster = cv2.imread(str(frames[len(frames) // 2])) if frames else None
    if poster is None and preview.exists():
        cap = cv2.VideoCapture(str(preview))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
        ok, poster = cap.read()
        cap.release()
        poster = poster if ok else None
    if poster is not None:
        save_jpg(out_dir / "poster.jpg", poster)

    rel = lambda name: f"samples/{sample_id}/{name}" if name else None  # noqa: E731
    eda = None
    if frames and poster is not None:
        brightness = [[float(i), round(float(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).mean()), 1)]
                      for i, p in enumerate(frames) if i % args.step == 0]
        motion, heat_name, flow_name = motion_eda(preview, out_dir, poster) if preview.exists() else ([], None, None)
        counts, signal, positions_name = [], [], None
        if model is not None:
            rows = detector_eda(frames, args.step, model, args.imgsz, args.conf, args.cache / f"{sample_id}_dets.json")
            for row in rows:
                tally = {name: 0 for name in SERIES}
                lights = {"red": 0, "green": 0}
                for cls, _, _ in row["dets"]:
                    if cls in DETECTOR_NAMES:
                        tally[DETECTOR_NAMES[cls]] += 1
                    elif cls in LIGHT_NAMES:
                        lights[LIGHT_NAMES[cls]] += 1
                counts.append({"t": row["t"], **tally})
                signal.append({"t": row["t"], **lights})
            positions_image(rows, poster, out_dir / "positions.jpg")
            positions_name = "positions.jpg"
        eda = {
            "step_sec": args.step,
            "detector": {"weights": args.weights.name if args.weights else "", "imgsz": args.imgsz,
                         "conf": args.conf, "input": "1080p frames at 1 fps"},
            "brightness": brightness,
            "counts": counts,
            "signal": signal,
            "motion": motion,
            "motion_heatmap_url": rel(heat_name),
            "positions_url": rel(positions_name),
            "flow_url": rel(flow_name),
        }

    return {
        "id": sample_id,
        "video": meta,
        "poster_url": rel("poster.jpg") if poster is not None else None,
        "preview_url": rel("preview.mp4") if preview.exists() else None,
        "annotated_url": rel("annotated.mp4") if (out_dir / "annotated.mp4").exists() else None,
        "predictions": predictions.get(sample_id),
        "labels": labels.get(sample_id),
        "eda": eda,
        "failures": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--labels", type=Path, nargs="*", default=[])
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--ids", nargs="*", help="sample ids, default: every MP4 in --videos")
    parser.add_argument("--step", type=int, default=2, help="seconds between EDA samples")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--cache", type=Path, default=WEB_DIR / ".cache")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    ids = args.ids or sorted(p.stem.upper() for p in args.videos.glob("*") if p.suffix.lower() == ".mp4")
    labels = load_labels(args.labels)
    predictions = load_predictions(args.predictions)
    model = None
    if args.weights:
        from ultralytics import YOLO
        model = YOLO(str(args.weights))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sample_id in ids:
        sample = build(sample_id, args, labels, predictions, model)
        (OUT_DIR / f"{sample_id}.json").write_text(json.dumps(sample, separators=(",", ":")), encoding="utf-8")
        print(f"{sample_id}: labels={bool(sample['labels'])} predictions={bool(sample['predictions'])} eda={bool(sample['eda'])}")

    existing = sorted(p.stem for p in OUT_DIR.glob("*.json") if p.stem != "index")
    (OUT_DIR / "index.json").write_text(json.dumps({"samples": existing}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
