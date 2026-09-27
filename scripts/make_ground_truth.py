"""Convert hand-labelled event CSV to the organizer's ground-truth JSON format."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2


CLASSES = {
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    gt = {}
    video_ids = {}
    for path in sorted(args.videos.iterdir()):
        if path.suffix.lower() != ".mp4":
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        gt[path.name] = {"duration": n_frames / fps, "fps": fps, "events": []}
        video_ids[path.stem.casefold()] = path.name

    with args.csv.open(newline="", encoding="utf-8-sig") as file:
        for row_num, row in enumerate(csv.DictReader(file), start=2):
            if not any(row.values()):
                continue
            raw_video = row["video"].strip()
            video = video_ids.get(Path(raw_video).stem.casefold(), raw_video)
            label = row["label"].strip()
            if video not in gt:
                raise ValueError(f"Row {row_num}: video {video!r} is not in {args.videos}")
            if label not in CLASSES:
                raise ValueError(f"Row {row_num}: unknown label {label!r}")
            start, end = float(row["start_sec"]), float(row["end_sec"])
            if not 0 <= start < end <= gt[video]["duration"] + 0.05:
                raise ValueError(f"Row {row_num}: invalid interval {start}, {end}")
            gt[video]["events"].append([start, min(end, gt[video]["duration"]), label])

    for video, entry in gt.items():
        entry["events"].sort(key=lambda event: (event[0], event[1], event[2]))
        by_class = defaultdict(list)
        for start, end, label in entry["events"]:
            by_class[label].append((start, end))
        merged_events = []
        for label, segments in by_class.items():
            segments.sort()
            for start, end in segments:
                if merged_events and merged_events[-1][2] == label and start <= merged_events[-1][1]:
                    merged_events[-1][1] = max(merged_events[-1][1], end)
                else:
                    merged_events.append([start, end, label])
        entry["events"] = sorted(merged_events, key=lambda event: (event[0], event[1], event[2]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} for {len(gt)} videos")


if __name__ == "__main__":
    main()
