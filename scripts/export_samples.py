"""Run the pipeline once over the sample videos and export everything the site needs.

Writes predictions_samples.json (same events and cleaning as run_submission.py,
which is deterministic, so a harness run reproduces it) and, per video,
web/public/samples/<id>/overlay.json with boxes, event flags, scene geometry,
and the causal risk curve from the same tracks.

The harness's wall-clock guard is disabled here: on a slow CPU it would cut
the run short, while the organizers' GPU run finishes every video.

    python scripts/export_samples.py --videos /path/to/sample-videos
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from detections.overlay import OverlayRecorder  # noqa: E402
from run_submission import clean_events, video_meta  # noqa: E402
from solution import CLASSES, SUPPRESSED_CLASSES, _detect_events  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--out", default=str(root / "predictions_samples.json"))
    parser.add_argument("--site", default=str(root / "web" / "public" / "samples"))
    parser.add_argument("--team", default="Lorem Ipsum")
    args = parser.parse_args()

    videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() == ".mp4")
    result = {"team": args.team, "videos": {}, "log": {}}
    for path in videos:
        meta = video_meta(path)
        recorder = OverlayRecorder(meta["fps"], meta["width"], meta["height"],
                                   hidden_labels=SUPPRESSED_CLASSES)
        started = time.perf_counter()
        events, problems = clean_events(
            _detect_events(str(path), on_sample=recorder, time_budget_factor=float("inf")), CLASSES, meta["duration"])
        seconds = round(time.perf_counter() - started, 1)
        result["videos"][path.name] = {"events": events, "risk": []}
        result["log"][path.name] = {
            "duration": round(meta["duration"], 2), "part_a_sec": seconds, "errors": problems,
            "note": "Part B needs a GPU run (RiskEstimator.reset skips it on CPU).",
        }
        target = Path(args.site) / path.stem
        target.mkdir(parents=True, exist_ok=True)
        (target / "overlay.json").write_text(
            json.dumps({**recorder.overlay(), "risk": recorder.risk}, separators=(",", ":")),
            encoding="utf-8")
        print(f"[{path.name}] {len(events)} events in {seconds}s ({seconds / meta['duration']:.2f}x)", flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
