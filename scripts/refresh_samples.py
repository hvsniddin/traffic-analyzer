"""Refresh website sample labels from validated ground truth without rebuilding EDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pred", default=str(root / "predictions_samples.json"),
                    help="predictions file whose events the sample pages show")
parser.add_argument("--note", default=None, help="caption shown under the predictions")
args = parser.parse_args()
ground_truth = json.loads((root / "annotations" / "ground_truth.json").read_text(encoding="utf-8"))
pred_path = Path(args.pred)
predictions = json.loads(pred_path.read_text(encoding="utf-8"))["videos"] if pred_path.exists() else {}
# Tolerate .mp4/.MP4 differences between prediction and label files.
predictions = {Path(name).stem: entry for name, entry in predictions.items()}
samples = root / "web" / "public" / "samples"
for video, entry in ground_truth.items():
    path = samples / f"{Path(video).stem}.json"
    sample = json.loads(path.read_text(encoding="utf-8"))
    sample["labels"] = {
        "source": "annotations/events.csv",
        "note": "Team-reviewed event intervals; overlapping same-class intervals merged for evaluation.",
        "events": entry["events"],
    }
    if sample.get("eda") and sample["eda"].get("detector"):
        sample["eda"]["detector"]["weights"] = "best.pt"
    if Path(video).stem in predictions:
        result = predictions[Path(video).stem]
        risk = result.get("risk") or []
        sample["predictions"] = {
            "source": pred_path.name,
            "note": args.note,
            "events": result["events"],
            "risk": risk[::max(1, len(risk) // 1500)] if any(score > 0 for _, score in risk) else None,
        }
    path.write_text(json.dumps(sample, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{video}: {len(entry['events'])} labels")
