"""Refresh website sample labels from validated ground truth without rebuilding EDA."""
from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
ground_truth = json.loads((root / "annotations" / "ground_truth.json").read_text(encoding="utf-8"))
pred_path = root / "predictions_samples.json"
predictions = json.loads(pred_path.read_text(encoding="utf-8"))["videos"] if pred_path.exists() else {}
samples = root / "web" / "public" / "samples"
for video, entry in ground_truth.items():
    path = samples / f"{Path(video).stem}.json"
    sample = json.loads(path.read_text(encoding="utf-8"))
    sample["labels"] = {
        "source": "annotations/events.csv",
        "note": "Team-reviewed event intervals; overlapping same-class intervals merged for evaluation.",
        "events": entry["events"],
    }
    if video in predictions:
        result = predictions[video]
        risk = result.get("risk") or []
        sample["predictions"] = {
            "source": "predictions_samples.json",
            "note": None,
            "events": result["events"],
            "risk": risk[::max(1, len(risk) // 1500)] if any(score > 0 for _, score in risk) else None,
        }
    path.write_text(json.dumps(sample, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{video}: {len(entry['events'])} labels")
