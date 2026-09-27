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
MAX_FAILURES = 8


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _name(label):
    return label.replace("_", " ")


def failure_cases(labels, predicted):
    """Plain-language mismatches between predictions and dev labels, longest first."""
    cases = []
    for start, end, label in labels:
        same = [(_iou((start, end), (ps, pe)), ps, pe) for ps, pe, pl in predicted if pl == label]
        best = max(same, default=(0.0, 0, 0))
        if best[0] >= 0.5:
            continue
        others = sorted({pl for ps, pe, pl in predicted if pl != label and _iou((start, end), (ps, pe)) > 0})
        if best[0] > 0:
            note = (f"Boundaries off: labelled {_name(label)} {start:g}-{end:g} s, predicted "
                    f"{best[1]:.1f}-{best[2]:.1f} s (IoU {best[0]:.2f}, below the 0.5 needed to match).")
        else:
            note = f"Missed: labelled {_name(label)} ({end - start:g} s) has no {_name(label)} prediction."
            if others:
                note += f" Other classes predicted in this span: {', '.join(map(_name, others))}."
        cases.append((end - start, start, end, note))
    for ps, pe, pl in predicted:
        if any(_iou((ps, pe), (s0, e0)) > 0 for s0, e0, l0 in labels if l0 == pl):
            continue
        others = sorted({l0 for s0, e0, l0 in labels if l0 != pl and _iou((ps, pe), (s0, e0)) > 0})
        note = f"False alarm: predicted {_name(pl)} ({pe - ps:.1f} s) where nothing of that class is labelled."
        if others:
            note += f" It overlaps a labelled {', '.join(map(_name, others))}, a likely confusion between the two."
        cases.append((pe - ps, ps, pe, note))
    seen, out = set(), []
    for _, start, end, note in sorted(cases, key=lambda case: -case[0]):
        key = (round(start, 2), round(end, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append({"start_sec": round(start, 2), "end_sec": round(end, 2), "note": note})
        if len(out) == MAX_FAILURES:
            break
    return sorted(out, key=lambda case: case["start_sec"])

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
        sample["failures"] = failure_cases(entry["events"], result["events"])
    path.write_text(json.dumps(sample, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{video}: {len(entry['events'])} labels")
