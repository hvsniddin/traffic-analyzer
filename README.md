# Lorem Ipsum — WIUT traffic event analysis

This repository is the offline inference submission and the source for our team website. It accepts fixed road-camera MP4 files and returns event intervals in the organizer's 14-class format.

## Install and run

Use Python 3.10–3.12. The repository includes `weights/best.pt` (about 40 MB); no download step is needed. Install with `requirements.txt`; on Linux, pip's default PyTorch wheel includes CUDA, and inference uses the first GPU when torch sees one and falls back to CPU otherwise (each video logs its device). The repository deliberately has no root `Dockerfile`: `deploy/Dockerfile.api` is the CPU-only image for the website's demo API, not the evaluation environment. From this directory:

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json --team "Lorem Ipsum"
python evaluate.py --pred predictions.json --validate-only
```

`run_submission.py` and `evaluate.py` are unchanged copies from the starter kit. `solution.py` is the required interface. The model and scene data are bundled, so inference does not need internet access. On CUDA machines, the detector uses the first GPU; otherwise it uses CPU. Inference samples 2 FPS on every device (`DEFAULT_INFERENCE_FPS`), so CPU and GPU runs produce the same events and `predictions_samples.json` can be reproduced on either. `WIUT_INFERENCE_FPS` overrides the rate.

**Time budget.** The harness scores a video as empty past 3× its duration, so `detect_events` stops at 2.7× (`WIUT_TIME_BUDGET_FACTOR`), and always at least 3 s before 3×, then returns the events found so far. Part A records its own decode speed and per-frame cost. `RiskEstimator.reset` uses these timings, never Part A's events, to check whether the harness's full-video decode for Part B still fits. If it would not, or there is no GPU, `reset` raises; the harness logs this and keeps Part A's events. `WIUT_RISK=0` always skips Part B and `WIUT_RISK=1` always runs it (local testing only).

## Approach

One YOLO11 medium model fine-tuned for bicycle, bus, car, green light, motorcycle, person, red light, and truck detects objects. Ultralytics ByteTrack associates detections over time. We run this detector and tracker once per sampled frame and share the resulting tracks among rule modules. `scene.json` contains hand-drawn road, lane, crosswalk, stop-line, island, and waiting-area geometry. `scene_reference.jpg` is the associated reference frame. A SIFT/RANSAC homography aligns that geometry to each video, with quality gates and an identity fallback.

The event rules use position, line crossing, tracker history, and detected traffic-light state. The integrated pipeline emits `jaywalking`, `failure_to_yield`, `red_light`, `stop_line`, `solid_line_crossing`, `illegal_turn`, `near_miss`, `stopped_vehicle`, and `wrong_way`:

- `stopped_vehicle`: the contact point of a vehicle track stays within a quarter of its box width for 10 s or more, inside a lane or the intersection. Stops in the signal approach lanes do not count if the signal was red or traffic was queued at any time during the stop.
- `wrong_way`: over a ~1 s window, a vehicle's displacement points more than 120° away from the drawn direction of its lane (`scene.json`) for at least 1.5 s.
- `near_miss`: an evasive manoeuvre (hard braking or swerving) by a vehicle while another road user is on a projected collision course, and no sustained box overlap. Its velocity estimate needs samples less than 1 s apart, so it only fires at the GPU sample rate.

`congestion` is still computed, because `stopped_vehicle` uses it to recognise queues, but it is not reported. On our dev labels it fired on ordinary red-light queues (37 false positives against 1 true event). Same-class intervals are merged across gaps shorter than 1.5 s, or 8 s for `jaywalking`, because pedestrians drop out of tracking briefly. `accident`, `illegal_u_turn`, `road_obstacle`, and `fire_smoke` have no detector. We had no examples to tune them on, and a false positive on a class missing from the test set lowers the macro score.

**Part B** (`detections/risk.py`) runs the same detector and a separate tracker at 5 FPS inside `RiskEstimator.step`, using only the frames received so far. For every pair of road users that includes a vehicle, it projects both at constant velocity. A pair adds risk when it closes faster than 2 box widths per second on non-parallel headings, and its closest approach is within 0.3 of the larger box width and less than 5 s away. The risk grows as that time and distance shrink, and a hard brake or swerve adds to it. The frame score is the worst pair, averaged over the last second. With no accidents in the sample videos, thresholds were set to keep false alarms rare on normal traffic, not calibrated against real crashes.

The detector is learned; tracking, scene alignment, event rules, and post-processing are algorithmic. The provided sample videos were manually labeled in `annotations/events.csv`; `annotations/ground_truth.json` is the merged official-format dev set. These labels are for evaluation and rule tuning, not a replacement for test data.

## Evaluate on sample videos

```bash
python scripts/make_ground_truth.py --videos /path/to/sample-videos --csv annotations/events.csv --out annotations/ground_truth.json
python run_submission.py --videos /path/to/sample-videos --out predictions_samples.json --team "Lorem Ipsum"
python evaluate.py --pred predictions_samples.json --gt annotations/ground_truth.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only
```

The four provided MP4s are deliberately excluded from Git because they total tens of GB. `predictions_samples.json` should contain the actual output of the pipeline, not manual labels.

`scripts/export_samples.py --videos /path/to/sample-videos` produces `predictions_samples.json` and the website's per-video overlays (`web/public/samples/<id>/overlay.json`: boxes, event flags, scene geometry, and the risk curve) in a single pass. It calls the same deterministic `_detect_events` and the harness's own `clean_events`, so its events match a `run_submission.py` run. After it, `python scripts/refresh_samples.py` copies the predictions into the sample pages.

## Live demo and website

The browser UI is in `web/`. The inference API is in `api/main.py`; it runs the same detection code as the submission, and additionally returns per-frame boxes, event flags, aligned scene geometry, and a risk curve. The player draws these over the browser's local copy of the upload, so no annotated video has to be encoded or downloaded.

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
cd web
npm ci
# Set NEXT_PUBLIC_API_URL to the public HTTPS URL of the API before building.
npm run build
```

The API supports `POST /api/jobs` (multipart MP4) and `GET /api/jobs/{job_id}`. Default limits are 200 MB and 120 seconds. Set `WIUT_CORS_ORIGINS` to the website origin in deployment; it defaults to `*` for local testing. `NEXT_PUBLIC_API_URL` is baked into the static site at build time. An empty value intentionally switches the site to a labeled mock demo, so the published build must have the real API URL.

## Reproducibility and provenance

Event rules and scene alignment are deterministic for a given detection stream. Inference does not use random augmentation. Ultralytics/PyTorch and ByteTrack may vary slightly across CPU/GPU, library versions, and floating-point kernels. Python package versions are pinned in `requirements.txt`; the model file is stored directly in `weights/`. Seeds for `random`, NumPy, and PyTorch are fixed to 0 (`solution.SEED`), and cuDNN runs in deterministic mode. The time-budget guards are the one wall-clock dependency: on a much slower machine, Part A can stop early or Part B can be skipped.

**Training data and licenses:** `best.pt` was fine-tuned only on team-annotated frames extracted from the four videos supplied for this hackathon. No public dataset was added for fine-tuning. The videos are organizer-provided competition data; no separate public redistribution license was supplied, so the videos and extracted training frames are not included in Git. The base YOLO11 model is distributed by Ultralytics under its licensing terms; see their license for reuse outside this competition. No training runs at inference time.

## Team

- Ja'farbek Yusupov — event annotations, scene layout, testing, event rules, integration, backend, and deployment.
- Husniddin Ravshanov — detector testing and fine-tuning, event rules, and solution implementation.
- Azizbek Rakhmatulloyev — website frontend and object-box annotations for fine-tuning.

## Repository layout

- `run_submission.py`, `evaluate.py`: unchanged starter scripts.
- `solution.py`: organizer interface; shared detector/tracker pass, event integration, and `RiskEstimator`.
- `detections/`: scene alignment, event rules, and the Part B risk model (`risk.py`).
- `tests/`: unit tests for the rules (`python -m pytest tests`).
- `weights/`: model weights.
- `annotations/`: team-reviewed development labels.
- `web/`: static Next.js website.
- `api/`: live inference API.
