# Lorem Ipsum — WIUT traffic event analysis

This repository is the offline inference submission and the source for our team website. It accepts fixed road-camera MP4 files and returns event intervals in the organizer's 14-class format.

## Install and run

Use Python 3.10 or 3.11. The repository includes `weights/best.pt` (about 40 MB); no download step is needed. From this directory:

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json --team "Lorem Ipsum"
python evaluate.py --pred predictions.json --validate-only
```

`run_submission.py` and `evaluate.py` are unchanged copies from the starter kit. `solution.py` is the required interface. The model and scene data are bundled, so inference does not need internet access. On CUDA machines, the detector uses the first GPU; otherwise it uses CPU. Inference samples at 5 FPS on GPU or 0.5 FPS on CPU by default. `WIUT_INFERENCE_FPS` overrides the rate. Part B is optional and unavailable in this version: the harness logs `NotImplementedError` and keeps the Part A events without decoding the video a second time.

## Approach

One YOLO11 medium model fine-tuned for bicycle, bus, car, green light, motorcycle, person, red light, and truck detects objects. Ultralytics ByteTrack associates detections over time. We run this detector and tracker once per sampled frame and share the resulting tracks among rule modules. `scene.json` contains hand-drawn road, lane, crosswalk, stop-line, island, and waiting-area geometry. `scene_reference.jpg` is the associated reference frame. A SIFT/RANSAC homography aligns that geometry to each video, with quality gates and an identity fallback.

The event rules use position, line crossing, tracker history, and detected traffic-light state. The integrated pipeline currently emits `jaywalking`, `failure_to_yield`, `red_light`, `stop_line`, `solid_line_crossing`, `illegal_turn`, and `congestion`. Overlapping intervals of each class are merged before returning. The other official classes remain in `CLASSES` but have no detector yet. Part B accident anticipation is not implemented, and the website leaves risk blank until a calibrated model exists.

The detector is learned; tracking, scene alignment, event rules, and post-processing are algorithmic. The provided sample videos were manually labeled in `annotations/events.csv`; `annotations/ground_truth.json` is the merged official-format dev set. These labels are for evaluation and rule tuning, not a replacement for test data.

## Evaluate on sample videos

```bash
python scripts/make_ground_truth.py --videos /path/to/sample-videos --csv annotations/events.csv --out annotations/ground_truth.json
python run_submission.py --videos /path/to/sample-videos --out predictions_samples.json --team "Lorem Ipsum"
python evaluate.py --pred predictions_samples.json --gt annotations/ground_truth.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only
```

The four provided MP4s are deliberately excluded from Git because they total tens of GB. `predictions_samples.json` should contain the actual output of the command above, not manual labels.

## Live demo and website

The browser UI is in `web/`. The inference API is in `api/main.py`; it runs the same `solution.detect_events` code as the submission.

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

Event rules and scene alignment are deterministic for a given detection stream. Inference does not use random augmentation. Ultralytics/PyTorch and ByteTrack may vary slightly across CPU/GPU, library versions, and floating-point kernels. Python package versions are pinned in `requirements.txt`; the model file is stored directly in `weights/`. No random seed is used by the inference pipeline.

**Training data and licenses:** `best.pt` was fine-tuned only on team-annotated frames extracted from the four videos supplied for this hackathon. No public dataset was added for fine-tuning. The videos are organizer-provided competition data; no separate public redistribution license was supplied, so the videos and extracted training frames are not included in Git. The base YOLO11 model is distributed by Ultralytics under its licensing terms; see their license for reuse outside this competition. No training runs at inference time.

## Team

- Ja'farbek Yusupov — event annotations, scene layout, testing, event rules, integration, backend, and deployment.
- Husniddin Ravshanov — detector testing and fine-tuning, event rules, and solution implementation.
- Azizbek Rakhmatulloyev — website frontend and object-box annotations for fine-tuning.

## Repository layout

- `solution.py`: organizer interface.
- `run_submission.py`, `evaluate.py`: unchanged starter scripts.
- `solution.py`: shared detector/tracker pass and event integration.
- `detections/`: scene alignment and event rules.
- `weights/`: model weights.
- `annotations/`: team-reviewed development labels.
- `web/`: static Next.js website.
- `api/`: live inference API.
