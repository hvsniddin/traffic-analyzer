# Traffic Analyzer website

Public site for the WIUT Hackathon 2026 CV task: live demo, annotated sample
results, EDA, approach, report and team. Next.js 16 static export; the live demo
calls the inference API straight from the browser.

## Run

```bash
cd web
npm install
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL, or leave it empty for mock mode
npm run dev                  # http://localhost:3000
npm run build                # static site in web/out/, deploy to any static host
```

With `NEXT_PUBLIC_API_URL` empty the demo runs against a built-in mock and
labels every result as mock output. Set the variable at build time; it is baked
into the static files.

## API contract the demo expects

From `docs/demo_plan.md` in the parent repository:

* `POST {API}/api/jobs`, multipart field `file` (MP4) → `202 {"job_id": "..."}`.
  Errors: any 4xx/5xx with `{"detail": "readable message"}` (FastAPI default).
* `GET {API}/api/jobs/{job_id}` →
  `{"job_id", "status": "queued|running|done|error", "stage", "progress": 0–1 | null, "result": {"duration_sec", "events": [[start, end, label]], "risk": [[t, score]] | null, "annotated_video_url"?: string} | null, "error": string | null}`.

The page polls every 1.5 s, gives up after four failed polls in a row, and
validates every response with zod (`types/job.ts`). The API must allow CORS from
the site origin. Keep the published limits (`NEXT_PUBLIC_MAX_UPLOAD_MB`,
`NEXT_PUBLIC_MAX_DURATION_SEC`) equal to what the API enforces.

## Sample data

Each sample page reads `public/samples/<id>.json` at build time, written by
`scripts/build_sample_data.py`. Sections without input are written as `null`
and shown as pending, never faked.

```bash
# 1. 540p preview + 1 fps 1080p frames in one decode (per clip)
ffmpeg -i ../../dataset/C3896.MP4 -an -filter_complex \
  "[0:v]split=2[a][b];[a]scale=960:540,format=yuv420p[p];[b]fps=1,scale=1920:1080[f]" \
  -map "[p]" -c:v libx264 -preset veryfast -crf 30 -movflags +faststart public/samples/C3896/preview.mp4 \
  -map "[f]" -q:v 3 /tmp/frames/C3896/%05d.jpg

# 2. JSON, poster, heatmaps, detector counts (from the parent repo root)
python traffic-analyzer/web/scripts/build_sample_data.py --videos dataset --frames /tmp/frames \
  --labels annotations/events.csv annotations/events_codex.csv \
  --weights weights/fixed_names.pt --predictions traffic-analyzer/predictions_samples.json
```

To show annotated playback, render `public/samples/<id>/annotated.mp4` with
`utils/annotate_video.py` (H.264, 540p, `+faststart`) and re-run step 2.

## Layout

```text
app/                 routes: / demo samples/[id] eda approach report team
components/video/    player + clickable event timeline (shared by demo and samples)
components/demo/     upload, validation, job status
components/samples/  sample cards, results, EDA charts
components/charts/   recharts wrappers
lib/api/             typed fetch (ApiResult), jobs service and mock
hooks/               job polling
content/             team, approach and per-class rule text (edit here)
types/               event classes, job contract, sample JSON shape
```
