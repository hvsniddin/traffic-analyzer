"""Minimal real inference API for the Next.js live demo.

Run from the repository root: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from detections.overlay import OverlayRecorder
from solution import SUPPRESSED_CLASSES, _detect_events


MAX_BYTES = int(os.getenv("WIUT_MAX_UPLOAD_MB", "200")) * 1024 * 1024
MAX_DURATION = float(os.getenv("WIUT_MAX_DURATION_SEC", "120"))
# Same sample rate as the submission by default; the demo has no 3x limit.
DEMO_FPS = float(os.getenv("WIUT_DEMO_FPS", "2"))

ORIGINS = [s.strip() for s in os.getenv("WIUT_CORS_ORIGINS", "*").split(",") if s.strip()]
app = FastAPI(title="Traffic Analyzer inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_worker = ThreadPoolExecutor(max_workers=1)


def _set(job_id: str, **changes) -> None:
    with _lock:
        _jobs[job_id].update(changes)


def _run(job_id: str, path: Path, duration: float, fps: float, frames: float,
         width: int, height: int) -> None:
    _set(job_id, status="running", stage="Detecting and tracking road users", progress=0.0)
    recorder = OverlayRecorder(
        fps, width, height,
        on_progress=lambda index: _set(job_id, progress=min(0.99, index / frames)) if frames > 0 else None,
        hidden_labels=SUPPRESSED_CLASSES,
    )

    try:
        events = _detect_events(str(path), on_sample=recorder, sample_rate=DEMO_FPS,
                                time_budget_factor=float("inf"))
        _set(
            job_id,
            status="done",
            stage="Complete",
            progress=1.0,
            result={"duration_sec": duration, "events": events, "risk": recorder.risk,
                    "overlay": recorder.overlay()},
        )
    except Exception as exc:
        _set(job_id, status="error", stage="Failed", error=str(exc))
    finally:
        path.unlink(missing_ok=True)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile) -> dict:
    if not file.filename or Path(file.filename).suffix.lower() != ".mp4":
        raise HTTPException(400, "Upload an MP4 file")
    fd, tmp_name = tempfile.mkstemp(prefix="wiut-demo-", suffix=".mp4")
    path = Path(tmp_name)
    import os as _os
    _os.close(fd)
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, f"Video exceeds {MAX_BYTES // 1048576} MB")
                out.write(chunk)
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise HTTPException(400, "Cannot decode this MP4")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frames / fps if fps > 0 else 0
        finally:
            cap.release()
        if not 0 < duration <= MAX_DURATION:
            raise HTTPException(400, f"Video must be at most {MAX_DURATION:g} seconds")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "Waiting for inference worker",
            "progress": None,
            "result": None,
            "error": None,
        }
    _worker.submit(_run, job_id, path, duration, fps, frames, width, height)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _lock:
        if job_id not in _jobs:
            raise HTTPException(404, "Job not found")
        return dict(_jobs[job_id])
