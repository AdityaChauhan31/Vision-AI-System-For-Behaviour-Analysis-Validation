"""
api/app.py
-----------
FastAPI surface for the Vision AI platform.

Endpoints
  GET  /                       → the web UI
  GET  /api/health             → provider + readiness
  GET  /api/use-cases          → configured use cases (for the dropdown)
  POST /api/analyze            → upload a video → {job_id}
  POST /api/analyze-demo       → run on the bundled demo clip → {job_id}
  GET  /api/jobs/{job_id}      → status + streaming frames + verdict + alerts
  GET  /api/frames/{job}/{f}   → a sampled frame JPEG (evidence thumbnails)

Run:  uvicorn api.app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# Importing config.settings parses .env into os.environ. Must happen BEFORE any
# os.environ.get() for keys/LOG_LEVEL below, or the API can't see .env values.
import config.settings  # noqa: F401,E402

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from api.jobs import FRAMES_ROOT, MAX_FRAMES, STORE, effective_provider, provider_status

HERE        = Path(__file__).parent
UI_FILE     = HERE / "static" / "index.html"
UPLOAD_DIR  = Path(tempfile.gettempdir()) / "vision_ai_uploads"
DEMO_VIDEO  = Path("data/demo_housekeeping.mp4")
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD  = int(os.environ.get("MAX_UPLOAD_MB", "60")) * 1024 * 1024

app = FastAPI(title="Vision AI — Behaviour Validation", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    if not DEMO_VIDEO.exists():
        try:
            from tools.make_demo_video import make_video
            make_video(str(DEMO_VIDEO), seconds=18)
            logger.info("Generated demo clip at %s", DEMO_VIDEO)
        except Exception as exc:                 # noqa: BLE001
            logger.warning("Could not generate demo clip: %s", exc)


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if UI_FILE.exists():
        return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Vision AI API</h1><p>UI file missing.</p>")


# ── meta ──────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    provider, note = effective_provider(None)
    return {
        "status": "ok",
        "provider": provider,            # the effective default (after key check)
        "note": note,
        "providers": provider_status(),  # [{id, ready}] for the UI dropdown
        "max_frames_per_job": MAX_FRAMES,
    }


@app.get("/api/use-cases")
def use_cases() -> dict:
    path = Path("config/use_cases.yaml")
    out = []
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        for uc in raw.get("use_cases", []):
            out.append({
                "id": uc["id"],
                "name": uc.get("name", uc["id"]),
                "identity_required": uc.get("identity_required", False),
            })
    return {"use_cases": out}


# ── analysis ──────────────────────────────────────────────────────────────────

def _launch(video_path: Path, use_case: str, interval: float, provider: str | None):
    job = STORE.create(video_path, use_case, interval, provider)
    return JSONResponse({"job_id": job.id})


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    use_case: str = Form("housekeeping_demo_short"),
    interval: float = Form(10.0),
    provider: str | None = Form(None),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    dest = UPLOAD_DIR / f"{os.urandom(6).hex()}{ext}"
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large (>{MAX_UPLOAD // (1024*1024)} MB)")
            f.write(chunk)
    return _launch(dest, use_case, interval, provider)


@app.post("/api/analyze-demo")
def analyze_demo(use_case: str = Form("housekeeping_demo_short"),
                 interval: float = Form(5.0),
                 provider: str | None = Form(None)):
    if not DEMO_VIDEO.exists():
        raise HTTPException(503, "Demo video not available.")
    return _launch(DEMO_VIDEO, use_case, interval, provider)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.snapshot()


@app.get("/api/frames/{job_id}/{filename}")
def frame(job_id: str, filename: str) -> FileResponse:
    # prevent path traversal
    if "/" in filename or ".." in filename or "\\" in filename:
        raise HTTPException(400, "Bad filename")
    path = FRAMES_ROOT / job_id / filename
    if not path.exists():
        raise HTTPException(404, "Frame not found")
    return FileResponse(path, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("api.app:app", host="0.0.0.0", port=port)
