"""
AI Background Music Engine — Main FastAPI Application.
IncuBrix Engineering Assessment PS1.

Provides REST API for generating AI background music with stem separation.
Follows IncuBrix Integration Standard v2.
"""
from __future__ import annotations

import math
import numpy as np
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
# Ensure python-multipart is available in container environments
try:
    import multipart
except ImportError:
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-multipart", "--quiet"])
    except Exception:
        pass

import structlog
from fastapi import FastAPI, HTTPException, Header, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.schemas import (
    ErrorResponse,
    HealthResponse,
    JobListResponse,
    JobStatus,
    MusicJobRequest,
    MusicJobResponse,
    RemixRequest,
    RemixResponse,
)

# ─── Structured logging ───
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

# ─── In-memory job store (will be replaced with Redis in production) ───
jobs_store: dict[str, MusicJobResponse] = {}
idempotency_store: dict[str, str] = {}  # idempotency_key -> job_id

# ─── App startup time for health endpoint ───
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    global _start_time
    _start_time = time.time()
    settings = get_settings()
    logger.info(
        "service_starting",
        musicgen_model=settings.musicgen_model,
        music_adapter=settings.music_adapter,
        api_port=settings.api_port,
    )
    yield
    logger.info("service_stopping")


# ─── FastAPI App ───
app = FastAPI(
    title="IncuBrix AI Background Music Engine",
    description=(
        "Generate original, commercially safe instrumental music "
        "that fits a video exactly — with real instrument stems."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/api/v1/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static file serving for generated outputs ───
from fastapi.staticfiles import StaticFiles
from pathlib import Path as _Path

_output_dir = _Path("./output")
_output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(_output_dir)), name="output")


# ─── Middleware: Correlation ID ───
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Inject X-Correlation-Id into request state and response headers."""
    correlation_id = request.headers.get("X-Correlation-Id", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-Id"] = correlation_id
    return response


# ─── Root: Interactive Audio Studio UI ───
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the interactive IncuBrix AI Background Music Studio UI."""
    studio_html_path = _Path(__file__).parent / "ui" / "studio.html"
    if studio_html_path.exists():
        return HTMLResponse(content=studio_html_path.read_text(encoding="utf-8"))
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# ─── Health endpoints (Integration Standard #8) ───

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Basic health check — is the service running?"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/ready", response_model=HealthResponse, tags=["Health"])
async def readiness_check():
    """Readiness check — verifies model adapter availability and storage directory."""
    from app.jobs.music_job import get_adapter, OUTPUT_BASE

    # 1. Check storage directory is writable
    if not OUTPUT_BASE.exists():
        try:
            OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Storage directory not writable: {e}")

    # 2. Check music generation adapter is ready
    try:
        adapter = get_adapter()
        if not adapter.is_available():
            raise HTTPException(status_code=503, detail="Music adapter not available")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Adapter initialization error: {e}")

    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 2),
    )


# ─── Music Job Endpoints (Integration Standard #2, #3, #4) ───

@app.post(
    "/api/v1/music/jobs",
    response_model=MusicJobResponse,
    status_code=202,
    tags=["Music Jobs"],
    responses={
        202: {"description": "Job accepted and queued"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        409: {"model": ErrorResponse, "description": "Idempotency conflict"},
    },
)
async def create_music_job(
    request_body: MusicJobRequest,
    request: Request,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-Id"),
):
    """
    Submit a music generation job.

    Returns 202 Accepted with a job_id. Poll GET /api/v1/music/jobs/{job_id}
    for status updates.

    Include an Idempotency-Key header to safely retry requests.
    """
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

    # ── Idempotency check (Integration Standard #4) ──
    if idempotency_key:
        if idempotency_key in idempotency_store:
            existing_job_id = idempotency_store[idempotency_key]
            if existing_job_id in jobs_store:
                logger.info(
                    "idempotent_request_returned",
                    idempotency_key=idempotency_key,
                    job_id=existing_job_id,
                    correlation_id=correlation_id,
                )
                return jobs_store[existing_job_id]

    # ── Validate request ──
    # Check prompt doesn't request real songs or vocals
    prompt_lower = request_body.prompt.lower()
    warnings = []

    # Check for real song/artist names (must stay instrumental, must not copy)
    copyrighted_indicators = ["cover of", "like the song", "copy of", "remix of"]
    for indicator in copyrighted_indicators:
        if indicator in prompt_lower:
            warnings.append(
                f"Request mentions '{indicator}' — output will be original instrumental "
                f"music inspired by the described mood, not a copy of any existing work."
            )

    # Check for vocal requests
    vocal_indicators = ["vocals", "singing", "lyrics", "spoken word", "rap", "verse"]
    for indicator in vocal_indicators:
        if indicator in prompt_lower:
            warnings.append(
                f"Request mentions '{indicator}' — output will be instrumental only. "
                f"Vocal elements are not supported."
            )

    # ── Create job ──
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    job = MusicJobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=now,
        updated_at=now,
        progress_percent=0.0,
        request=request_body,
    )

    # Store job
    jobs_store[job_id] = job
    if idempotency_key:
        idempotency_store[idempotency_key] = job_id

    logger.info(
        "job_created",
        job_id=job_id,
        prompt=request_body.prompt[:100],
        duration_s=request_body.duration_seconds,
        correlation_id=correlation_id,
    )

    # ── Enqueue background task ──
    # Import here to avoid circular imports
    from app.jobs.music_job import process_music_job

    try:
        process_music_job.delay(job_id)
    except Exception as e:
        logger.warning(
            "celery_unavailable_running_sync",
            job_id=job_id,
            error=str(e),
        )
        # If Celery/Redis not available, run synchronously for demo
        # This is a fallback — production should always use Celery
        import threading
        thread = threading.Thread(
            target=_run_job_sync,
            args=(job_id,),
            daemon=True,
        )
        thread.start()

    return job


def _run_job_sync(job_id: str):
    """Fallback: run job synchronously when Celery is not available."""
    from app.jobs.music_job import _process_music_job_impl
    try:
        _process_music_job_impl(job_id)
    except Exception as e:
        logger.error("sync_job_failed", job_id=job_id, error=str(e))
        if job_id in jobs_store:
            jobs_store[job_id].status = JobStatus.FAILED
            jobs_store[job_id].error = str(e)
            jobs_store[job_id].updated_at = datetime.now(timezone.utc)


@app.get(
    "/api/v1/music/jobs/{job_id}",
    response_model=MusicJobResponse,
    tags=["Music Jobs"],
    responses={404: {"model": ErrorResponse}},
)
async def get_job_status(job_id: str):
    """Get the current status and results of a music generation job."""
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "JOB_NOT_FOUND",
                    "message": f"Job '{job_id}' not found",
                    "retryable": False,
                }
            },
        )
    return jobs_store[job_id]


@app.get(
    "/api/v1/music/jobs",
    response_model=JobListResponse,
    tags=["Music Jobs"],
)
async def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[JobStatus] = Query(None),
):
    """List all music generation jobs with optional filtering."""
    all_jobs = list(jobs_store.values())

    # Filter by status
    if status:
        all_jobs = [j for j in all_jobs if j.status == status]

    # Sort by created_at descending
    all_jobs.sort(key=lambda j: j.created_at, reverse=True)

    # Paginate
    total = len(all_jobs)
    start = (page - 1) * page_size
    end = start + page_size
    page_jobs = all_jobs[start:end]

    return JobListResponse(
        jobs=page_jobs,
        total=total,
        page=page,
        page_size=page_size,
    )


@app.delete(
    "/api/v1/music/jobs/{job_id}",
    status_code=204,
    tags=["Music Jobs"],
    responses={404: {"model": ErrorResponse}},
)
async def cancel_job(job_id: str):
    """Cancel a pending or processing job."""
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "JOB_NOT_FOUND",
                    "message": f"Job '{job_id}' not found",
                    "retryable": False,
                }
            },
        )

    job = jobs_store[job_id]
    if job.status in (JobStatus.PENDING, JobStatus.PROCESSING):
        job.status = JobStatus.FAILED
        job.error = "Job cancelled by user"
        job.updated_at = datetime.now(timezone.utc)
        logger.info("job_cancelled", job_id=job_id)

    return Response(status_code=204)


# ─── Audio Inspection & Stem Separation Endpoint ───

@app.post(
    "/api/v1/audio/analyze",
    tags=["Audio Inspection"],
    summary="Upload and inspect audio file with stem separation and broadcast compliance",
)
async def analyze_uploaded_audio(
    request: Request,
):
    """
    Upload an audio file (WAV, MP3, FLAC, OGG) to:
    1. Measure ITU-R BS.1770-4 integrated loudness & true peak.
    2. Optionally normalize to broadcast standard (-16 LUFS, <= -1 dBTP).
    3. Separate into 4 instrument stems (Drums, Bass, Melody, Harmony).
    4. Verify lossless float32 stem summation (< -40 dBFS difference).
    """
    import soundfile as sf
    from app.audio.loudness import measure_loudness, normalize_loudness
    from app.audio.stems import separate_stems, verify_stems_sum_to_mix

    contents = b""
    filename = "uploaded_audio.wav"
    normalize = False
    target_lufs = -16.0
    separate = True

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
            file_item = form.get("file")
            if file_item is not None and hasattr(file_item, "read"):
                contents = await file_item.read()
                filename = getattr(file_item, "filename", "uploaded_audio.wav")
            elif isinstance(file_item, (bytes, bytearray)):
                contents = bytes(file_item)

            norm_val = form.get("normalize", "false")
            normalize = str(norm_val).lower() in ("true", "1", "yes")
            target_lufs = float(form.get("target_lufs", -16.0))
            sep_val = form.get("separate", "true")
            separate = str(sep_val).lower() in ("true", "1", "yes")
        except Exception:
            contents = await request.body()
    else:
        contents = await request.body()
        filename = request.headers.get("X-Filename", "uploaded_audio.wav")
        normalize = request.query_params.get("normalize", "false").lower() in ("true", "1", "yes")
        target_lufs = float(request.query_params.get("target_lufs", -16.0))
        separate = request.query_params.get("separate", "true").lower() in ("true", "1", "yes")

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty or missing.")

    ext = _Path(filename).suffix.lower()
    allowed_exts = [".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"]
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed formats: {', '.join(allowed_exts)}",
        )

    upload_id = str(uuid.uuid4())
    upload_dir = _output_dir / "uploads" / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    raw_path = upload_dir / f"original{ext}"

    raw_path.write_bytes(contents)
    del contents  # Free upload bytes from memory

    try:
        info = sf.info(str(raw_path))
        original_sr = info.samplerate
        original_frames = info.frames
        original_duration = original_frames / original_sr
        max_analyze_duration = 30.0
        truncated = original_duration > max_analyze_duration
        max_frames = int(original_sr * max_analyze_duration)
        read_frames = min(max_frames, original_frames)
        audio, sr = sf.read(str(raw_path), frames=read_frames, dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio file: {e}")

    # Downmix stereo to mono to save memory (halves RAM usage)
    original_channels = audio.shape[1] if audio.ndim > 1 else 1
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)

    duration = len(audio) / sr
    channels = original_channels

    # Measure initial loudness
    initial_metrics = measure_loudness(audio, sr)

    # Optional normalization
    if normalize:
        processed_audio = normalize_loudness(audio, sr, target_lufs=target_lufs, max_true_peak_dbtp=-1.0)
        final_metrics = measure_loudness(processed_audio, sr)
    else:
        processed_audio = audio
        final_metrics = initial_metrics

    # Save master WAV in 32-bit float for lossless stem operations
    master_path = upload_dir / "master.wav"
    sf.write(str(master_path), processed_audio, sr, subtype="FLOAT")
    master_url = f"/output/uploads/{upload_id}/master.wav"

    # Free audio arrays from memory before stem separation
    del audio, processed_audio
    import gc; gc.collect()

    # Separate stems
    stems_dict = {}
    stem_verification = None
    if separate:
        stems_dir = upload_dir / "stems"
        sep_res = separate_stems(master_path, stems_dir)

        stem_paths = []
        for s in sep_res.stems:
            stem_rel = f"/output/uploads/{upload_id}/stems/{s.stem_name}.wav"
            stems_dict[s.stem_name] = stem_rel
            stem_paths.append(s.path)

        verification = verify_stems_sum_to_mix(master_path, stem_paths)
        stem_verification = {
            "passes": verification["passes"],
            "difference_db": verification["difference_db"],
            "tolerance_db": verification["tolerance_db"],
            "mix_rms": verification["mix_rms"],
            "diff_rms": verification["diff_rms"],
        }

    is_compliant = (
        abs(final_metrics["loudness_lufs"] - (-16.0)) <= 1.0 and
        final_metrics["true_peak_dbtp"] <= -1.0
    )

    logger.info(
        "audio_upload_analyzed",
        upload_id=upload_id,
        filename=filename,
        duration_s=float(round(duration, 3)),
        lufs=float(final_metrics["loudness_lufs"]),
        peak=float(final_metrics["true_peak_dbtp"]),
        stems=list(stems_dict.keys()),
        stems_verified=bool(stem_verification["passes"]) if stem_verification else None,
    )

    return {
        "upload_id": upload_id,
        "filename": filename,
        "duration_seconds": float(round(duration, 3)),
        "original_duration_seconds": float(round(original_duration, 3)),
        "truncated_to_preview": bool(truncated),
        "note": f"Analyzed first {duration:.1f}s window (optimized for short-form video background music)" if truncated else None,
        "sample_rate": int(sr),
        "channels": int(channels),
        "initial_loudness_lufs": float(initial_metrics["loudness_lufs"]),
        "initial_true_peak_dbtp": float(initial_metrics["true_peak_dbtp"]),
        "loudness_lufs": float(final_metrics["loudness_lufs"]),
        "true_peak_dbtp": float(final_metrics["true_peak_dbtp"]),
        "broadcast_compliant": bool(is_compliant),
        "normalization_applied": bool(normalize),
        "master_url": master_url,
        "stems": stems_dict,
        "stem_verification": {
            "passes": bool(stem_verification["passes"]),
            "difference_db": float(-100.0 if (math.isnan(stem_verification["difference_db"]) or math.isinf(stem_verification["difference_db"])) else stem_verification["difference_db"]),
            "tolerance_db": float(stem_verification["tolerance_db"]),
            "mix_rms": float(0.0 if (math.isnan(stem_verification["mix_rms"]) or math.isinf(stem_verification["mix_rms"])) else stem_verification["mix_rms"]),
            "diff_rms": float(0.0 if (math.isnan(stem_verification["diff_rms"]) or math.isinf(stem_verification["diff_rms"])) else stem_verification["diff_rms"]),
        } if stem_verification else None,
    }


# ─── Portion-Based Stem Automation & Remix Endpoint ───

@app.post(
    "/api/v1/music/remix",
    tags=["Music Jobs"],
    summary="Render custom mix with dynamic portion-based stem automation",
    response_model=RemixResponse,
)
async def remix_track_portions(request: RemixRequest):
    """
    Render a custom track where individual stems (drums, bass, melody, harmony)
    are dynamically modulated across specific portions of the timeline.
    """
    import soundfile as sf
    from app.audio.loudness import normalize_loudness

    if request.job_id:
        stems_dir = _output_dir / request.job_id / "stems"
        out_mix_path = _output_dir / request.job_id / "custom_mix.wav"
        rel_url = f"/output/{request.job_id}/custom_mix.wav"
    elif request.upload_id:
        stems_dir = _output_dir / "uploads" / request.upload_id / "stems"
        out_mix_path = _output_dir / "uploads" / request.upload_id / "custom_mix.wav"
        rel_url = f"/output/uploads/{request.upload_id}/custom_mix.wav"
    else:
        if not jobs_store:
            raise HTTPException(status_code=400, detail="No active or previous job found to remix.")
        latest_job_id = list(jobs_store.keys())[-1]
        stems_dir = _output_dir / latest_job_id / "stems"
        out_mix_path = _output_dir / latest_job_id / "custom_mix.wav"
        rel_url = f"/output/{latest_job_id}/custom_mix.wav"

    if not stems_dir.exists():
        raise HTTPException(status_code=404, detail="Stem files not found for the requested track.")

    stem_files = {
        "drums": stems_dir / "drums.wav",
        "bass": stems_dir / "bass.wav",
        "melody": stems_dir / "melody.wav",
        "other": stems_dir / "other.wav",
    }
    if not stem_files["melody"].exists() and (stems_dir / "vocals.wav").exists():
        stem_files["melody"] = stems_dir / "vocals.wav"

    loaded_stems = {}
    sr = 48000
    max_len = 0

    for name, path in stem_files.items():
        if path.exists():
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            loaded_stems[name] = data
            max_len = max(max_len, len(data))

    if not loaded_stems:
        raise HTTPException(status_code=404, detail="No stems could be loaded.")

    for name in loaded_stems:
        if len(loaded_stems[name]) < max_len:
            loaded_stems[name] = np.pad(loaded_stems[name], (0, max_len - len(loaded_stems[name])))

    total_duration = max_len / sr
    mixed_audio = np.zeros(max_len, dtype=np.float32)

    for stem_name, stem_audio in loaded_stems.items():
        gain_envelope = np.ones(max_len, dtype=np.float32)
        for p in request.portions:
            start_idx = max(0, int(p.start_s * sr))
            end_idx = min(max_len, int(p.end_s * sr))
            if start_idx >= end_idx:
                continue

            if stem_name == "drums":
                target_g = p.gains.drums
            elif stem_name == "bass":
                target_g = p.gains.bass
            elif stem_name == "melody":
                target_g = p.gains.melody
            else:
                target_g = p.gains.harmony

            gain_envelope[start_idx:end_idx] = target_g

        mixed_audio += stem_audio * gain_envelope

    from app.audio.loudness import measure_loudness, normalize_loudness

    final_audio = normalize_loudness(
        mixed_audio,
        sr,
        target_lufs=-16.0,
        max_true_peak_dbtp=-1.0,
    )
    metrics = measure_loudness(final_audio, sr)
    final_lufs = metrics["loudness_lufs"]
    final_peak = metrics["true_peak_dbtp"]

    sf.write(str(out_mix_path), final_audio, sr, subtype="PCM_16")

    return RemixResponse(
        status="completed",
        custom_mix_url=rel_url,
        duration_seconds=float(round(total_duration, 2)),
        loudness_lufs=float(final_lufs),
        true_peak_dbtp=float(final_peak),
        portions_count=len(request.portions),
    )


# ─── Entry point ───

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
