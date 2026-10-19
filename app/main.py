"""
AI Background Music Engine — Main FastAPI Application.
IncuBrix Engineering Assessment PS1.

Provides REST API for generating AI background music with stem separation.
Follows IncuBrix Integration Standard v2.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

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


# ─── Root Redirect to Docs ───
from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive Swagger API docs."""
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
