"""FastAPI application for MusicGen + ACE-Step + Demucs API."""

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.acestep_runner import acestep
from app.demucs_runner import demucs
from app.job_queue import Job, JobType, job_queue
from app.models import (
    ACEStepRequest,
    GenerateRequest,
    HealthResponse,
    JobResponse,
    JobStatus,
    JobStatusResponse,
    MetricsResponse,
    ProgressDetail,
    SeparateRequest,
    SoundtrackRequest,
)
from app.musicgen import musicgen

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv("API_KEY", "")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Verify API key if configured.

    Args:
        x_api_key: API key from header

    Raises:
        HTTPException: If API key is invalid
    """
    if not API_KEY:
        return  # No API key configured, allow all requests

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")


async def handle_generate_job(job: Job) -> list[str]:
    """Handle music generation job.

    Args:
        job: Job to process

    Returns:
        List of result URLs
    """
    params = job.params
    output_path = await musicgen.generate(
        prompt=params["prompt"],
        duration=params["duration"],
        mood=params.get("mood"),
        job_id=job.id,
        progress_callback=lambda p: update_job_progress(job, p),
    )
    return [f"/files/{Path(output_path).name}"]


async def handle_separate_job(job: Job) -> list[str]:
    """Handle audio separation job.

    Args:
        job: Job to process

    Returns:
        List of result URLs
    """
    params = job.params
    output_paths = await demucs.separate(
        audio_path=params.get("audio_path"),
        audio_url=params.get("audio_url"),
        job_id=job.id,
        progress_callback=lambda p: update_job_progress(job, p),
    )
    return [f"/files/{Path(p).name}" for p in output_paths]


async def handle_soundtrack_job(job: Job) -> list[str]:
    """Handle multi-mood soundtrack generation job.

    Args:
        job: Job to process

    Returns:
        List of result URLs
    """
    from app.soundtrack import soundtrack_generator

    params = job.params

    async def progress_callback(
        progress: float,
        current_scene: int | None = None,
        total_scenes: int | None = None,
        stage: str | None = None,
    ):
        job.progress = progress
        if any([current_scene, total_scenes, stage]):
            job.progress_detail = {
                "current_scene": current_scene,
                "total_scenes": total_scenes,
                "stage": stage,
            }

    if soundtrack_generator is None:
        raise RuntimeError("Soundtrack generator not initialized")
    output_path = await soundtrack_generator.generate_soundtrack(
        scenes=params["scenes"],
        base_prompt=params["base_prompt"],
        melody_audio_url=params.get("melody_audio_url"),
        use_beat_aligned_crossfade=params.get("use_beat_aligned_crossfade", True),
        crossfade_duration=params.get("crossfade_duration", 2.0),
        job_id=job.id,
        progress_callback=progress_callback,
    )
    return [f"/files/{Path(output_path).name}"]


async def handle_acestep_job(job: Job) -> list[str]:
    """Handle ACE-Step music generation job.

    Args:
        job: Job to process

    Returns:
        List of result URLs
    """
    params = job.params
    output_path = await acestep.generate(
        prompt=params["prompt"],
        duration=params.get("duration", 30.0),
        lyrics=params.get("lyrics", ""),
        instrumental=params.get("instrumental", True),
        infer_steps=params.get("infer_steps", 8),
        guidance_scale=params.get("guidance_scale", 7.0),
        seed=params.get("seed", -1),
        audio_format=params.get("audio_format", "wav"),
        thinking=params.get("thinking", True),
        job_id=job.id,
        progress_callback=lambda p: update_job_progress(job, p),
    )
    return [f"/files/{Path(output_path).name}"]


async def update_job_progress(job: Job, progress: float) -> None:
    """Update job progress.

    Args:
        job: Job to update
        progress: Progress percentage
    """
    job.progress = progress


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Initialize soundtrack generator
    import app.soundtrack as soundtrack_module
    from app.soundtrack import SoundtrackGenerator

    soundtrack_module.soundtrack_generator = SoundtrackGenerator(
        musicgen, output_dir=str(OUTPUT_DIR)
    )

    # Register job handlers
    job_queue.register_handler(JobType.GENERATE, handle_generate_job)
    job_queue.register_handler(JobType.GENERATE_SOUNDTRACK, handle_soundtrack_job)
    job_queue.register_handler(JobType.SEPARATE, handle_separate_job)
    job_queue.register_handler(JobType.GENERATE_ACESTEP, handle_acestep_job)

    # Start job queue
    await job_queue.start()
    logger.info("Job queue started")

    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    # Stop cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Stop job queue
    await job_queue.stop()
    logger.info("Job queue stopped")

    # Unload models
    musicgen.unload_model()
    demucs.unload_model()
    acestep.unload_model()
    logger.info("Models unloaded")


async def cleanup_loop():
    """Periodic cleanup of old jobs and files."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await job_queue.cleanup_old_jobs(max_age_hours=24)
            await cleanup_old_files(max_age_hours=24)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Cleanup error: {e}")


async def cleanup_old_files(max_age_hours: int = 24):
    """Clean up old output files.

    Args:
        max_age_hours: Maximum age in hours
    """
    import time

    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0

    for file_path in OUTPUT_DIR.iterdir():
        if file_path.is_file() and file_path.stat().st_mtime < cutoff:
            file_path.unlink()
            removed += 1

    if removed:
        logger.info(f"Cleaned up {removed} old files")


app = FastAPI(
    title="MusicGen + ACE-Step + Demucs API",
    description="AI Music Generation (MusicGen & ACE-Step 1.5) and Stem Separation API",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for liveness/readiness probes."""
    from app.musicgen import get_device

    device = get_device()
    return HealthResponse(
        status="healthy",
        device=device,
        gpu_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
        models_loaded={
            "musicgen": musicgen.is_loaded,
            "demucs": demucs.is_loaded,
            "acestep": acestep.is_loaded,
        },
    )


@app.post("/generate", response_model=JobResponse, dependencies=[Depends(verify_api_key)])
async def generate(request: GenerateRequest):
    """Generate music from text prompt.

    Args:
        request: Generation request with prompt, duration, and optional mood

    Returns:
        Job response with job_id and status
    """
    job = await job_queue.enqueue(
        JobType.GENERATE,
        {
            "prompt": request.prompt,
            "duration": request.duration,
            "mood": request.mood,
        },
    )
    return JobResponse(job_id=job.id, status=JobStatus.QUEUED)


@app.post(
    "/generate/acestep",
    response_model=JobResponse,
    dependencies=[Depends(verify_api_key)],
)
async def generate_acestep(request: ACEStepRequest):
    """Generate music using ACE-Step 1.5.

    Full-song generation with vocals, lyrics, and advanced music understanding.
    Optimized for budget GPUs (8GB VRAM) with turbo mode (8 diffusion steps).

    Args:
        request: ACE-Step generation request

    Returns:
        Job response with job_id and status
    """
    job = await job_queue.enqueue(
        JobType.GENERATE_ACESTEP,
        {
            "prompt": request.prompt,
            "duration": request.duration,
            "lyrics": request.lyrics,
            "instrumental": request.instrumental,
            "infer_steps": request.infer_steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "audio_format": request.audio_format,
            "thinking": request.thinking,
        },
    )
    return JobResponse(job_id=job.id, status=JobStatus.QUEUED)


@app.post("/separate", response_model=JobResponse, dependencies=[Depends(verify_api_key)])
async def separate(
    request: SeparateRequest | None = None,
    file: UploadFile | None = File(default=None),
):
    """Separate audio into stems.

    Args:
        request: Optional request with audio_url
        file: Optional uploaded audio file

    Returns:
        Job response with job_id and status

    Raises:
        HTTPException: If neither URL nor file is provided
    """
    audio_path = None
    audio_url = None

    if file:
        # Save uploaded file
        import uuid

        file_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix if file.filename else ".wav"
        audio_path = str(OUTPUT_DIR / f"{file_id}_upload{ext}")

        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Saved uploaded file to {audio_path}")

    elif request and request.audio_url:
        audio_url = request.audio_url
    else:
        raise HTTPException(
            status_code=400,
            detail="Either audio_url or file upload is required",
        )

    job = await job_queue.enqueue(
        JobType.SEPARATE,
        {
            "audio_path": audio_path,
            "audio_url": audio_url,
        },
    )
    return JobResponse(job_id=job.id, status=JobStatus.QUEUED)


@app.post(
    "/generate/soundtrack",
    response_model=JobResponse,
    dependencies=[Depends(verify_api_key)],
)
async def generate_soundtrack(request: SoundtrackRequest):
    """Generate multi-mood soundtrack for video.

    Creates a seamless soundtrack with mood transitions aligned to scene timeline.
    Supports optional beat-aligned crossfading for professional-quality transitions.

    Args:
        request: Soundtrack request with scene timeline

    Returns:
        Job response with job_id and status
    """
    job = await job_queue.enqueue(
        JobType.GENERATE_SOUNDTRACK,
        {
            "base_prompt": request.base_prompt,
            "scenes": [s.model_dump() for s in request.scenes],
            "melody_audio_url": request.melody_audio_url,
            "use_beat_aligned_crossfade": request.use_beat_aligned_crossfade,
            "crossfade_duration": request.crossfade_duration,
        },
    )
    return JobResponse(job_id=job.id, status=JobStatus.QUEUED)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status.

    Args:
        job_id: Job ID

    Returns:
        Job status response

    Raises:
        HTTPException: If job not found
    """
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Build progress detail if available
    progress_detail = None
    if job.progress_detail:
        progress_detail = ProgressDetail(**job.progress_detail)

    return JobStatusResponse(
        job_id=job.id,
        status=JobStatus(job.status.value),
        result_urls=job.result_urls,
        error=job.error,
        progress=job.progress,
        progress_detail=progress_detail,
    )


@app.get("/files/{filename}")
async def get_file(filename: str):
    """Download generated file.

    Args:
        filename: Filename to download

    Returns:
        File response

    Raises:
        HTTPException: If file not found
    """
    file_path = OUTPUT_DIR / filename

    # Security: ensure file is within output directory
    try:
        file_path = file_path.resolve()
        OUTPUT_DIR.resolve()
        if not str(file_path).startswith(str(OUTPUT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=403, detail="Access denied") from None

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/json", response_model=MetricsResponse)
async def metrics_json():
    """JSON metrics endpoint."""
    data = job_queue.get_metrics()
    return MetricsResponse(**data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
