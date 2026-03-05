"""Pydantic models for request/response validation."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    """Job status enumeration."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    """Request model for music generation."""

    prompt: str = Field(..., min_length=1, max_length=500, description="Text prompt for music generation")
    duration: int = Field(default=30, ge=10, le=120, description="Duration in seconds (10-120)")
    mood: str | None = Field(default=None, max_length=100, description="Optional mood modifier")


class SeparateRequest(BaseModel):
    """Request model for audio separation."""

    audio_url: str | None = Field(default=None, description="URL of audio file to separate")


class Scene(BaseModel):
    """Scene definition for soundtrack timeline."""

    mood: str | None = Field(
        default=None,
        max_length=100,
        description="Optional mood for this scene, e.g., 'tense', 'joyful', 'melancholic'",
    )
    duration: int = Field(
        ...,
        ge=5,
        le=120,
        description="Scene duration in seconds (5-120)",
    )
    prompt: str | None = Field(
        default=None,
        max_length=500,
        description="Optional scene-specific prompt override",
    )


class SoundtrackRequest(BaseModel):
    """Request model for multi-mood soundtrack generation."""

    base_prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Base musical description for thematic consistency",
    )
    scenes: list[Scene] = Field(
        ...,
        min_length=1,
        max_length=40,
        description="Scene timeline with moods and durations (up to 40 scenes)",
    )
    melody_audio_url: str | None = Field(
        default=None,
        description="URL of melody reference audio for chromagram conditioning (reserved for future)",
    )
    use_beat_aligned_crossfade: bool = Field(
        default=True,
        description="Align crossfade boundaries to beats for musical coherence",
    )
    crossfade_duration: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="Crossfade overlap duration in seconds",
    )

    @field_validator("scenes")
    @classmethod
    def validate_total_duration(cls, scenes: list[Scene]) -> list[Scene]:
        """Validate total duration is within allowed range."""
        total = sum(s.duration for s in scenes)
        if total < 30:
            raise ValueError(f"Total duration must be at least 30 seconds, got {total}s")
        if total > 1200:
            raise ValueError(f"Total duration must not exceed 1200 seconds (20 min), got {total}s")
        return scenes


class ProgressDetail(BaseModel):
    """Detailed progress information for multi-stage jobs."""

    current_scene: int | None = Field(
        default=None,
        description="Current scene being processed (1-indexed)",
    )
    total_scenes: int | None = Field(
        default=None,
        description="Total number of scenes",
    )
    stage: str | None = Field(
        default=None,
        description="Current processing stage: 'generating', 'crossfading', 'finalizing'",
    )


class JobResponse(BaseModel):
    """Response model for job creation."""

    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    """Response model for job status check."""

    job_id: str
    status: JobStatus
    result_urls: list[str] | None = None
    error: str | None = None
    progress: float | None = Field(default=None, ge=0, le=100, description="Progress percentage")
    progress_detail: ProgressDetail | None = Field(
        default=None,
        description="Detailed progress for multi-stage jobs like soundtrack generation",
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    device: str
    gpu_available: bool
    mps_available: bool
    models_loaded: dict[str, bool]


class MetricsResponse(BaseModel):
    """Response model for metrics."""

    jobs_total: int
    jobs_completed: int
    jobs_failed: int
    jobs_in_progress: int
    avg_generation_time_seconds: float | None
    avg_soundtrack_time_seconds: float | None
    avg_separation_time_seconds: float | None
