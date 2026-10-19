"""
Pydantic models for the AI Background Music Engine API.
Follows IncuBrix Integration Standard v2.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MusicStyle(str, enum.Enum):
    """Supported music style categories."""
    AMBIENT = "ambient"
    CINEMATIC = "cinematic"
    CORPORATE = "corporate"
    ELECTRONIC = "electronic"
    ACOUSTIC = "acoustic"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    HIP_HOP = "hip-hop"
    POP = "pop"
    ROCK = "rock"
    LO_FI = "lo-fi"
    CUSTOM = "custom"


class Mood(str, enum.Enum):
    """Supported mood tags."""
    HAPPY = "happy"
    SAD = "sad"
    ENERGETIC = "energetic"
    CALM = "calm"
    DRAMATIC = "dramatic"
    MYSTERIOUS = "mysterious"
    UPLIFTING = "uplifting"
    TENSE = "tense"
    ROMANTIC = "romantic"
    DARK = "dark"
    PLAYFUL = "playful"
    INSPIRATIONAL = "inspirational"


class JobStatus(str, enum.Enum):
    """Background job lifecycle states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class StemType(str, enum.Enum):
    """Audio stem types from source separation."""
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    OTHER = "other"
    FULL_MIX = "full_mix"


# ─── Request Models ───


class Section(BaseModel):
    """A section within a music track with its own mood/style."""
    start_seconds: float = Field(..., ge=0, description="Start time in seconds")
    end_seconds: float = Field(..., gt=0, description="End time in seconds")
    mood: Optional[Mood] = Field(None, description="Mood for this section")
    style: Optional[MusicStyle] = Field(None, description="Style for this section")
    prompt: Optional[str] = Field(None, max_length=500, description="Free-text prompt for this section")
    energy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Energy level 0.0-1.0")

    @field_validator("end_seconds")
    @classmethod
    def end_after_start(cls, v, info):
        if "start_seconds" in info.data and v <= info.data["start_seconds"]:
            raise ValueError("end_seconds must be greater than start_seconds")
        return v


class VoiceoverDucking(BaseModel):
    """Configuration for ducking music under voiceover segments."""
    enabled: bool = Field(True, description="Whether ducking is active")
    duck_level_db: float = Field(-12.0, ge=-30.0, le=0.0, description="How much to reduce music (dB)")
    fade_ms: int = Field(200, ge=50, le=2000, description="Fade in/out duration in ms")
    voiceover_timestamps: list[tuple[float, float]] = Field(
        default_factory=list,
        description="List of (start_s, end_s) tuples where voiceover is active"
    )


class MusicJobRequest(BaseModel):
    """Request to create a background music track."""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Describe the music you want (mood, style, instruments, tempo)"
    )
    duration_seconds: float = Field(
        ...,
        gt=0,
        le=300,
        description="Exact duration of the output track in seconds"
    )
    style: Optional[MusicStyle] = Field(None, description="Music style category")
    mood: Optional[Mood] = Field(None, description="Mood tag")
    bpm: Optional[int] = Field(None, ge=40, le=300, description="Desired tempo in BPM")
    energy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Energy level 0.0-1.0")
    sections: Optional[list[Section]] = Field(
        None,
        description="Optional sections with different moods/styles at different timestamps"
    )
    voiceover_ducking: Optional[VoiceoverDucking] = Field(
        None,
        description="Configuration for ducking music under voiceover"
    )
    output_format: str = Field("wav", pattern=r"^(wav|mp3|flac|ogg)$", description="Output audio format")
    loudness_lufs: float = Field(-16.0, ge=-24.0, le=-6.0, description="Target loudness in LUFS")
    include_stems: bool = Field(False, description="Whether to include separated stems in output")
    callback_url: Optional[str] = Field(None, description="Webhook URL for job completion notification")


# ─── Response Models ───


class StemInfo(BaseModel):
    """Information about a separated audio stem."""
    stem_type: StemType
    url: str
    duration_seconds: float
    loudness_lufs: Optional[float] = None


class MusicJobResponse(BaseModel):
    """Response for a submitted music generation job."""
    job_id: str = Field(..., description="Unique job identifier (UUID)")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime
    updated_at: Optional[datetime] = None
    progress_percent: Optional[float] = Field(None, ge=0, le=100)
    request: MusicJobRequest
    # Populated on completion:
    output_url: Optional[str] = Field(None, description="URL to download the generated track")
    stems: Optional[list[StemInfo]] = Field(None, description="Separated stems if requested")
    actual_duration_seconds: Optional[float] = None
    actual_loudness_lufs: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    """Paginated list of jobs."""
    jobs: list[MusicJobResponse]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    """Health check response per Integration Standard."""
    status: str = "ok"
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response per Integration Standard."""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
