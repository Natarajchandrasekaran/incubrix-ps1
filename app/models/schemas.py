"""
Pydantic models for the AI Background Music Engine API.
Follows IncuBrix Integration Standard v2.

Field names align with the brief's API specification:
- duration_s (brief) mapped from duration_seconds
- tempo_bpm (brief) mapped from bpm
- narration_url (brief) for voiceover ducking
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


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
    MELODY = "melody"
    HARMONY = "harmony"
    VOCALS = "vocals"
    OTHER = "other"
    FULL_MIX = "full_mix"


# ─── Request Models ───


class Section(BaseModel):
    """A section within a music track with its own mood/style."""
    start_s: float = Field(
        ..., ge=0, description="Start time in seconds",
        validation_alias=AliasChoices("start_s", "start_seconds"),
    )
    end_s: float = Field(
        ..., gt=0, description="End time in seconds",
        validation_alias=AliasChoices("end_s", "end_seconds"),
    )
    mood: Optional[Mood] = Field(None, description="Mood for this section")
    style: Optional[MusicStyle] = Field(None, description="Style for this section")
    prompt: Optional[str] = Field(None, max_length=500, description="Free-text prompt for this section")
    energy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Energy level 0.0-1.0")

    model_config = {"populate_by_name": True}

    @field_validator("end_s")
    @classmethod
    def end_after_start(cls, v, info):
        if "start_s" in info.data and v <= info.data["start_s"]:
            raise ValueError("end_s must be greater than start_s")
        return v


class OutputConfig(BaseModel):
    """Output format configuration matching the brief's spec."""
    sample_rate: int = Field(default=48000, description="Output sample rate")
    target_lufs: float = Field(default=-16.0, description="Target loudness in LUFS")
    formats: list[str] = Field(default=["wav"], description="Output audio formats")


class MusicJobRequest(BaseModel):
    """
    Request to create a background music track.

    Accepts both brief-style field names (duration_s, tempo_bpm, narration_url)
    and expanded names (duration_seconds, bpm) for flexibility.
    """
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Describe the music you want (mood, style, instruments, tempo)"
    )
    duration_s: float = Field(
        ...,
        gt=0,
        le=300,
        description="Exact duration of the output track in seconds",
        validation_alias=AliasChoices("duration_s", "duration_seconds"),
    )
    style: Optional[MusicStyle] = Field(None, description="Music style category")
    mood: Optional[Mood] = Field(None, description="Mood tag")
    tempo_bpm: Optional[int] = Field(
        None, ge=40, le=300, description="Desired tempo in BPM",
        validation_alias=AliasChoices("tempo_bpm", "bpm"),
    )
    key: Optional[str] = Field(None, description="Musical key (e.g., 'C major', 'A minor')")
    energy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Energy level 0.0-1.0")
    sections: Optional[list[Section]] = Field(
        None,
        description="Optional sections with different moods/styles at different timestamps"
    )
    narration_url: Optional[str] = Field(
        None,
        description="URL to voiceover audio for automatic ducking"
    )
    loopable: bool = Field(False, description="Whether the track should loop seamlessly")
    seed: Optional[int] = Field(None, description="Random seed for reproducible output")
    output: Optional[OutputConfig] = Field(
        None,
        description="Output format configuration"
    )
    callback_url: Optional[str] = Field(None, description="Webhook URL for job completion notification")

    model_config = {"populate_by_name": True}

    # ── Backward-compatible accessors ──
    @property
    def duration_seconds(self) -> float:
        """Alias for duration_s (backward compatibility)."""
        return self.duration_s

    @property
    def bpm(self) -> Optional[int]:
        """Alias for tempo_bpm (backward compatibility)."""
        return self.tempo_bpm

    @property
    def loudness_lufs(self) -> float:
        """Get target loudness from output config or default."""
        if self.output:
            return self.output.target_lufs
        return -16.0

    @property
    def include_stems(self) -> bool:
        """Always include stems (must-have requirement)."""
        return True


# ─── Response Models ───


class StemInfo(BaseModel):
    """Information about a separated audio stem."""
    stem_type: StemType
    url: str
    duration_seconds: float
    loudness_lufs: Optional[float] = None
    gain: float = Field(default=1.0, description="Gain to reconstruct mix from stems")


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
    manifest_url: Optional[str] = Field(None, description="URL to the output manifest JSON")
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


# ─── Portion / Section Remix Models ───

class PortionGain(BaseModel):
    drums: float = Field(default=1.0, ge=0.0, le=2.0)
    bass: float = Field(default=1.0, ge=0.0, le=2.0)
    melody: float = Field(default=1.0, ge=0.0, le=2.0)
    harmony: float = Field(default=1.0, ge=0.0, le=2.0, validation_alias=AliasChoices("harmony", "other"))

    model_config = {"populate_by_name": True}


class PortionConfig(BaseModel):
    name: str = Field(default="Section")
    start_s: float = Field(..., ge=0.0)
    end_s: float = Field(..., gt=0.0)
    gains: PortionGain


class RemixRequest(BaseModel):
    job_id: Optional[str] = None
    upload_id: Optional[str] = None
    portions: list[PortionConfig]


class RemixResponse(BaseModel):
    status: str = "completed"
    custom_mix_url: str
    duration_seconds: float
    loudness_lufs: float
    true_peak_dbtp: float
    portions_count: int


# ─── Natify AI Assistant Models ───

class NatifyAction(BaseModel):
    type: str
    payload: dict = Field(default_factory=dict)


class NatifyMessage(BaseModel):
    message: str
    current_mode: Optional[str] = "create"
    duration_s: Optional[float] = 15.0
    tempo_bpm: Optional[int] = 120


class NatifyResponse(BaseModel):
    reply: str
    action: Optional[NatifyAction] = None
    suggestions: list[str] = Field(default_factory=list)


# ─── AI Voiceover Models ───

class VoiceGender(str, enum.Enum):
    FEMALE = "female"
    MALE = "male"


class VoiceAgeCategory(str, enum.Enum):
    CHILD = "child"
    YOUNG_ADULT = "young_adult"
    MIDDLE_AGED = "middle_aged"
    SENIOR = "senior"


class VoiceGenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Script or speech text to read aloud")
    gender: VoiceGender = Field(default=VoiceGender.FEMALE)
    age_category: VoiceAgeCategory = Field(default=VoiceAgeCategory.YOUNG_ADULT)
    rate: str = Field(default="+0%", description="Speaking speed adjustment, e.g. +10% or -10%")
    pitch: str = Field(default="+0Hz", description="Voice pitch adjustment, e.g. +5Hz or -5Hz")


class VoiceGenerateResponse(BaseModel):
    voice_id: str
    audio_url: str
    text: str
    gender: str
    age_category: str
    voice_name: str
    duration_seconds: float

