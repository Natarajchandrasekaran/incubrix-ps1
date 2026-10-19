"""
Configuration management for the AI Background Music Engine.
All settings come from environment variables — no hard-coded paths or keys.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    # API
    api_host: str = Field(default="0.0.0.0", alias="INCUBRIX_API_HOST")
    api_port: int = Field(default=8000, alias="INCUBRIX_API_PORT")

    # Redis (for Celery)
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # S3-compatible storage (MinIO)
    s3_endpoint: str = Field(default="http://localhost:9000", alias="S3_ENDPOINT")
    s3_access_key: str = Field(default="minioadmin", alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="minioadmin", alias="S3_SECRET_KEY")
    s3_bucket: str = Field(default="incubrix-music", alias="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", alias="S3_REGION")

    # MusicGen model
    musicgen_model: str = Field(
        default="facebook/musicgen-small",
        alias="MUSICGEN_MODEL",
        description="HuggingFace model ID for MusicGen"
    )

    # Audio defaults
    default_loudness_lufs: float = Field(default=-16.0, alias="DEFAULT_LOUDNESS_LUFS")
    default_sample_rate: int = Field(default=48000, alias="DEFAULT_SAMPLE_RATE")
    max_duration_seconds: float = Field(default=300.0, alias="MAX_DURATION_SECONDS")

    # Adapter selection
    music_adapter: str = Field(
        default="musicgen",
        alias="MUSIC_ADAPTER",
        description="Which music adapter to use: musicgen, suno, mock"
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Demo / rate limiting
    rate_limit_per_minute: int = Field(default=10, alias="RATE_LIMIT_PER_MINUTE")



@lru_cache()
def get_settings() -> Settings:
    """Cached singleton for application settings."""
    return Settings()
