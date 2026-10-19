"""
Abstract base class for music generation adapters.
Integration Standard point 3: Swappable providers behind a small adapter.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GeneratedAudio:
    """Result from a music generation adapter."""
    audio_path: Path  # Path to the generated audio file
    sample_rate: int
    duration_seconds: float
    model_name: str
    model_version: str
    seed: Optional[int] = None
    cost_estimate_usd: Optional[float] = None
    generation_time_seconds: Optional[float] = None
    warnings: list[str] | None = None


class MusicAdapter(abc.ABC):
    """
    Abstract interface for music generation backends.

    Every music provider (MusicGen, Suno, etc.) must implement this interface.
    This allows swapping providers via configuration without changing application code.
    """

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        duration_seconds: float,
        *,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
        seed: Optional[int] = None,
        sample_rate: int = 48000,
        sections: Optional[list[dict]] = None,
    ) -> GeneratedAudio:
        """
        Generate instrumental music from a text prompt.

        Args:
            prompt: Text description of desired music style/mood.
            duration_seconds: Exact duration of output audio.
            bpm: Optional desired tempo in beats per minute.
            key: Optional musical key (e.g., "C major", "A minor").
            seed: Optional random seed for reproducibility.
            sample_rate: Output sample rate in Hz.
            sections: Optional list of section dicts with start_s and mood.

        Returns:
            GeneratedAudio with path to the generated WAV file.
        """
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this adapter's backend is available."""
        ...

    @abc.abstractmethod
    def get_info(self) -> dict:
        """Return adapter metadata: name, version, license, etc."""
        ...
