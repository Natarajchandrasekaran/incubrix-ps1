"""
Mock music adapter for testing without API keys.
Generates sine wave tones that match the requested duration exactly.
Integration Standard: tests run with no API keys.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from app.adapters.base import GeneratedAudio, MusicAdapter


class MockMusicAdapter(MusicAdapter):
    """
    Mock adapter that generates multi-layered sine waves.

    Used for testing the full pipeline without loading MusicGen or calling APIs.
    Produces deterministic output when seed is provided.
    """

    def __init__(self, output_dir: Path | str = "/tmp/incubrix_mock"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        """Generate a multi-tone sine wave that matches the exact duration."""
        start_time = time.time()

        if seed is not None:
            np.random.seed(seed)

        # Number of samples for exact duration
        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples, endpoint=False)

        # Generate multiple tones to simulate different instruments
        # Base frequency varies with prompt hash for variety
        prompt_hash = hash(prompt) % 200 + 200  # 200-400 Hz range
        frequencies = [
            prompt_hash,           # "bass"
            prompt_hash * 2,       # "harmony"
            prompt_hash * 3,       # "melody"
            prompt_hash * 0.5,     # "sub"
        ]

        # Mix tones with different amplitudes
        audio = np.zeros(num_samples, dtype=np.float32)
        for i, freq in enumerate(frequencies):
            amplitude = 0.15 / (i + 1)  # Decreasing amplitude
            audio += (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        # Apply envelope: fade in first 5% and fade out last 10%
        fade_in_samples = int(num_samples * 0.05)
        fade_out_samples = int(num_samples * 0.10)

        if fade_in_samples > 0:
            fade_in = np.linspace(0, 1, fade_in_samples, dtype=np.float32)
            audio[:fade_in_samples] *= fade_in

        if fade_out_samples > 0:
            fade_out = np.linspace(1, 0, fade_out_samples, dtype=np.float32)
            audio[-fade_out_samples:] *= fade_out

        # Apply section changes if provided
        if sections:
            for i, section in enumerate(sections):
                start_s = section.get("start_s", section.get("start_seconds", 0))
                start_sample = int(start_s * sample_rate)
                if start_sample < num_samples:
                    # Shift frequency slightly at section boundary
                    end_sample = num_samples
                    if i + 1 < len(sections):
                        next_start = sections[i + 1].get(
                            "start_s", sections[i + 1].get("start_seconds", duration_seconds)
                        )
                        end_sample = min(int(next_start * sample_rate), num_samples)

                    section_t = t[start_sample:end_sample]
                    shift = (i + 1) * 50  # Frequency shift per section
                    section_tone = 0.1 * np.sin(
                        2 * np.pi * (prompt_hash + shift) * section_t
                    ).astype(np.float32)
                    audio[start_sample:end_sample] += section_tone

        # Normalize to prevent clipping
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.8

        # Write to file
        output_path = self.output_dir / f"mock_{int(time.time() * 1000)}.wav"
        sf.write(str(output_path), audio, sample_rate)

        generation_time = time.time() - start_time

        return GeneratedAudio(
            audio_path=output_path,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            model_name="mock-sine-generator",
            model_version="1.0.0",
            seed=seed,
            cost_estimate_usd=0.0,
            generation_time_seconds=round(generation_time, 3),
            warnings=["This is mock audio (sine waves) for testing only."],
        )

    def is_available(self) -> bool:
        """Mock adapter is always available."""
        return True

    def get_info(self) -> dict:
        return {
            "name": "MockMusicAdapter",
            "version": "1.0.0",
            "description": "Generates sine wave tones for testing. No ML model required.",
            "license": "MIT",
            "cost_per_minute": 0.0,
        }
