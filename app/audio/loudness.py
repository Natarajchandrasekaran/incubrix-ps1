"""
Loudness normalization module.
Ensures output audio meets broadcast standards:
- Target LUFS (default -16)
- True peak at or below -1 dBTP
- No clipping
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def measure_loudness(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Measure the integrated loudness and true peak of audio.

    Returns:
        dict with 'loudness_lufs' and 'true_peak_dbtp'
    """
    meter = pyln.Meter(sample_rate)

    # Ensure audio is at least 0.4s for pyloudnorm
    min_samples = int(sample_rate * 0.4)
    if len(audio) < min_samples:
        padded = np.zeros(min_samples, dtype=audio.dtype)
        padded[: len(audio)] = audio
        audio = padded

    # Handle mono vs stereo
    if audio.ndim == 1:
        loudness = meter.integrated_loudness(audio)
    else:
        loudness = meter.integrated_loudness(audio)

    # True peak measurement
    true_peak = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)

    return {
        "loudness_lufs": round(loudness, 2) if np.isfinite(loudness) else -70.0,
        "true_peak_dbtp": round(true_peak, 2),
    }


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -16.0,
    max_true_peak_dbtp: float = -1.0,
) -> np.ndarray:
    """
    Normalize audio to target loudness with true peak limiting.

    Args:
        audio: Input audio array (mono or stereo).
        sample_rate: Sample rate in Hz.
        target_lufs: Target integrated loudness in LUFS.
        max_true_peak_dbtp: Maximum true peak in dBTP.

    Returns:
        Normalized audio array.
    """
    meter = pyln.Meter(sample_rate)

    # Ensure minimum length for measurement
    min_samples = int(sample_rate * 0.4)
    if len(audio) < min_samples:
        padded = np.zeros(min_samples, dtype=audio.dtype)
        padded[: len(audio)] = audio
        audio_for_measurement = padded
    else:
        audio_for_measurement = audio

    # Measure current loudness
    current_loudness = meter.integrated_loudness(audio_for_measurement)

    if not np.isfinite(current_loudness) or current_loudness < -70:
        # Audio is effectively silent
        return audio

    # Normalize to target LUFS
    normalized = pyln.normalize.loudness(audio, current_loudness, target_lufs)

    # Apply true peak limiter
    max_true_peak_linear = 10 ** (max_true_peak_dbtp / 20)
    peak = np.max(np.abs(normalized))

    if peak > max_true_peak_linear:
        normalized = normalized * (max_true_peak_linear / peak)

    return normalized.astype(np.float32)


def normalize_file(
    input_path: str | Path,
    output_path: str | Path,
    target_lufs: float = -16.0,
    max_true_peak_dbtp: float = -1.0,
) -> dict:
    """
    Read a WAV file, normalize it, and write the result.

    Returns:
        dict with before/after loudness measurements.
    """
    audio, sr = sf.read(str(input_path), dtype="float32")

    before = measure_loudness(audio, sr)
    normalized = normalize_loudness(audio, sr, target_lufs, max_true_peak_dbtp)
    after = measure_loudness(normalized, sr)

    sf.write(str(output_path), normalized, sr)

    return {
        "before": before,
        "after": after,
        "target_lufs": target_lufs,
        "max_true_peak_dbtp": max_true_peak_dbtp,
    }
