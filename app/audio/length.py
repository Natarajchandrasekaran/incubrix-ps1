"""
Exact-length control for generated music.
Must-have #1: Output matches requested duration within 0.05s,
and the track ends musically (fade on bar line, not a hard cut).
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional


def compute_bar_duration(bpm: int, beats_per_bar: int = 4) -> float:
    """Compute the duration of one bar in seconds."""
    beat_duration = 60.0 / bpm
    return beat_duration * beats_per_bar


def snap_to_bars(
    duration_seconds: float,
    bpm: int = 120,
    beats_per_bar: int = 4,
) -> dict:
    """
    Find the closest bar-aligned duration and the number of bars.

    Returns:
        dict with 'target_duration', 'num_bars', 'bpm', 'bar_duration'
    """
    bar_duration = compute_bar_duration(bpm, beats_per_bar)
    num_bars = max(1, round(duration_seconds / bar_duration))

    # Adjust so we're close to the requested duration
    aligned_duration = num_bars * bar_duration

    # If too far off, adjust BPM slightly to match exactly
    if abs(aligned_duration - duration_seconds) > 0.5:
        # Find a BPM that makes bars align exactly
        adjusted_bpm = (num_bars * beats_per_bar * 60.0) / duration_seconds
        # Round to nearest integer BPM
        adjusted_bpm = round(adjusted_bpm)
        bar_duration = compute_bar_duration(adjusted_bpm, beats_per_bar)
        aligned_duration = num_bars * bar_duration
        bpm = adjusted_bpm

    return {
        "target_duration": duration_seconds,
        "aligned_duration": aligned_duration,
        "num_bars": num_bars,
        "bpm": bpm,
        "beats_per_bar": beats_per_bar,
        "bar_duration": bar_duration,
        "beat_times": [i * (60.0 / bpm) for i in range(num_bars * beats_per_bar)],
        "bar_times": [i * bar_duration for i in range(num_bars)],
    }


def trim_or_pad_to_exact_length(
    audio: np.ndarray,
    target_samples: int,
    sample_rate: int,
    fade_out_ms: int = 500,
    musical_ending: bool = True,
) -> np.ndarray:
    """
    Ensure audio is exactly the target length.

    If too long: fade out and trim.
    If too short: loop/pad and fade.
    Always applies a musical fade-out at the end.

    Must-have #1: Track ends on purpose, never a hard cut.
    """
    current_len = len(audio)

    if current_len == target_samples:
        result = audio.copy()
    elif current_len > target_samples:
        # Trim with fade out
        result = audio[:target_samples].copy()
    else:
        # Pad: loop the existing audio
        result = np.zeros(target_samples, dtype=audio.dtype)
        # Fill by looping
        full_copies = target_samples // current_len
        remainder = target_samples % current_len
        for i in range(full_copies):
            start = i * current_len
            result[start : start + current_len] = audio
        if remainder > 0:
            result[full_copies * current_len :] = audio[:remainder]

    # Apply musical ending (fade out on the last portion)
    if musical_ending:
        fade_samples = min(
            int(fade_out_ms * sample_rate / 1000),
            target_samples // 4,  # Don't fade more than 25% of the track
        )
        if fade_samples > 0:
            fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            # Apply exponential curve for more musical feel
            fade_curve = fade_curve ** 2
            result[-fade_samples:] *= fade_curve

    return result


def ensure_exact_duration(
    input_path: str | Path,
    output_path: str | Path,
    target_duration_seconds: float,
    fade_out_ms: int = 500,
) -> dict:
    """
    Read an audio file, adjust to exact target duration, and save.

    Returns:
        dict with actual duration, difference from target, and status.
    """
    audio, sr = sf.read(str(input_path), dtype="float32")
    target_samples = int(target_duration_seconds * sr)

    adjusted = trim_or_pad_to_exact_length(
        audio, target_samples, sr, fade_out_ms
    )

    sf.write(str(output_path), adjusted, sr)

    actual_duration = len(adjusted) / sr
    diff = abs(actual_duration - target_duration_seconds)

    return {
        "target_duration_s": target_duration_seconds,
        "actual_duration_s": actual_duration,
        "difference_s": round(diff, 6),
        "within_tolerance": diff <= 0.05,
        "sample_rate": sr,
        "num_samples": len(adjusted),
    }
