"""
Voiceover ducking module.
Advanced level: lower the music 8-15 dB under speech,
come back smoothly in gaps, no pumping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import structlog

logger = structlog.get_logger()


def detect_speech_regions(
    voiceover_path: str | Path,
    threshold_db: float = -40.0,
    min_speech_duration_ms: int = 100,
    min_silence_duration_ms: int = 200,
) -> list[tuple[float, float]]:
    """
    Detect speech regions in a voiceover audio file using energy-based VAD.

    Returns:
        List of (start_seconds, end_seconds) tuples for speech regions.
    """
    audio, sr = sf.read(str(voiceover_path), dtype="float32")

    # Frame-based energy detection
    frame_size = int(sr * 0.02)  # 20ms frames
    hop_size = int(sr * 0.01)    # 10ms hop

    threshold_linear = 10 ** (threshold_db / 20)

    speech_frames = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i : i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        speech_frames.append(rms > threshold_linear)

    # Smooth: merge short silence gaps
    min_silence_frames = int(min_silence_duration_ms / 10)
    min_speech_frames = int(min_speech_duration_ms / 10)

    smoothed = list(speech_frames)

    # Fill short silence gaps
    i = 0
    while i < len(smoothed):
        if not smoothed[i]:
            gap_start = i
            while i < len(smoothed) and not smoothed[i]:
                i += 1
            gap_len = i - gap_start
            if gap_len < min_silence_frames:
                for j in range(gap_start, min(i, len(smoothed))):
                    smoothed[j] = True
        else:
            i += 1

    # Remove short speech segments
    i = 0
    while i < len(smoothed):
        if smoothed[i]:
            seg_start = i
            while i < len(smoothed) and smoothed[i]:
                i += 1
            seg_len = i - seg_start
            if seg_len < min_speech_frames:
                for j in range(seg_start, min(i, len(smoothed))):
                    smoothed[j] = False
        else:
            i += 1

    # Convert frames to time regions
    regions = []
    in_speech = False
    speech_start = 0.0

    for idx, is_speech in enumerate(smoothed):
        time_s = idx * 0.01  # 10ms hop
        if is_speech and not in_speech:
            speech_start = time_s
            in_speech = True
        elif not is_speech and in_speech:
            regions.append((speech_start, time_s))
            in_speech = False

    if in_speech:
        regions.append((speech_start, len(audio) / sr))

    return regions


def apply_ducking(
    music_audio: np.ndarray,
    sample_rate: int,
    speech_regions: list[tuple[float, float]],
    duck_level_db: float = -12.0,
    fade_ms: int = 200,
) -> tuple[np.ndarray, list[dict]]:
    """
    Apply volume ducking to music under speech regions.

    Args:
        music_audio: Music audio array.
        sample_rate: Sample rate in Hz.
        speech_regions: List of (start_s, end_s) speech timestamps.
        duck_level_db: How much to reduce music under speech (dB).
        fade_ms: Fade in/out duration at duck boundaries (ms).

    Returns:
        Tuple of (ducked_audio, volume_curve_data)
    """
    ducked = music_audio.copy()
    duck_linear = 10 ** (duck_level_db / 20)
    fade_samples = int(fade_ms * sample_rate / 1000)

    volume_curve = np.ones(len(ducked), dtype=np.float32)

    for start_s, end_s in speech_regions:
        start_sample = int(start_s * sample_rate)
        end_sample = int(end_s * sample_rate)

        # Clamp to valid range
        start_sample = max(0, min(start_sample, len(ducked)))
        end_sample = max(0, min(end_sample, len(ducked)))

        if start_sample >= end_sample:
            continue

        # Set ducked volume for the speech region
        volume_curve[start_sample:end_sample] = duck_linear

    # Apply smooth fades at transitions
    smoothed_curve = _smooth_volume_curve(volume_curve, fade_samples)

    ducked *= smoothed_curve

    # Build volume curve data for manifest
    curve_data = []
    for start_s, end_s in speech_regions:
        curve_data.append({
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "gain_db": duck_level_db,
            "fade_ms": fade_ms,
        })

    return ducked, curve_data


def _smooth_volume_curve(
    curve: np.ndarray,
    fade_samples: int,
) -> np.ndarray:
    """Apply smooth transitions to a volume curve to avoid pumping."""
    if fade_samples <= 0:
        return curve

    smoothed = curve.copy()

    # Find transitions
    diff = np.diff(curve)
    transitions = np.where(np.abs(diff) > 0.01)[0]

    for t in transitions:
        # Apply cosine fade around transition
        half_fade = fade_samples // 2
        start = max(0, t - half_fade)
        end = min(len(smoothed), t + half_fade + 1)
        length = end - start

        if length > 1:
            # Cosine interpolation
            before_val = curve[max(0, start - 1)]
            after_val = curve[min(len(curve) - 1, end)]
            cos_curve = np.linspace(0, np.pi, length)
            interp = before_val + (after_val - before_val) * (1 - np.cos(cos_curve)) / 2
            smoothed[start:end] = interp

    return smoothed


def duck_music_file(
    music_path: str | Path,
    voiceover_path: str | Path,
    output_path: str | Path,
    duck_level_db: float = -12.0,
    fade_ms: int = 200,
) -> dict:
    """
    Apply ducking to a music file based on speech in a voiceover file.

    Returns:
        dict with speech regions, ducking parameters, and volume curve.
    """
    music_audio, sr = sf.read(str(music_path), dtype="float32")

    # Detect speech regions
    speech_regions = detect_speech_regions(voiceover_path)

    logger.info(
        "ducking_applied",
        num_speech_regions=len(speech_regions),
        duck_level_db=duck_level_db,
        fade_ms=fade_ms,
    )

    # Apply ducking
    ducked_audio, volume_curve = apply_ducking(
        music_audio, sr, speech_regions, duck_level_db, fade_ms
    )

    # Write output
    sf.write(str(output_path), ducked_audio, sr)

    return {
        "speech_regions": speech_regions,
        "duck_level_db": duck_level_db,
        "fade_ms": fade_ms,
        "volume_curve": volume_curve,
        "num_ducked_regions": len(speech_regions),
    }
