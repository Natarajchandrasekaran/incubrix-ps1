"""
Audio stem separation using Demucs.
Separates a mixed audio track into individual instrument stems.

Must-have #2: At least 3 stems, each a different instrument part.
The mix must equal the sum of stems using the gains in the manifest.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import structlog

logger = structlog.get_logger()


@dataclass
class StemResult:
    """Result of stem separation."""
    stem_name: str
    audio: np.ndarray
    path: Optional[Path] = None
    gain: float = 1.0  # Gain applied when mixing back
    bleed_estimate_db: Optional[float] = None


@dataclass
class SeparationResult:
    """Complete result of audio separation into stems."""
    stems: list[StemResult]
    model_name: str
    model_version: str
    processing_time_seconds: float
    sample_rate: int
    duration_seconds: float


def separate_stems(
    audio_path: str | Path,
    output_dir: str | Path,
    model_name: str = "htdemucs",
    device: str = "cpu",
) -> SeparationResult:
    """
    Separate an audio file into instrument stems using Demucs.

    Args:
        audio_path: Path to input audio file.
        output_dir: Directory to save separated stems.
        model_name: Demucs model to use (htdemucs, htdemucs_ft, etc.)
        device: Device to run on ('cpu' or 'cuda').

    Returns:
        SeparationResult with paths to separated stems.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        # Load model
        model = get_model(model_name)
        model.to(device)

        # Load audio
        wav, sr = torchaudio.load(str(audio_path))

        # Demucs expects specific sample rate
        if sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            sr = model.samplerate

        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        # Apply model
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model(
                model,
                wav[None],
                device=device,
                progress=False,
            )[0]

        # De-normalize
        sources = sources * ref.std() + ref.mean()

        # Save stems
        stem_names = model.sources  # e.g., ['drums', 'bass', 'other', 'vocals']
        stems = []

        for i, name in enumerate(stem_names):
            stem_audio = sources[i].cpu().numpy()

            # Convert to mono if needed for consistency
            if stem_audio.ndim > 1:
                stem_mono = stem_audio.mean(axis=0)
            else:
                stem_mono = stem_audio

            stem_path = output_dir / f"{name}.wav"
            sf.write(str(stem_path), stem_mono, sr, subtype="FLOAT")

            stems.append(StemResult(
                stem_name=name,
                audio=stem_mono,
                path=stem_path,
                gain=1.0,
            ))

        processing_time = time.time() - start_time
        duration = len(stems[0].audio) / sr if stems else 0

        logger.info(
            "stem_separation_complete",
            model=model_name,
            num_stems=len(stems),
            stem_names=[s.stem_name for s in stems],
            duration_s=round(duration, 2),
            processing_time_s=round(processing_time, 2),
        )

        return SeparationResult(
            stems=stems,
            model_name=model_name,
            model_version="4.1.0",
            processing_time_seconds=round(processing_time, 3),
            sample_rate=sr,
            duration_seconds=round(duration, 3),
        )

    except ImportError:
        logger.warning("demucs_not_available_using_mock_separation")
        return _mock_separate(audio_path, output_dir, start_time)


def _mock_separate(
    audio_path: Path,
    output_dir: Path,
    start_time: float,
) -> SeparationResult:
    """
    Fallback mock separation using frequency band splitting.
    Used when Demucs is not installed (e.g., in lightweight test environments).

    NOTE: This produces frequency bands, not true stems. It's only for
    pipeline testing. Real submission uses Demucs.
    """
    from scipy import signal

    audio, sr = sf.read(str(audio_path), dtype="float32")
    duration = len(audio) / sr

    # Simple frequency band separation (for testing pipeline only)
    stems = []
    band_configs = [
        ("drums", 20, 250),      # Low frequencies → drums
        ("bass", 60, 400),       # Bass range
        ("other", 400, 4000),    # Midrange → harmony/pads
        ("melody", 2000, 16000), # High frequencies → melody
    ]

    for name, low_hz, high_hz in band_configs:
        # Bandpass filter
        nyquist = sr / 2
        low = max(low_hz / nyquist, 0.001)
        high = min(high_hz / nyquist, 0.999)

        if low >= high:
            stem_audio = np.zeros_like(audio)
        else:
            b, a = signal.butter(4, [low, high], btype="band")
            stem_audio = signal.filtfilt(b, a, audio).astype(np.float32)

        stem_path = output_dir / f"{name}.wav"
        sf.write(str(stem_path), stem_audio, sr, subtype="FLOAT")

        stems.append(StemResult(
            stem_name=name,
            audio=stem_audio,
            path=stem_path,
            gain=1.0,
        ))

    processing_time = time.time() - start_time

    return SeparationResult(
        stems=stems,
        model_name="mock-frequency-band-split",
        model_version="1.0.0",
        processing_time_seconds=round(processing_time, 3),
        sample_rate=sr,
        duration_seconds=round(duration, 3),
    )


def verify_stems_sum_to_mix(
    mix_path: str | Path,
    stem_paths: list[str | Path],
    gains: Optional[list[float]] = None,
    tolerance_db: float = -40.0,
) -> dict:
    """
    Verify that the sum of stems equals the mix.

    Must-have #2: The mix equals the sum of the stems using the gains
    in the manifest.

    Returns:
        dict with 'passes', 'difference_db', and details.
    """
    mix_audio, sr = sf.read(str(mix_path), dtype="float32")

    if gains is None:
        gains = [1.0] * len(stem_paths)

    # Sum stems
    reconstructed = np.zeros_like(mix_audio)
    for path, gain in zip(stem_paths, gains):
        stem_audio, stem_sr = sf.read(str(path), dtype="float32")
        assert stem_sr == sr, f"Sample rate mismatch: {stem_sr} vs {sr}"

        # Ensure same length
        min_len = min(len(reconstructed), len(stem_audio))
        reconstructed[:min_len] += gain * stem_audio[:min_len]

    # Compute difference
    min_len = min(len(mix_audio), len(reconstructed))
    diff = mix_audio[:min_len] - reconstructed[:min_len]
    diff_rms = np.sqrt(np.mean(diff ** 2) + 1e-10)
    mix_rms = np.sqrt(np.mean(mix_audio[:min_len] ** 2) + 1e-10)

    if mix_rms > 0:
        diff_db = 20 * np.log10(diff_rms / mix_rms + 1e-10)
    else:
        diff_db = -100.0

    passes = bool(diff_db <= tolerance_db)

    return {
        "passes": passes,
        "difference_db": round(float(diff_db), 2),
        "tolerance_db": tolerance_db,
        "mix_rms": round(float(mix_rms), 6),
        "diff_rms": round(float(diff_rms), 6),
    }
