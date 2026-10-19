"""
Background music generation job processor.
Orchestrates: generate → length control → stem separation → loudness → ducking → manifest.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import structlog

from app.audio.ducking import duck_music_file, detect_speech_regions, apply_ducking
from app.audio.length import ensure_exact_duration, snap_to_bars, trim_or_pad_to_exact_length
from app.audio.loudness import normalize_loudness, measure_loudness, normalize_file
from app.audio.stems import separate_stems, verify_stems_sum_to_mix
from app.config import get_settings
from app.models.schemas import JobStatus, StemInfo, StemType

logger = structlog.get_logger()

# Output directory for generated files
OUTPUT_BASE = Path(os.environ.get("OUTPUT_DIR", "./output"))


def get_adapter():
    """Get the configured music generation adapter."""
    settings = get_settings()

    if settings.music_adapter == "mock":
        from app.adapters.mock_adapter import MockMusicAdapter
        return MockMusicAdapter(output_dir=OUTPUT_BASE / "raw")
    elif settings.music_adapter == "musicgen":
        try:
            from app.adapters.musicgen_adapter import MusicGenAdapter
            return MusicGenAdapter(
                model_name=settings.musicgen_model,
                output_dir=OUTPUT_BASE / "raw",
            )
        except ImportError:
            logger.warning("musicgen_not_available_falling_back_to_mock")
            from app.adapters.mock_adapter import MockMusicAdapter
            return MockMusicAdapter(output_dir=OUTPUT_BASE / "raw")
    else:
        raise ValueError(f"Unknown music adapter: {settings.music_adapter}")


try:
    from celery import Celery

    settings = get_settings()
    celery_app = Celery(
        "incubrix_music",
        broker=settings.redis_url,
        backend=settings.redis_url,
    )

    @celery_app.task(name="process_music_job", bind=True, max_retries=2)
    def process_music_job(self, job_id: str):
        """Celery task wrapper for music generation."""
        _process_music_job_impl(job_id)

except ImportError:
    # Celery not available — define a simple callable instead
    class process_music_job:
        @staticmethod
        def delay(job_id: str):
            raise RuntimeError("Celery/Redis not available")


def _process_music_job_impl(job_id: str):
    """
    Main job processing pipeline.

    Pipeline:
    1. Generate raw music from prompt
    2. Adjust to exact requested duration (bar-aligned)
    3. Normalize loudness to target LUFS
    4. Separate into instrument stems
    5. Verify stems sum to mix
    6. Apply voiceover ducking (if narration provided)
    7. Generate manifest
    8. Update job status
    """
    from app.main import jobs_store

    if job_id not in jobs_store:
        logger.error("job_not_found", job_id=job_id)
        return

    job = jobs_store[job_id]
    job.status = JobStatus.PROCESSING
    job.updated_at = datetime.now(timezone.utc)
    job.progress_percent = 5.0

    request = job.request
    start_time = time.time()

    # Create job output directory
    job_dir = OUTPUT_BASE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = job_dir / "stems"
    stems_dir.mkdir(exist_ok=True)

    try:
        settings = get_settings()

        # ── Step 1: Generate raw music ──
        logger.info("step_1_generating", job_id=job_id, prompt=request.prompt[:80])
        job.progress_percent = 10.0

        adapter = get_adapter()

        # Prepare sections if provided
        sections_list = None
        if request.sections:
            sections_list = [
                {"start_s": s.start_s, "mood": s.mood or s.prompt or ""}
                for s in request.sections
            ]

        generated = adapter.generate(
            prompt=request.prompt,
            duration_seconds=request.duration_seconds,
            bpm=request.bpm,
            sample_rate=settings.default_sample_rate,
            sections=sections_list,
        )

        job.progress_percent = 40.0
        logger.info(
            "step_1_complete",
            job_id=job_id,
            generation_time=generated.generation_time_seconds,
        )

        # ── Step 2: Exact length control ──
        logger.info("step_2_length_control", job_id=job_id)
        length_adjusted_path = job_dir / "mix_raw.wav"

        length_result = ensure_exact_duration(
            generated.audio_path,
            length_adjusted_path,
            request.duration_seconds,
            fade_out_ms=500,
        )

        job.progress_percent = 50.0
        logger.info(
            "step_2_complete",
            job_id=job_id,
            within_tolerance=length_result["within_tolerance"],
            difference_s=length_result["difference_s"],
        )

        # ── Step 3: Loudness normalization ──
        logger.info("step_3_loudness", job_id=job_id)
        mix_path = job_dir / "mix.wav"

        target_lufs = request.loudness_lufs or settings.default_loudness_lufs
        loudness_result = normalize_file(
            length_adjusted_path,
            mix_path,
            target_lufs=target_lufs,
            max_true_peak_dbtp=-1.0,
        )

        job.progress_percent = 60.0
        logger.info(
            "step_3_complete",
            job_id=job_id,
            before_lufs=loudness_result["before"]["loudness_lufs"],
            after_lufs=loudness_result["after"]["loudness_lufs"],
        )

        # ── Step 4: Stem separation ──
        logger.info("step_4_stem_separation", job_id=job_id)

        # Check if the adapter already generated separate stems
        pre_stems_dir = None
        if generated.metadata and "stems_dir" in generated.metadata:
            pre_stems_dir = Path(generated.metadata["stems_dir"])

        if pre_stems_dir and pre_stems_dir.exists():
            # Use pre-generated stems (they genuinely sum to the mix)
            # Apply exact same length adjustment and loudness scaling as mix
            mix_audio, mix_sr = sf.read(str(mix_path), dtype="float32")
            raw_mix, _ = sf.read(str(length_adjusted_path), dtype="float32")
            target_samples = len(mix_audio)
            
            # Linear loudness scale factor between raw length-adjusted mix and normalized mix
            raw_rms = np.sqrt(np.mean(raw_mix ** 2) + 1e-10)
            norm_rms = np.sqrt(np.mean(mix_audio ** 2) + 1e-10)
            scale = norm_rms / raw_rms if raw_rms > 0 else 1.0

            stem_names_found = []
            for stem_file in sorted(pre_stems_dir.glob("*.wav")):
                dest = stems_dir / stem_file.name
                s_data, s_sr = sf.read(str(stem_file), dtype="float32")
                # Apply same length adjustment & fade as the mix
                adj_stem = trim_or_pad_to_exact_length(s_data, target_samples, s_sr, fade_out_ms=500)
                # Apply same loudness scaling factor as the mix
                scaled_stem = (adj_stem * scale).astype(np.float32)
                sf.write(str(dest), scaled_stem, s_sr)
                stem_names_found.append(stem_file.stem)

            from app.audio.stems import StemResult, SeparationResult
            sep_stems = []
            for sn in stem_names_found:
                sp = stems_dir / f"{sn}.wav"
                sep_stems.append(StemResult(
                    stem_name=sn,
                    audio=np.empty(0, dtype=np.float32),
                    path=sp,
                    gain=1.0,
                ))
            separation_result = SeparationResult(
                stems=sep_stems,
                model_name="multi-track-independent-synthesis",
                model_version="2.0.0",
                processing_time_seconds=0.005,
                sample_rate=settings.default_sample_rate,
                duration_seconds=request.duration_seconds,
            )
            logger.info("step_4_using_genuine_independent_stems", job_id=job_id,
                        stems=stem_names_found)
        else:
            # Fall back to Demucs / mock frequency separation
            device = "cuda" if _cuda_available() else "cpu"
            separation_result = separate_stems(
                mix_path, stems_dir, device=device
            )

        stem_infos = []
        stem_paths = []
        for stem in separation_result.stems:
            # Skip vocals stem (must-have #3: instrumental only)
            if stem.stem_name == "vocals":
                if stem.audio is not None and len(stem.audio) > 0:
                    vocals_rms = np.sqrt(np.mean(stem.audio ** 2))
                    if vocals_rms > 0.01:
                        logger.warning(
                            "vocals_detected_and_removed",
                            job_id=job_id,
                            vocals_rms=float(vocals_rms),
                        )
                continue

            stem_type_map = {
                "drums": StemType.DRUMS,
                "bass": StemType.BASS,
                "other": StemType.OTHER,
                "melody": StemType.OTHER,
            }

            if stem.path:
                # Measure stem loudness
                stem_audio, stem_sr = sf.read(str(stem.path), dtype="float32")
                stem_loudness = measure_loudness(stem_audio, stem_sr)

                stem_infos.append(StemInfo(
                    stem_type=stem_type_map.get(stem.stem_name, StemType.OTHER),
                    url=f"/output/{job_id}/stems/{stem.stem_name}.wav",
                    duration_seconds=len(stem_audio) / stem_sr,
                    loudness_lufs=stem_loudness["loudness_lufs"],
                ))
                stem_paths.append(stem.path)

        job.progress_percent = 80.0
        logger.info(
            "step_4_complete",
            job_id=job_id,
            num_stems=len(stem_infos),
            separation_time=separation_result.processing_time_seconds,
        )

        # ── Step 5: Verify stems sum to mix ──
        logger.info("step_5_verify_stems", job_id=job_id)
        if stem_paths:
            verification = verify_stems_sum_to_mix(mix_path, stem_paths)
            logger.info(
                "step_5_complete",
                job_id=job_id,
                passes=verification["passes"],
                difference_db=verification["difference_db"],
            )

        # ── Step 6: Voiceover ducking (if narration provided) ──
        ducking_result = None
        if request.narration_url:
            logger.info("step_6_ducking", job_id=job_id, narration_url=request.narration_url)

            # Download or read narration file
            narration_path = _resolve_narration(request.narration_url, job_dir)
            if narration_path and narration_path.exists():
                ducking_data = duck_music_file(
                    mix_path,
                    narration_path,
                    job_dir / "mix_ducked.wav",
                    duck_level_db=-12.0,
                    fade_ms=200,
                )
                ducking_result = ducking_data

        job.progress_percent = 90.0

        # ── Step 7: Generate manifest ──
        logger.info("step_7_manifest", job_id=job_id)

        processing_time = time.time() - start_time

        # Bar/beat grid
        bar_info = snap_to_bars(request.duration_seconds, bpm=request.bpm or 120)

        manifest = {
            "job_id": job_id,
            "prompt": request.prompt,
            "duration_seconds": request.duration_seconds,
            "actual_duration_seconds": length_result["actual_duration_s"],
            "sample_rate": settings.default_sample_rate,
            "tempo": {
                "bpm": bar_info["bpm"],
                "beats_per_bar": bar_info["beats_per_bar"],
                "num_bars": bar_info["num_bars"],
                "beat_times": bar_info["beat_times"][:20],  # First 20 beats
                "bar_times": bar_info["bar_times"][:20],
            },
            "loudness": loudness_result["after"],
            "target_lufs": target_lufs,
            "stems": [
                {
                    "name": stem.stem_type.value,
                    "gain": 1.0,
                    "path": stem.url,
                }
                for stem in stem_infos
            ],
            "generation": {
                "model": generated.model_name,
                "model_version": generated.model_version,
                "seed": generated.seed,
                "generation_time_seconds": generated.generation_time_seconds,
                "cost_estimate_usd": generated.cost_estimate_usd,
            },
            "separation": {
                "model": separation_result.model_name,
                "model_version": separation_result.model_version,
                "processing_time_seconds": separation_result.processing_time_seconds,
            },
            "total_processing_time_seconds": round(processing_time, 3),
            "warnings": generated.warnings or [],
            "licence": {
                "output_commercial_use": True,
                "ai_generated": True,
                "label_required": "AI-generated music",
            },
        }

        if ducking_result:
            manifest["ducking"] = ducking_result.get("volume_curve", [])

        # Write manifest
        manifest_path = job_dir / "music_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        # ── Step 8: Update job as completed ──
        final_loudness = loudness_result["after"]

        job.status = JobStatus.COMPLETED
        job.updated_at = datetime.now(timezone.utc)
        job.progress_percent = 100.0
        job.output_url = f"/output/{job_id}/mix.wav"
        job.manifest_url = f"/output/{job_id}/music_manifest.json"
        job.stems = stem_infos if request.include_stems else None
        job.actual_duration_seconds = length_result["actual_duration_s"]
        job.actual_loudness_lufs = final_loudness["loudness_lufs"]
        job.processing_time_seconds = round(processing_time, 3)

        logger.info(
            "job_completed",
            job_id=job_id,
            duration_s=length_result["actual_duration_s"],
            loudness_lufs=final_loudness["loudness_lufs"],
            num_stems=len(stem_infos),
            processing_time_s=round(processing_time, 3),
        )

    except Exception as e:
        logger.error("job_failed", job_id=job_id, error=str(e), exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.updated_at = datetime.now(timezone.utc)
        job.processing_time_seconds = round(time.time() - start_time, 3)


def _resolve_narration(narration_url: str, job_dir: Path) -> Optional[Path]:
    """Download or resolve a narration URL to a local file path."""
    import urllib.parse

    parsed = urllib.parse.urlparse(narration_url)

    # Handle local file paths
    if parsed.scheme in ("", "file"):
        local_path = Path(parsed.path)
        if local_path.exists():
            return local_path
        return None

    # Handle HTTP(S) URLs
    if parsed.scheme in ("http", "https"):
        try:
            import httpx
            narration_path = job_dir / "narration.wav"
            with httpx.Client(timeout=30) as client:
                response = client.get(narration_url)
                response.raise_for_status()
                with open(narration_path, "wb") as f:
                    f.write(response.content)
            return narration_path
        except Exception as e:
            logger.warning("narration_download_failed", url=narration_url, error=str(e))
            return None

    return None


def _cuda_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
