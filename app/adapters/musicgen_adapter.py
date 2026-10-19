"""
MusicGen adapter — generates music using Meta's MusicGen model.
Runs locally on GPU (RTX 2050) or CPU.
Commercial use: MusicGen is MIT licensed, outputs are commercially usable.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import structlog

from app.adapters.base import GeneratedAudio, MusicAdapter

logger = structlog.get_logger()


class MusicGenAdapter(MusicAdapter):
    """
    Music generation adapter using Meta's MusicGen (audiocraft).

    MusicGen-small runs on RTX 2050 (4GB VRAM).
    Generates ~30s of music in ~15-20s on GPU.
    """

    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        output_dir: str | Path = "./output/raw",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._processor = None

        # Auto-detect device
        if device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        logger.info(
            "musicgen_adapter_init",
            model=model_name,
            device=self._device,
        )

    def _load_model(self):
        """Lazy-load the MusicGen model."""
        if self._model is not None:
            return

        logger.info("loading_musicgen_model", model=self.model_name)
        start = time.time()

        from audiocraft.models import MusicGen

        self._model = MusicGen.get_pretrained(self.model_name)
        self._model.set_generation_params(use_sampling=True, top_k=250)

        load_time = time.time() - start
        logger.info("musicgen_model_loaded", load_time_s=round(load_time, 2))

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
        Generate instrumental music from a text prompt using MusicGen.

        Strategy for exact length:
        1. Calculate bars from duration and BPM
        2. Generate slightly longer than needed
        3. Trim to exact sample count with musical fade-out

        Strategy for sections:
        - Generate each section separately with its own prompt
        - Crossfade between sections for smooth transitions
        """
        self._load_model()
        start_time = time.time()

        import torch

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Ensure prompt requests instrumental music
        instrumental_prompt = self._make_instrumental_prompt(prompt, bpm, key)

        warnings = []

        # Check for copyrighted content requests
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["cover", "copy", "remix", "sample from"]):
            warnings.append(
                "Request may reference existing music. "
                "Output is original AI-generated instrumental music."
            )

        if any(w in prompt_lower for w in ["vocals", "singing", "lyrics", "rap"]):
            warnings.append(
                "Vocal elements requested but not supported. "
                "Output is instrumental only."
            )

        if sections and len(sections) > 1:
            # Generate each section and crossfade
            audio = self._generate_with_sections(
                instrumental_prompt, duration_seconds, sections, sample_rate
            )
        else:
            # Single generation
            # Generate slightly longer to allow musical ending
            gen_duration = min(duration_seconds + 2.0, 300.0)
            self._model.set_generation_params(duration=gen_duration)

            with torch.no_grad():
                wav = self._model.generate([instrumental_prompt])

            # wav shape: [batch, channels, samples]
            audio = wav[0, 0].cpu().numpy()

        # Resample if needed (MusicGen outputs at 32kHz)
        model_sr = self._model.sample_rate
        if model_sr != sample_rate:
            audio = self._resample(audio, model_sr, sample_rate)

        # Trim/pad to exact length
        target_samples = int(duration_seconds * sample_rate)
        if len(audio) > target_samples:
            # Apply fade-out on last 500ms
            fade_samples = min(int(0.5 * sample_rate), target_samples // 4)
            audio = audio[:target_samples]
            if fade_samples > 0:
                fade = np.linspace(1.0, 0.0, fade_samples) ** 2
                audio[-fade_samples:] *= fade
        elif len(audio) < target_samples:
            # Pad with silence (faded)
            padded = np.zeros(target_samples, dtype=np.float32)
            padded[:len(audio)] = audio
            audio = padded

        # Normalize to prevent clipping
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = audio * (0.95 / peak)

        # Save output
        output_path = self.output_dir / f"musicgen_{int(time.time() * 1000)}.wav"
        sf.write(str(output_path), audio, sample_rate)

        generation_time = time.time() - start_time

        return GeneratedAudio(
            audio_path=output_path,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            model_name=self.model_name,
            model_version="1.4.0a1",
            seed=seed,
            cost_estimate_usd=0.0,  # Local inference, no API cost
            generation_time_seconds=round(generation_time, 3),
            warnings=warnings if warnings else None,
        )

    def _make_instrumental_prompt(
        self,
        prompt: str,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
    ) -> str:
        """Enhance the prompt to ensure instrumental-only output."""
        parts = [prompt.strip()]

        # Always specify instrumental
        if "instrumental" not in prompt.lower():
            parts.append("instrumental only, no vocals")

        if bpm:
            parts.append(f"{bpm} BPM")

        if key:
            parts.append(f"in the key of {key}")

        return ", ".join(parts)

    def _generate_with_sections(
        self,
        base_prompt: str,
        total_duration: float,
        sections: list[dict],
        sample_rate: int,
    ) -> np.ndarray:
        """Generate music with different moods at section boundaries."""
        import torch

        model_sr = self._model.sample_rate
        crossfade_samples = int(0.3 * model_sr)  # 300ms crossfade

        full_audio = np.array([], dtype=np.float32)

        for i, section in enumerate(sections):
            start_s = section.get("start_s", section.get("start_seconds", 0))
            mood = section.get("mood", section.get("prompt", ""))

            # Calculate section duration
            if i + 1 < len(sections):
                next_start = sections[i + 1].get(
                    "start_s", sections[i + 1].get("start_seconds", total_duration)
                )
                section_duration = next_start - start_s
            else:
                section_duration = total_duration - start_s

            if section_duration <= 0:
                continue

            # Generate section with appropriate duration + extra for crossfade
            gen_duration = min(section_duration + 1.0, 30.0)
            self._model.set_generation_params(duration=gen_duration)

            section_prompt = f"{base_prompt}, {mood}" if mood else base_prompt

            with torch.no_grad():
                wav = self._model.generate([section_prompt])

            section_audio = wav[0, 0].cpu().numpy()

            # Trim to section duration
            target_samples = int(section_duration * model_sr)
            if len(section_audio) > target_samples:
                section_audio = section_audio[:target_samples]
            elif len(section_audio) < target_samples:
                padded = np.zeros(target_samples, dtype=np.float32)
                padded[:len(section_audio)] = section_audio
                section_audio = padded

            # Crossfade with previous section
            if len(full_audio) > 0 and crossfade_samples > 0:
                overlap = min(crossfade_samples, len(full_audio), len(section_audio))
                if overlap > 0:
                    fade_out = np.linspace(1, 0, overlap)
                    fade_in = np.linspace(0, 1, overlap)
                    full_audio[-overlap:] = (
                        full_audio[-overlap:] * fade_out +
                        section_audio[:overlap] * fade_in
                    )
                    full_audio = np.concatenate([
                        full_audio, section_audio[overlap:]
                    ])
                else:
                    full_audio = np.concatenate([full_audio, section_audio])
            else:
                full_audio = np.concatenate([full_audio, section_audio])

        return full_audio

    def _resample(
        self, audio: np.ndarray, from_sr: int, to_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if from_sr == to_sr:
            return audio

        try:
            import torchaudio
            import torch

            tensor = torch.from_numpy(audio).unsqueeze(0)
            resampled = torchaudio.functional.resample(tensor, from_sr, to_sr)
            return resampled.squeeze(0).numpy()
        except ImportError:
            # Fallback: simple linear interpolation
            ratio = to_sr / from_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def is_available(self) -> bool:
        """Check if MusicGen can be loaded."""
        try:
            import audiocraft
            return True
        except ImportError:
            return False

    def get_info(self) -> dict:
        return {
            "name": "MusicGenAdapter",
            "version": "1.4.0a1",
            "model": self.model_name,
            "device": self._device,
            "description": "Meta's MusicGen text-to-music model. MIT licensed.",
            "license": "MIT (model) — commercial use of outputs allowed",
            "license_url": "https://github.com/facebookresearch/audiocraft/blob/main/LICENSE",
            "cost_per_minute": 0.0,
        }
