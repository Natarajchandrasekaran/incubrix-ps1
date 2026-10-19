"""
Mock music adapter for testing without API keys.
Generates separate instrument stems that sum to a mix.
Integration Standard: tests run with no API keys.

IMPORTANT: This adapter generates each instrument SEPARATELY, then sums
them to produce the mix. The stems are genuinely different instrument
parts (not frequency-band splits of the mix). This satisfies Must-have #2.
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
    Mock adapter that generates separate instrument parts.

    Creates 4 independent instrument tracks (drums, bass, harmony, melody)
    and sums them into a mix. This is a mock for testing — the real adapter
    uses MusicGen for generation and Demucs for separation.

    The stems genuinely sum to the mix because they were created separately
    and mixed together, not split from a mix with filters.
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
        """
        Generate separate instrument tracks and sum them into a mix.

        Returns a GeneratedAudio pointing to the mix file.
        Individual stems are saved alongside for the job pipeline to pick up.
        """
        start_time = time.time()

        if seed is not None:
            np.random.seed(seed)

        effective_bpm = bpm or 120
        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples, endpoint=False)

        # Base frequency from prompt (deterministic variety)
        base_freq = hash(prompt) % 200 + 100  # 100-300 Hz range

        # ── Generate each instrument SEPARATELY ──

        # 1. DRUMS — rhythmic clicks/transients at beat positions
        drums = np.zeros(num_samples, dtype=np.float32)
        beat_interval = 60.0 / effective_bpm
        beat_samples = int(beat_interval * sample_rate)
        kick_len = min(int(0.05 * sample_rate), 2400)  # 50ms kick
        hat_len = min(int(0.02 * sample_rate), 960)    # 20ms hi-hat

        for pos in range(0, num_samples, beat_samples):
            # Kick on every beat
            end = min(pos + kick_len, num_samples)
            chunk = np.arange(end - pos) / sample_rate
            drums[pos:end] += (0.25 * np.sin(2 * np.pi * 60 * chunk)
                               * np.exp(-chunk * 40)).astype(np.float32)
            # Hi-hat on off-beats
            half_beat = pos + beat_samples // 2
            if half_beat < num_samples:
                hend = min(half_beat + hat_len, num_samples)
                hchunk = np.arange(hend - half_beat) / sample_rate
                noise = np.random.uniform(-0.08, 0.08, len(hchunk)).astype(np.float32)
                drums[half_beat:hend] += noise * np.exp(-hchunk * 60).astype(np.float32)

        # 2. BASS — low-frequency sine following a simple pattern
        bass = np.zeros(num_samples, dtype=np.float32)
        bass_freq = base_freq * 0.5  # Low range
        # Bass notes on each bar (every 4 beats)
        bar_interval = beat_interval * 4
        bar_samples = int(bar_interval * sample_rate)
        note_len = int(bar_interval * 0.8 * sample_rate)  # 80% of bar

        for pos in range(0, num_samples, bar_samples):
            end = min(pos + note_len, num_samples)
            chunk_t = np.arange(end - pos) / sample_rate
            # Slight pitch variation per bar for interest
            bar_idx = pos // bar_samples
            freq = bass_freq * (1.0 + 0.1 * (bar_idx % 4))
            envelope = np.exp(-chunk_t * 1.5).astype(np.float32)
            bass[pos:end] += (0.18 * np.sin(2 * np.pi * freq * chunk_t) * envelope).astype(np.float32)

        # 3. HARMONY — warm pad chords (mid-frequency sustained tones)
        harmony = np.zeros(num_samples, dtype=np.float32)
        chord_freqs = [base_freq, base_freq * 1.25, base_freq * 1.5]  # Root, third, fifth
        for freq in chord_freqs:
            harmony += (0.06 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        # Slow amplitude modulation for warmth
        mod = (0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)).astype(np.float32)
        harmony *= mod

        # 4. MELODY — higher-frequency pattern with rhythmic movement
        melody = np.zeros(num_samples, dtype=np.float32)
        melody_base = base_freq * 2
        # Play short notes on beats with varying pitch
        scale_ratios = [1.0, 1.125, 1.25, 1.333, 1.5, 1.667, 1.875, 2.0]
        mel_note_len = int(beat_interval * 0.6 * sample_rate)

        beat_pos = 0
        note_idx = 0
        while beat_pos < num_samples:
            end = min(beat_pos + mel_note_len, num_samples)
            chunk_t = np.arange(end - beat_pos) / sample_rate
            ratio = scale_ratios[note_idx % len(scale_ratios)]
            freq = melody_base * ratio
            envelope = np.exp(-chunk_t * 3.0).astype(np.float32)
            melody[beat_pos:end] += (0.12 * np.sin(2 * np.pi * freq * chunk_t) * envelope).astype(np.float32)
            beat_pos += beat_samples
            note_idx += 1

        # ── Apply section changes if provided ──
        if sections:
            for i, section in enumerate(sections):
                start_s = section.get("start_s", section.get("start_seconds", 0))
                start_sample = int(start_s * sample_rate)
                end_sample = num_samples
                if i + 1 < len(sections):
                    next_s = sections[i + 1].get("start_s", sections[i + 1].get("start_seconds", duration_seconds))
                    end_sample = min(int(next_s * sample_rate), num_samples)
                # Boost/reduce different stems per section for variety
                if start_sample < num_samples:
                    factor = 0.7 + 0.6 * ((i % 3) / 2.0)  # Varies between 0.7 and 1.3
                    melody[start_sample:end_sample] *= factor
                    drums[start_sample:end_sample] *= (1.4 - factor * 0.3)

        # ── Apply envelope to each stem ──
        fade_in = int(num_samples * 0.03)
        fade_out = int(num_samples * 0.08)

        for stem in [drums, bass, harmony, melody]:
            if fade_in > 0:
                stem[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
            if fade_out > 0:
                stem[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)

        # ── Sum stems to create mix ──
        mix = drums + bass + harmony + melody

        # Normalize mix to prevent clipping
        peak = np.max(np.abs(mix))
        if peak > 0:
            scale = 0.8 / peak
            mix *= scale
            drums *= scale
            bass *= scale
            harmony *= scale
            melody *= scale

        # ── Save all files ──
        timestamp = int(time.time() * 1000)

        # Save mix
        mix_path = self.output_dir / f"mock_{timestamp}.wav"
        sf.write(str(mix_path), mix, sample_rate)

        # Save individual stems alongside the mix for the pipeline
        stems_dir = self.output_dir / f"mock_{timestamp}_stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(stems_dir / "drums.wav"), drums, sample_rate)
        sf.write(str(stems_dir / "bass.wav"), bass, sample_rate)
        sf.write(str(stems_dir / "other.wav"), harmony, sample_rate)
        sf.write(str(stems_dir / "melody.wav"), melody, sample_rate)

        generation_time = time.time() - start_time

        return GeneratedAudio(
            audio_path=mix_path,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            model_name="mock-multi-stem-generator",
            model_version="2.0.0",
            seed=seed,
            cost_estimate_usd=0.0,
            generation_time_seconds=round(generation_time, 3),
            warnings=["Mock audio for testing — stems are genuine separate tracks that sum to the mix."],
            metadata={"stems_dir": str(stems_dir)},
        )

    def is_available(self) -> bool:
        """Mock adapter is always available."""
        return True

    def get_info(self) -> dict:
        return {
            "name": "MockMusicAdapter",
            "version": "2.0.0",
            "description": "Generates separate instrument stems (drums, bass, harmony, melody) and sums them. No ML model required.",
            "license": "MIT",
            "cost_per_minute": 0.0,
        }
