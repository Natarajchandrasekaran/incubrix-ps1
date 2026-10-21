"""
Mock music adapter for testing without API keys.
Generates genre-specific multi-track arrangements from text prompts.
Integration Standard: tests run with no API keys.

Composes distinctly different instruments, tempos, rhythms, chord progressions,
and melodic scales depending on whether the prompt asks for Corporate,
Cinematic, Lo-Fi, Electronic/Tech, Podcast, Jazz, Acoustic, Ambient,
Hip-Hop, or Rock music.
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
    Mock adapter with genre-specific synthesis.

    Analyzes prompt keywords (e.g. corporate, cinematic, lofi, tech, podcast)
    to compose completely unique instrument tracks for drums, bass, harmony,
    and melody that match the requested style, duration, and tempo.
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
        Generate genre-tailored instrument tracks and sum into a mix.
        """
        start_time = time.time()

        if seed is not None:
            np.random.seed(seed)
        else:
            # Seed based on prompt text so same prompt produces consistent music,
            # but different prompts produce completely different tracks
            prompt_seed = sum(ord(c) * (i + 1) for i, c in enumerate(prompt)) % 100000
            np.random.seed(prompt_seed)

        p_lower = prompt.lower()

        # ── 1. Detect Genre from Prompt Keywords ──
        if any(w in p_lower for w in ["cinema", "film", "movie", "drama", "epic", "string", "trailer", "orchestr"]):
            genre = "cinematic"
            target_bpm = bpm or 84
            scale = [146.8, 164.8, 174.6, 196.0, 220.0, 233.1, 261.6, 293.7] # D Minor
            chord_roots = [146.8, 116.5, 98.0, 110.0]  # Dm, Bb, Gm, Am
        elif any(w in p_lower for w in ["lofi", "lo-fi", "chill", "relax", "study", "mellow", "vinyl", "coffee"]):
            genre = "lofi"
            target_bpm = bpm or 78
            scale = [174.6, 196.0, 220.0, 246.9, 261.6, 293.7, 329.6, 349.2] # F Major 7
            chord_roots = [174.6, 220.0, 146.8, 196.0]  # Fmaj7, Am7, Dm7, Gm7
        elif any(w in p_lower for w in ["tech", "electron", "synth", "edm", "club", "dance", "drive", "cyber", "future"]):
            genre = "tech"
            target_bpm = bpm or 126
            scale = [220.0, 246.9, 261.6, 293.7, 329.6, 349.2, 392.0, 440.0] # A Minor
            chord_roots = [110.0, 130.8, 146.8, 164.8]  # Am, C, Dm, Em
        elif any(w in p_lower for w in ["podcast", "bed", "talk", "voice", "subtle", "gentle", "intro"]):
            genre = "podcast"
            target_bpm = bpm or 94
            scale = [196.0, 220.0, 246.9, 261.6, 293.7, 329.6, 369.9, 392.0] # G Major
            chord_roots = [98.0, 123.5, 130.8, 146.8]   # G, Bm, C, D
        elif any(w in p_lower for w in ["jazz", "swing", "blues", "bebop", "saxophone", "trumpet", "smooth"]):
            genre = "jazz"
            target_bpm = bpm or 108
            scale = [233.1, 261.6, 293.7, 311.1, 349.2, 392.0, 415.3, 466.2] # Bb Mixolydian
            chord_roots = [116.5, 130.8, 146.8, 155.6]  # Bb7, C7, Dm7, Eb
        elif any(w in p_lower for w in ["acoustic", "folk", "country", "guitar", "campfire", "unplugged", "fingerpick"]):
            genre = "acoustic"
            target_bpm = bpm or 100
            scale = [196.0, 220.0, 246.9, 261.6, 293.7, 329.6, 369.9, 392.0] # G Major
            chord_roots = [98.0, 130.8, 146.8, 110.0]   # G, C, D, Am
        elif any(w in p_lower for w in ["ambient", "space", "atmospheric", "ethereal", "drone", "meditation", "nature"]):
            genre = "ambient"
            target_bpm = bpm or 60
            scale = [261.6, 293.7, 329.6, 370.0, 392.0, 440.0, 493.9, 523.3] # C Lydian
            chord_roots = [130.8, 146.8, 164.8, 196.0]  # C, D, E, G
        elif any(w in p_lower for w in ["hip-hop", "hiphop", "rap", "trap", "boom bap", "urban", "beat"]):
            genre = "hiphop"
            target_bpm = bpm or 90
            scale = [164.8, 196.0, 220.0, 246.9, 293.7, 329.6, 392.0, 440.0] # E minor pentatonic
            chord_roots = [82.4, 110.0, 130.8, 98.0]    # Em, Am, C, G
        elif any(w in p_lower for w in ["rock", "punk", "grunge", "power", "heavy", "metal", "band"]):
            genre = "rock"
            target_bpm = bpm or 140
            scale = [164.8, 196.0, 220.0, 246.9, 293.7, 329.6, 392.0, 440.0] # E minor
            chord_roots = [82.4, 110.0, 130.8, 146.8]   # Em, Am, C, D
        else:
            # Default / Corporate Upbeat
            genre = "corporate"
            target_bpm = bpm or 120
            scale = [130.8, 146.8, 164.8, 174.6, 196.0, 220.0, 246.9, 261.6] # C Major
            chord_roots = [130.8, 196.0, 110.0, 174.6]  # C, G, Am, F

        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
        beat_interval = 60.0 / target_bpm
        beat_samples = max(1, int(beat_interval * sample_rate))
        bar_samples = beat_samples * 4

        drums = np.zeros(num_samples, dtype=np.float32)
        bass = np.zeros(num_samples, dtype=np.float32)
        harmony = np.zeros(num_samples, dtype=np.float32)
        melody = np.zeros(num_samples, dtype=np.float32)

        # ── 2. Synthesize DRUMS tailored to Genre ──
        if genre == "tech":
            # Driving 4-on-the-floor kick on every beat + crisp 16th hats
            for pos in range(0, num_samples, beat_samples):
                k_end = min(pos + int(0.08 * sample_rate), num_samples)
                kt = np.arange(k_end - pos) / sample_rate
                # Punchy techno kick sweep 150Hz -> 45Hz
                drums[pos:k_end] += (0.4 * np.sin(2 * np.pi * (150 * np.exp(-kt * 35) + 45) * kt) * np.exp(-kt * 25)).astype(np.float32)
                # Clap / snare on beats 2 and 4
                beat_num = (pos // beat_samples) % 4
                if beat_num in (1, 3):
                    c_end = min(pos + int(0.12 * sample_rate), num_samples)
                    ct = np.arange(c_end - pos) / sample_rate
                    c_noise = np.random.uniform(-0.25, 0.25, len(ct)).astype(np.float32)
                    drums[pos:c_end] += (c_noise * np.exp(-ct * 30)).astype(np.float32)
            # Offbeat open hi-hat
            for pos in range(beat_samples // 2, num_samples, beat_samples):
                h_end = min(pos + int(0.04 * sample_rate), num_samples)
                ht = np.arange(h_end - pos) / sample_rate
                h_noise = np.random.uniform(-0.12, 0.12, len(ht)).astype(np.float32)
                drums[pos:h_end] += (h_noise * np.exp(-ht * 50)).astype(np.float32)

        elif genre == "cinematic":
            # Heavy orchestral taiko impact on bar 1 and 3 with long rumble
            for pos in range(0, num_samples, bar_samples):
                t_end = min(pos + int(0.6 * sample_rate), num_samples)
                tt = np.arange(t_end - pos) / sample_rate
                drums[pos:t_end] += (0.45 * np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 6)).astype(np.float32)
                # Marching snare rim hits on beat 3
                beat3 = pos + beat_samples * 2
                if beat3 < num_samples:
                    r_end = min(beat3 + int(0.1 * sample_rate), num_samples)
                    rt = np.arange(r_end - beat3) / sample_rate
                    drums[beat3:r_end] += (0.2 * np.sin(2 * np.pi * 180 * rt) * np.exp(-rt * 20)).astype(np.float32)

        elif genre == "lofi":
            # Dusty, swinging boom-bap (soft thumpy kick on 1 & 2-and, soft snare on 2 & 4)
            for bar_pos in range(0, num_samples, bar_samples):
                # Kick 1
                k_end = min(bar_pos + int(0.09 * sample_rate), num_samples)
                kt = np.arange(k_end - bar_pos) / sample_rate
                drums[bar_pos:k_end] += (0.3 * np.sin(2 * np.pi * 70 * kt) * np.exp(-kt * 22)).astype(np.float32)
                # Snare on 2 and 4 with slight vinyl rustle
                for b_idx in [1, 3]:
                    s_pos = bar_pos + b_idx * beat_samples
                    if s_pos < num_samples:
                        s_end = min(s_pos + int(0.12 * sample_rate), num_samples)
                        st = np.arange(s_end - s_pos) / sample_rate
                        noise = np.random.uniform(-0.15, 0.15, len(st)).astype(np.float32)
                        drums[s_pos:s_end] += (noise * np.exp(-st * 25) + 0.15 * np.sin(2 * np.pi * 180 * st) * np.exp(-st * 30)).astype(np.float32)

        elif genre == "jazz":
            # Brushed swing ride cymbal with ghost snare notes
            for pos in range(0, num_samples, beat_samples):
                # Ride cymbal shimmer on every beat
                r_end = min(pos + int(0.15 * sample_rate), num_samples)
                rt = np.arange(r_end - pos) / sample_rate
                ride = np.random.uniform(-0.08, 0.08, len(rt)).astype(np.float32)
                drums[pos:r_end] += (ride * np.exp(-rt * 12)).astype(np.float32)
                # Swung offbeat hat
                swing_pos = pos + int(beat_samples * 0.66)
                if swing_pos < num_samples:
                    sw_end = min(swing_pos + int(0.05 * sample_rate), num_samples)
                    swt = np.arange(sw_end - swing_pos) / sample_rate
                    drums[swing_pos:sw_end] += (np.random.uniform(-0.06, 0.06, len(swt)) * np.exp(-swt * 40)).astype(np.float32)
                # Light kick on beats 1 and 3
                beat_num = (pos // beat_samples) % 4
                if beat_num in (0, 2):
                    k_end = min(pos + int(0.07 * sample_rate), num_samples)
                    kt = np.arange(k_end - pos) / sample_rate
                    drums[pos:k_end] += (0.2 * np.sin(2 * np.pi * 65 * kt) * np.exp(-kt * 25)).astype(np.float32)

        elif genre == "acoustic":
            # Soft shaker + light kick, very minimal
            for pos in range(0, num_samples, beat_samples):
                # Gentle kick on 1 and 3
                beat_num = (pos // beat_samples) % 4
                if beat_num in (0, 2):
                    k_end = min(pos + int(0.05 * sample_rate), num_samples)
                    kt = np.arange(k_end - pos) / sample_rate
                    drums[pos:k_end] += (0.18 * np.sin(2 * np.pi * 75 * kt) * np.exp(-kt * 35)).astype(np.float32)
                # Soft shaker on every 8th note
                for sub in range(2):
                    sh_pos = pos + sub * (beat_samples // 2)
                    if sh_pos < num_samples:
                        sh_end = min(sh_pos + int(0.03 * sample_rate), num_samples)
                        sht = np.arange(sh_end - sh_pos) / sample_rate
                        drums[sh_pos:sh_end] += (np.random.uniform(-0.05, 0.05, len(sht)) * np.exp(-sht * 60)).astype(np.float32)

        elif genre == "ambient":
            # No traditional drums — just subtle textural noise swells
            swell_len = bar_samples * 2
            for pos in range(0, num_samples, swell_len):
                s_end = min(pos + swell_len, num_samples)
                st = np.arange(s_end - pos) / sample_rate
                noise_swell = np.random.uniform(-0.03, 0.03, len(st)).astype(np.float32)
                env = np.sin(np.pi * np.linspace(0, 1, len(st))).astype(np.float32)
                drums[pos:s_end] += noise_swell * env

        elif genre == "hiphop":
            # 808 sub kick + trap hi-hats with triplet rolls
            for bar_pos in range(0, num_samples, bar_samples):
                # Heavy 808 kick on beat 1
                k_end = min(bar_pos + int(0.25 * sample_rate), num_samples)
                kt = np.arange(k_end - bar_pos) / sample_rate
                drums[bar_pos:k_end] += (0.4 * np.sin(2 * np.pi * (120 * np.exp(-kt * 20) + 40) * kt) * np.exp(-kt * 8)).astype(np.float32)
                # Snare on 2 and 4
                for b_idx in [1, 3]:
                    s_pos = bar_pos + b_idx * beat_samples
                    if s_pos < num_samples:
                        s_end = min(s_pos + int(0.1 * sample_rate), num_samples)
                        st = np.arange(s_end - s_pos) / sample_rate
                        drums[s_pos:s_end] += (np.random.uniform(-0.25, 0.25, len(st)) * np.exp(-st * 20)).astype(np.float32)
                # Triplet hi-hats (3 per beat)
                triplet_step = beat_samples // 3
                for tp in range(bar_pos, min(bar_pos + bar_samples, num_samples), triplet_step):
                    h_end = min(tp + int(0.02 * sample_rate), num_samples)
                    ht = np.arange(h_end - tp) / sample_rate
                    drums[tp:h_end] += (np.random.uniform(-0.1, 0.1, len(ht)) * np.exp(-ht * 80)).astype(np.float32)

        elif genre == "rock":
            # Driving kick-snare with crash cymbals and fast 8th-note hats
            for pos in range(0, num_samples, beat_samples):
                beat_num = (pos // beat_samples) % 4
                # Kick on 1 and 3
                if beat_num in (0, 2):
                    k_end = min(pos + int(0.07 * sample_rate), num_samples)
                    kt = np.arange(k_end - pos) / sample_rate
                    drums[pos:k_end] += (0.38 * np.sin(2 * np.pi * 90 * kt) * np.exp(-kt * 28)).astype(np.float32)
                # Heavy snare on 2 and 4
                if beat_num in (1, 3):
                    s_end = min(pos + int(0.12 * sample_rate), num_samples)
                    st = np.arange(s_end - pos) / sample_rate
                    drums[pos:s_end] += (0.15 * np.sin(2 * np.pi * 200 * st) * np.exp(-st * 18) + np.random.uniform(-0.2, 0.2, len(st)).astype(np.float32) * np.exp(-st * 22)).astype(np.float32)
                # 8th note hats
                for sub in range(2):
                    h_pos = pos + sub * (beat_samples // 2)
                    if h_pos < num_samples:
                        h_end = min(h_pos + int(0.03 * sample_rate), num_samples)
                        ht = np.arange(h_end - h_pos) / sample_rate
                        drums[h_pos:h_end] += (np.random.uniform(-0.12, 0.12, len(ht)) * np.exp(-ht * 55)).astype(np.float32)

        else:
            # Corporate & Podcast: clean acoustic pop beat with tambourine / claps
            for pos in range(0, num_samples, beat_samples):
                k_end = min(pos + int(0.06 * sample_rate), num_samples)
                kt = np.arange(k_end - pos) / sample_rate
                drums[pos:k_end] += (0.28 * np.sin(2 * np.pi * 80 * kt) * np.exp(-kt * 30)).astype(np.float32)
                beat_num = (pos // beat_samples) % 4
                if beat_num in (1, 3): # Clap on 2 and 4
                    c_end = min(pos + int(0.08 * sample_rate), num_samples)
                    ct = np.arange(c_end - pos) / sample_rate
                    c_noise = np.random.uniform(-0.18, 0.18, len(ct)).astype(np.float32)
                    drums[pos:c_end] += (c_noise * np.exp(-ct * 35)).astype(np.float32)

        # ── 3. Synthesize BASS tailored to Genre ──
        for bar_idx, bar_pos in enumerate(range(0, num_samples, bar_samples)):
            root_f = chord_roots[bar_idx % len(chord_roots)]
            b_end = min(bar_pos + bar_samples, num_samples)
            bar_len = b_end - bar_pos

            if genre == "tech":
                # Rolling 16th-note synth saw bass
                sub_step = beat_samples // 4
                for step_p in range(bar_pos, b_end, sub_step):
                    s_end = min(step_p + sub_step, num_samples)
                    st = np.arange(s_end - step_p) / sample_rate
                    # Saw wave rich in harmonics
                    saw = (0.25 * np.sin(2 * np.pi * root_f * st) +
                           0.12 * np.sin(4 * np.pi * root_f * st) +
                           0.06 * np.sin(6 * np.pi * root_f * st)) * np.exp(-st * 14)
                    bass[step_p:s_end] += saw.astype(np.float32)

            elif genre == "cinematic":
                # Heavy sustained brass/sub-drone across entire bar
                bt = np.arange(bar_len) / sample_rate
                sub = (0.35 * np.sin(2 * np.pi * (root_f * 0.5) * bt) +
                       0.15 * np.sin(2 * np.pi * root_f * bt)) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * bt))
                bass[bar_pos:b_end] += sub.astype(np.float32)

            elif genre == "lofi":
                # Warm acoustic upright walking bass notes on beats 1 and 3
                for b_step in [0, beat_samples * 2]:
                    note_p = bar_pos + b_step
                    if note_p < num_samples:
                        n_end = min(note_p + int(beat_samples * 1.5), num_samples)
                        nt = np.arange(n_end - note_p) / sample_rate
                        w_bass = (0.28 * np.sin(2 * np.pi * (root_f * 0.75) * nt) * np.exp(-nt * 3.5)).astype(np.float32)
                        bass[note_p:n_end] += w_bass

            elif genre == "jazz":
                # Walking bass: quarter notes stepping through chord tones
                walk_notes = [root_f * 0.5, root_f * 0.63, root_f * 0.75, root_f * 0.5 * 1.5]
                for b_idx, b_pos in enumerate(range(bar_pos, b_end, beat_samples)):
                    if b_pos < num_samples:
                        n_end = min(b_pos + int(beat_samples * 0.85), num_samples)
                        nt = np.arange(n_end - b_pos) / sample_rate
                        wf = walk_notes[b_idx % 4]
                        bass[b_pos:n_end] += (0.3 * np.sin(2 * np.pi * wf * nt) * np.exp(-nt * 4.0)).astype(np.float32)

            elif genre == "acoustic":
                # Fingerpicked root-fifth pattern
                for b_idx, b_pos in enumerate(range(bar_pos, b_end, beat_samples)):
                    if b_pos < num_samples:
                        n_end = min(b_pos + int(beat_samples * 0.7), num_samples)
                        nt = np.arange(n_end - b_pos) / sample_rate
                        pitch = root_f * 0.5 if b_idx % 2 == 0 else root_f * 0.75
                        bass[b_pos:n_end] += (0.22 * np.sin(2 * np.pi * pitch * nt) * np.exp(-nt * 5.0)).astype(np.float32)

            elif genre == "ambient":
                # Deep sub drone across entire bar
                bt = np.arange(bar_len) / sample_rate
                sub = (0.2 * np.sin(2 * np.pi * (root_f * 0.25) * bt) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.15 * bt))).astype(np.float32)
                bass[bar_pos:b_end] += sub

            elif genre == "hiphop":
                # 808 sustained sub bass on beats 1 and 3
                for b_step in [0, beat_samples * 2]:
                    note_p = bar_pos + b_step
                    if note_p < num_samples:
                        n_end = min(note_p + beat_samples * 2, num_samples)
                        nt = np.arange(n_end - note_p) / sample_rate
                        bass[note_p:n_end] += (0.35 * np.sin(2 * np.pi * (root_f * 0.5) * nt) * np.exp(-nt * 2.0)).astype(np.float32)

            elif genre == "rock":
                # Driving 8th note power bass
                step = beat_samples // 2
                for s_pos in range(bar_pos, b_end, step):
                    if s_pos < num_samples:
                        s_end = min(s_pos + int(step * 0.8), num_samples)
                        st = np.arange(s_end - s_pos) / sample_rate
                        bass[s_pos:s_end] += (0.3 * np.sin(2 * np.pi * (root_f * 0.5) * st) * np.exp(-st * 8.0)).astype(np.float32)

            else:
                # Corporate: Bouncy root-fifth groove
                for step_idx, step_p in enumerate(range(bar_pos, b_end, beat_samples)):
                    s_end = min(step_p + int(beat_samples * 0.8), num_samples)
                    st = np.arange(s_end - step_p) / sample_rate
                    pitch = root_f if step_idx % 2 == 0 else root_f * 1.5 # root then fifth
                    c_bass = (0.26 * np.sin(2 * np.pi * pitch * st) * np.exp(-st * 5.0)).astype(np.float32)
                    bass[step_p:s_end] += c_bass

        # ── 4. Synthesize HARMONY / PADS tailored to Genre ──
        for bar_idx, bar_pos in enumerate(range(0, num_samples, bar_samples)):
            root_f = chord_roots[bar_idx % len(chord_roots)]
            h_end = min(bar_pos + bar_samples, num_samples)
            ht = np.arange(h_end - bar_pos) / sample_rate

            if genre == "cinematic":
                # Minor orchestral string chord: root, minor third (1.189), fifth (1.498)
                chord_multipliers = [1.0, 1.189, 1.498, 2.0]
                chord_audio = np.zeros_like(ht, dtype=np.float32)
                for mult in chord_multipliers:
                    chord_audio += (0.07 * np.sin(2 * np.pi * (root_f * 2 * mult) * ht)).astype(np.float32)
                # Slow crescendo envelope per bar
                swell = (0.4 + 0.6 * np.sin(np.pi * (ht / (bar_samples / sample_rate)))).astype(np.float32)
                harmony[bar_pos:h_end] += chord_audio * swell

            elif genre == "lofi":
                # Lush jazzy Rhodes chords (Major 7th: 1.0, 1.26, 1.498, 1.887)
                chord_audio = np.zeros_like(ht, dtype=np.float32)
                for mult in [1.0, 1.26, 1.498, 1.887]:
                    chord_audio += (0.06 * np.sin(2 * np.pi * (root_f * 1.5 * mult) * ht)).astype(np.float32)
                # Rhodes bell-like decay
                decay = np.exp(-ht * 2.0).astype(np.float32)
                harmony[bar_pos:h_end] += chord_audio * decay

            elif genre == "jazz":
                # Rich 7th/9th voicings with gentle tremolo
                chord_audio = np.zeros_like(ht, dtype=np.float32)
                for mult in [1.0, 1.26, 1.498, 1.78, 2.24]:  # root, 3rd, 5th, 7th, 9th
                    chord_audio += (0.05 * np.sin(2 * np.pi * (root_f * 1.5 * mult) * ht)).astype(np.float32)
                tremolo = (0.85 + 0.15 * np.sin(2 * np.pi * 3.5 * ht)).astype(np.float32)
                harmony[bar_pos:h_end] += chord_audio * np.exp(-ht * 1.5).astype(np.float32) * tremolo

            elif genre == "acoustic":
                # Warm strummed triads with gentle attack
                chord_audio = np.zeros_like(ht, dtype=np.float32)
                for mult in [1.0, 1.26, 1.498, 2.0]:
                    chord_audio += (0.07 * np.sin(2 * np.pi * (root_f * 2 * mult) * ht)).astype(np.float32)
                env = np.minimum(ht * 8, 1.0).astype(np.float32) * np.exp(-ht * 0.8).astype(np.float32)
                harmony[bar_pos:h_end] += chord_audio * env

            elif genre == "ambient":
                # Evolving pad with slow filter sweep
                chord_audio = np.zeros_like(ht, dtype=np.float32)
                for mult in [1.0, 1.498, 2.0, 2.997]:
                    phase = np.random.uniform(0, 2 * np.pi)
                    chord_audio += (0.06 * np.sin(2 * np.pi * (root_f * mult) * ht + phase)).astype(np.float32)
                swell = np.sin(np.pi * np.linspace(0, 1, len(ht))).astype(np.float32)
                harmony[bar_pos:h_end] += chord_audio * swell

            elif genre == "hiphop":
                # Dark minor stabs on beat 1 with decay
                stab_end = min(bar_pos + beat_samples * 2, num_samples)
                stab_len = stab_end - bar_pos
                st = np.arange(stab_len) / sample_rate
                stab = np.zeros(stab_len, dtype=np.float32)
                for mult in [1.0, 1.189, 1.498]:  # minor triad
                    stab += (0.08 * np.sin(2 * np.pi * (root_f * 2 * mult) * st)).astype(np.float32)
                harmony[bar_pos:stab_end] += stab * np.exp(-st * 3.0).astype(np.float32)

            elif genre == "rock":
                # Power chords (root + fifth) with distortion-like overtones
                chord_audio = np.zeros_like(ht, dtype=np.float32)
                for mult in [1.0, 1.498, 2.0, 2.997]:
                    chord_audio += (0.09 * np.sin(2 * np.pi * (root_f * 2 * mult) * ht)).astype(np.float32)
                # Clip for distortion effect
                chord_audio = np.clip(chord_audio * 2.5, -0.25, 0.25).astype(np.float32)
                harmony[bar_pos:h_end] += chord_audio

            elif genre == "tech":
                # Supersaw sidechained synth chords on every beat
                for beat_p in range(bar_pos, h_end, beat_samples):
                    b_end = min(beat_p + beat_samples, num_samples)
                    bt = np.arange(b_end - beat_p) / sample_rate
                    stab = np.zeros_like(bt, dtype=np.float32)
                    for mult in [1.0, 1.25, 1.5, 2.0]:
                        stab += (0.05 * np.sin(2 * np.pi * (root_f * 2 * mult) * bt) +
                                 0.03 * np.sin(2 * np.pi * (root_f * 2 * mult * 1.008) * bt)).astype(np.float32)
                    # Sidechain ducking curve (duck on beat hit, swell up)
                    sidechain = (0.2 + 0.8 * (bt / (beat_samples / sample_rate))).astype(np.float32)
                    harmony[beat_p:b_end] += stab * sidechain

            else:
                # Corporate: Warm bright major triad acoustic strum
                chord_audio = np.zeros_like(ht, dtype=np.float32)
                for mult in [1.0, 1.26, 1.498]:
                    chord_audio += (0.08 * np.sin(2 * np.pi * (root_f * 2 * mult) * ht)).astype(np.float32)
                harmony[bar_pos:h_end] += chord_audio * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * ht)).astype(np.float32)

        # ── 5. Synthesize MELODY tailored to Genre ──
        if genre == "tech":
            # Fast arpeggiated 16th notes
            step_len = beat_samples // 4
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                s_end = min(s_pos + step_len, num_samples)
                mt = np.arange(s_end - s_pos) / sample_rate
                f_idx = (s_idx * 3) % len(scale)
                lead_f = scale[f_idx] * 2.0
                lead = (0.12 * np.sin(2 * np.pi * lead_f * mt) * np.exp(-mt * 15)).astype(np.float32)
                melody[s_pos:s_end] += lead

        elif genre == "cinematic":
            # Soaring, slow dramatic violin lines that change every 2 beats
            step_len = beat_samples * 2
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                s_end = min(s_pos + step_len, num_samples)
                mt = np.arange(s_end - s_pos) / sample_rate
                f_idx = [0, 2, 4, 6, 7, 5, 3, 1][s_idx % 8]
                lead_f = scale[f_idx] * 2.0
                # Add expressive vibrato
                vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 5.5 * mt)
                lead = (0.16 * np.sin(2 * np.pi * (lead_f * vibrato) * mt) *
                        np.sin(np.pi * (mt / (step_len / sample_rate)))).astype(np.float32)
                melody[s_pos:s_end] += lead

        elif genre == "lofi":
            # Gentle pentatonic electric piano notes with pauses
            step_len = beat_samples
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                # Play on some beats, leave space on others
                if s_idx % 3 != 0:
                    s_end = min(s_pos + int(step_len * 0.9), num_samples)
                    mt = np.arange(s_end - s_pos) / sample_rate
                    f_idx = (s_idx * 2) % len(scale)
                    lead_f = scale[f_idx] * 1.5
                    lead = (0.14 * np.sin(2 * np.pi * lead_f * mt) * np.exp(-mt * 4.0)).astype(np.float32)
                    melody[s_pos:s_end] += lead

        elif genre == "jazz":
            # Chromatic bebop runs with swung timing
            step_len = int(beat_samples * 0.66)
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                if s_idx % 5 != 0:  # leave gaps
                    s_end = min(s_pos + int(step_len * 0.7), num_samples)
                    mt = np.arange(s_end - s_pos) / sample_rate
                    f_idx = (s_idx * 3 + 1) % len(scale)
                    lead_f = scale[f_idx] * 1.5
                    vibrato = 1.0 + 0.015 * np.sin(2 * np.pi * 6 * mt)
                    lead = (0.12 * np.sin(2 * np.pi * (lead_f * vibrato) * mt) * np.exp(-mt * 6.0)).astype(np.float32)
                    melody[s_pos:s_end] += lead

        elif genre == "acoustic":
            # Simple fingerpicked melody on every other beat
            step_len = beat_samples
            pattern = [0, 4, 2, 5, 3, 7, 4, 0]
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                if s_idx % 2 == 0:
                    s_end = min(s_pos + int(step_len * 0.8), num_samples)
                    mt = np.arange(s_end - s_pos) / sample_rate
                    f_idx = pattern[s_idx % len(pattern)]
                    lead_f = scale[f_idx] * 2.0
                    lead = (0.11 * np.sin(2 * np.pi * lead_f * mt) * np.exp(-mt * 5.0)).astype(np.float32)
                    melody[s_pos:s_end] += lead

        elif genre == "ambient":
            # Very slow evolving tones with long tails
            step_len = bar_samples * 2
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                s_end = min(s_pos + step_len, num_samples)
                mt = np.arange(s_end - s_pos) / sample_rate
                f_idx = (s_idx * 2) % len(scale)
                lead_f = scale[f_idx]
                env = np.sin(np.pi * np.linspace(0, 1, len(mt))).astype(np.float32)
                lead = (0.1 * np.sin(2 * np.pi * lead_f * mt) * env).astype(np.float32)
                melody[s_pos:s_end] += lead

        elif genre == "hiphop":
            # Dark catchy top-line, sparse notes
            step_len = beat_samples
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                if s_idx % 4 in (0, 2, 3):
                    s_end = min(s_pos + int(step_len * 0.5), num_samples)
                    mt = np.arange(s_end - s_pos) / sample_rate
                    f_idx = (s_idx * 2) % len(scale)
                    lead_f = scale[f_idx] * 1.5
                    lead = (0.1 * np.sin(2 * np.pi * lead_f * mt) * np.exp(-mt * 7.0)).astype(np.float32)
                    melody[s_pos:s_end] += lead

        elif genre == "rock":
            # Fast pentatonic riffs
            step_len = beat_samples // 4
            pattern = [0, 3, 4, 3, 0, 4, 7, 4]
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                if s_idx % 3 != 2:
                    s_end = min(s_pos + int(step_len * 0.7), num_samples)
                    mt = np.arange(s_end - s_pos) / sample_rate
                    f_idx = pattern[s_idx % len(pattern)]
                    lead_f = scale[f_idx] * 2.0
                    lead = (0.14 * np.sin(2 * np.pi * lead_f * mt) * np.exp(-mt * 12)).astype(np.float32)
                    melody[s_pos:s_end] += lead

        else:
            # Corporate: Catchy glockenspiel / chime bell hook
            step_len = beat_samples // 2
            pattern = [0, 2, 4, 7, 4, 2, 0, 4]
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                s_end = min(s_pos + int(step_len * 0.7), num_samples)
                mt = np.arange(s_end - s_pos) / sample_rate
                f_idx = pattern[s_idx % len(pattern)]
                lead_f = scale[f_idx] * 2.5
                lead = (0.13 * np.sin(2 * np.pi * lead_f * mt) * np.exp(-mt * 8.0)).astype(np.float32)
                melody[s_pos:s_end] += lead

        # ── 6. Musical Ending & Envelope ──
        fade_in = int(num_samples * 0.02)
        fade_out = int(num_samples * 0.06)
        for stem in [drums, bass, harmony, melody]:
            if fade_in > 0:
                stem[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
            if fade_out > 0:
                stem[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)

        # ── 7. Sum Stems into Mix with Clean Headroom ──
        mix = drums + bass + harmony + melody
        peak = np.max(np.abs(mix))
        if peak > 0:
            scale_factor = 0.85 / peak
            mix *= scale_factor
            drums *= scale_factor
            bass *= scale_factor
            harmony *= scale_factor
            melody *= scale_factor

        # ── 8. Save Files ──
        timestamp = int(time.time() * 1000)
        mix_path = self.output_dir / f"mock_{genre}_{timestamp}.wav"
        sf.write(str(mix_path), mix, sample_rate)

        stems_dir = self.output_dir / f"mock_{genre}_{timestamp}_stems"
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
            model_name=f"mock-{genre}-engine",
            model_version="3.0.0",
            seed=seed,
            cost_estimate_usd=0.0,
            generation_time_seconds=round(generation_time, 3),
            warnings=[f"Composed in {genre.upper()} genre based on your prompt keywords."],
            metadata={"stems_dir": str(stems_dir), "genre": genre},
        )

    def is_available(self) -> bool:
        return True

    def get_info(self) -> dict:
        return {
            "name": "MockMusicAdapter",
            "version": "3.0.0",
            "description": "Composes genre-tailored multi-track music (Cinematic, Lo-Fi, Tech, Corporate, Podcast, Jazz, Acoustic, Ambient, Hip-Hop, Rock) with real stem separation.",
            "license": "MIT",
            "cost_per_minute": 0.0,
        }
