"""
Mock music adapter for testing and studio-grade offline synthesis without API keys.
Generates situation-aware, emotionally expressive multi-track acoustic arrangements
using physical modeling (Karplus-Strong plucked strings, felt-hammer grand piano,
bowed string ensemble chorus, acoustic drum transient modeling, and studio room reverberation).
Follows IncuBrix Integration Standard v2.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal as signal
import soundfile as sf

from app.adapters.base import GeneratedAudio, MusicAdapter


# ─── Acoustic Physical Modeling Instruments ───

def _synthesize_piano_chord(freqs: list[float], duration_s: float, sr: int = 48000, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Acoustic grand piano model: Felt hammer strike + inharmonic string dispersion + wooden soundboard resonance."""
    if rng is None:
        rng = np.random.RandomState(42)
    n_samples = int(duration_s * sr)
    t = np.arange(n_samples) / sr
    chord = np.zeros(n_samples, dtype=np.float32)

    # Piano string stiffness inharmonicity factor B
    B = 0.00035
    partials = [
        (1.0, 0.62, 1.8),
        (2.0, 0.36, 2.8),
        (3.0, 0.22, 3.9),
        (4.0, 0.13, 5.4),
        (5.0, 0.07, 7.0),
        (6.0, 0.03, 8.8),
    ]

    for f in freqs:
        # 1. Felt hammer contact transient
        hammer = (rng.uniform(-0.05, 0.05, n_samples) * np.exp(-t * 90)).astype(np.float32)
        # 2. Inharmonic partials with dual-string unison detuning
        tone = np.zeros(n_samples, dtype=np.float32)
        for mult, amp, decay in partials:
            f_inharm = (mult * f) * np.sqrt(1.0 + B * (mult ** 2))
            string_a = np.sin(2 * np.pi * f_inharm * t) * np.exp(-t * decay)
            string_b = np.sin(2 * np.pi * (f_inharm * 1.0016) * t) * np.exp(-t * decay)
            tone += (amp * (string_a + 0.7 * string_b)).astype(np.float32)

        chord += hammer + tone

    # 3. Wooden soundboard resonance (body formant peak at 520 Hz)
    try:
        b, a = signal.iirpeak(520.0, 2.2, sr)
        body = signal.lfilter(b, a, chord).astype(np.float32)
        out = (0.76 * chord + 0.24 * body) * 0.35
    except Exception:
        out = chord * 0.35

    return np.tanh(1.22 * out).astype(np.float32)


def _synthesize_bowed_strings(freqs: list[float], duration_s: float, sr: int = 48000) -> np.ndarray:
    """Orchestral string ensemble model: Expressive bowing attack + 3-voice unison chorus + 280Hz cello body cavity formant."""
    n_samples = int(duration_s * sr)
    t = np.arange(n_samples) / sr
    strings = np.zeros(n_samples, dtype=np.float32)

    # Expressive bowing envelope (280ms crescendo, smooth release)
    attack = np.minimum(t / 0.28, 1.0)
    release = np.minimum((duration_s - t) / 0.35, 1.0)
    envelope = (attack * release).astype(np.float32)

    # Natural vibrato LFO (5.2 Hz) with gradual intensity ramp
    vibrato = (1.0 + 0.015 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(t / 0.35, 1.0)).astype(np.float32)

    for f in freqs:
        # Multi-voice chorus (3 players per section) with warm harmonics
        p1 = np.sin(2 * np.pi * (f * vibrato) * t) + 0.48 * np.sin(4 * np.pi * (f * vibrato) * t) + 0.22 * np.sin(6 * np.pi * (f * vibrato) * t)
        p2 = np.sin(2 * np.pi * (f * 1.0022 * vibrato) * t) + 0.38 * np.sin(4 * np.pi * (f * 1.0022 * vibrato) * t)
        p3 = np.sin(2 * np.pi * (f * 0.9978 * vibrato) * t) + 0.38 * np.sin(4 * np.pi * (f * 0.9978 * vibrato) * t)
        strings += ((p1 + p2 + p3) * envelope).astype(np.float32)

    # Cello/violin wooden body cavity resonance at 280 Hz
    try:
        b, a = signal.iirpeak(280.0, 2.0, sr)
        body = signal.lfilter(b, a, strings).astype(np.float32)
        res = (0.75 * strings + 0.25 * body) * 0.12
    except Exception:
        res = strings * 0.12

    return np.tanh(1.2 * res).astype(np.float32)


def _karplus_strong_guitar(freq: float, duration_s: float, sr: int = 48000, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Karplus-Strong physical plucked string model with wooden soundbox Helmholtz resonance at 110 Hz."""
    if rng is None:
        rng = np.random.RandomState(42)
    n_samples = int(duration_s * sr)
    period = max(2, int(sr / max(20.0, freq)))
    # Initial noise burst for pluck transient
    ring = rng.uniform(-0.85, 0.85, period).astype(np.float32)
    output = np.zeros(n_samples, dtype=np.float32)
    output[:period] = ring

    decay = 0.994
    for i in range(period, n_samples):
        # Low-pass filter averaging adjacent string samples
        output[i] = 0.5 * (output[i - period] + output[i - period - 1]) * decay

    # Guitar wooden body Helmholtz resonance at 110 Hz
    try:
        b, a = signal.iirpeak(110.0, 2.5, sr)
        body = signal.lfilter(b, a, output).astype(np.float32)
        res = (0.8 * output + 0.2 * body) * 0.35
    except Exception:
        res = output * 0.35

    return np.tanh(1.18 * res).astype(np.float32)


def _apply_studio_reverb(audio: np.ndarray, sr: int = 48000, room_decay: float = 0.8, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Fast convolution studio room reverberator for organic acoustic depth."""
    if rng is None:
        rng = np.random.RandomState(42)
    ir_len = int(min(0.7, room_decay) * sr)
    t_ir = np.linspace(0, 0.7, ir_len, endpoint=False)
    ir = (rng.uniform(-0.5, 0.5, ir_len) * np.exp(-t_ir * (3.8 / max(0.2, room_decay)))).astype(np.float32)
    # Filter IR so high frequencies absorb naturally
    b = np.ones(6, dtype=np.float32) / 6.0
    ir_smooth = signal.lfilter(b, [1.0], ir)

    wet = signal.fftconvolve(audio, ir_smooth, mode="same")
    # 82% direct dry sound + 18% acoustic room ambience
    return (0.82 * audio + 0.18 * wet).astype(np.float32)


# ─── MockMusicAdapter Class ───

class MockMusicAdapter(MusicAdapter):
    """
    Emotionally expressive, situation-aware music generation adapter.
    Synthesizes rich acoustic instruments matching the psychological mood,
    cinematic scene, or situational context of any text prompt.
    """

    def __init__(self, output_dir: Path | str = "/tmp/incubrix_mock"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _analyze_situation(self, prompt: str) -> dict:
        """Extract situational context, emotion, and musical parameters from text with word-boundary matching."""
        import re

        def _match_any(words: list[str]) -> bool:
            pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
            return bool(re.search(pattern, prompt, re.IGNORECASE))

        # 1. Primary emotional feeling takes first precedence
        if _match_any(["sad", "melancholy", "melancholic", "lonely", "rain", "rainy", "grief", "cry", "crying", "tears", "heartbreak", "heartbroken", "sorrow", "pain", "miss", "lost", "empty", "silent", "memory", "depressed"]):
            return {
                "genre": "cinematic",
                "situation": "melancholy",
                "mood": "Heartfelt Melancholy & Emotion",
                "target_bpm": 72,
                "scale": [146.83, 164.81, 174.61, 196.0, 220.0, 233.08, 261.63, 293.66],  # D Natural Minor
                "chord_roots": [146.83, 116.54, 174.61, 130.81],  # Dm, Bb, F, C
                "chord_types": ["minor", "major", "major", "major"],
                "instrument": "piano_strings",
                "drum_style": "minimal_brush",
                "reverb_decay": 1.2,
            }

        if _match_any(["epic", "battle", "warrior", "hero", "heroic", "glory", "triumph", "triumphant", "majestic", "empire", "crown", "victory", "clash"]):
            return {
                "genre": "cinematic",
                "situation": "epic_triumph",
                "mood": "Epic Heroic & Triumphant",
                "target_bpm": 118,
                "scale": [146.83, 164.81, 174.61, 196.0, 220.0, 233.08, 261.63, 293.66],  # D Minor
                "chord_roots": [146.83, 116.54, 196.0, 110.0],  # Dm, Bb, Gm, Am
                "chord_types": ["minor", "major", "minor", "minor"],
                "instrument": "orchestral_brass_strings",
                "drum_style": "taiko_march",
                "reverb_decay": 1.4,
            }

        if _match_any(["peace", "peaceful", "calm", "zen", "meditation", "meditate", "breathing", "ocean", "gentle", "healing", "sunrise", "river", "ambient", "drone", "ethereal"]):
            return {
                "genre": "ambient",
                "situation": "peaceful_nature",
                "mood": "Serene Peace & Tranquility",
                "target_bpm": 64,
                "scale": [196.0, 220.0, 246.94, 293.66, 329.63, 392.0, 440.0, 493.88],  # G Pentatonic
                "chord_roots": [196.0, 130.81, 164.81, 146.83],  # G, C, Em, D
                "chord_types": ["major7", "major7", "minor7", "sus4"],
                "instrument": "acoustic_guitar_pad",
                "drum_style": "ambient_swell",
                "reverb_decay": 1.6,
            }

        if _match_any(["tense", "suspense", "thriller", "mystery", "horror", "creepy", "shadow", "shadows", "fear", "danger", "nervous", "clock", "chase", "escape", "detective"]):
            return {
                "genre": "tech",
                "situation": "tense_thriller",
                "mood": "Dark Suspense & Tension",
                "target_bpm": 96,
                "scale": [123.47, 138.59, 146.83, 164.81, 174.61, 196.0, 220.0, 246.94],  # B Harmonic Minor
                "chord_roots": [123.47, 174.61, 130.81, 164.81],  # Bm, Fdim, C, Em
                "chord_types": ["minor", "dim", "major", "minor"],
                "instrument": "ticking_plucks_sub",
                "drum_style": "ticking_sub",
                "reverb_decay": 0.9,
            }

        if _match_any(["romantic", "romance", "love", "cozy", "intimate", "candle", "candlelight", "date", "sweet", "tender", "passion", "soul"]):
            return {
                "genre": "acoustic",
                "situation": "romantic_acoustic",
                "mood": "Warm Romance & Intimacy",
                "target_bpm": 86,
                "scale": [196.0, 220.0, 246.94, 261.63, 293.66, 329.63, 369.99, 392.0],  # G Major
                "chord_roots": [196.0, 164.81, 130.81, 146.83],  # G, Em, C, D
                "chord_types": ["major", "minor", "major", "major"],
                "instrument": "plucked_guitar_piano",
                "drum_style": "soft_acoustic_pop",
                "reverb_decay": 0.8,
            }

        # 2. Distinct musical styles
        if _match_any(["hip-hop", "hiphop", "trap", "urban", "808", "rap", "drill"]):
            return {
                "genre": "hiphop",
                "situation": "hiphop_trap",
                "mood": "Heavy Urban Hip-Hop",
                "target_bpm": 92,
                "scale": [164.81, 196.0, 220.0, 246.94, 293.66, 329.63, 392.0, 440.0],  # E Minor Pentatonic
                "chord_roots": [82.41, 110.0, 130.81, 123.47],  # Em, Am, C, B7
                "chord_types": ["minor", "minor", "major", "dom7"],
                "instrument": "dark_piano_sub",
                "drum_style": "808_trap",
                "reverb_decay": 0.6,
            }

        if _match_any(["lofi", "lo-fi", "study", "vinyl"]):
            return {
                "genre": "lofi",
                "situation": "lofi_study",
                "mood": "Chill Nostalgic Lo-Fi",
                "target_bpm": 78,
                "scale": [174.61, 196.0, 220.0, 246.94, 261.63, 293.66, 329.63, 349.23],  # F Major 7
                "chord_roots": [174.61, 220.0, 146.83, 196.0],  # Fmaj7, Am7, Dm7, Gm7
                "chord_types": ["major7", "minor7", "minor7", "minor7"],
                "instrument": "rhodes_piano",
                "drum_style": "dusty_boombap",
                "reverb_decay": 0.8,
            }

        if _match_any(["jazz", "swing", "blues", "saxophone", "trumpet", "cocktail", "lounge", "bebop"]):
            return {
                "genre": "jazz",
                "situation": "smooth_jazz",
                "mood": "Sophisticated Lounge Jazz",
                "target_bpm": 108,
                "scale": [233.08, 261.63, 293.66, 311.13, 349.23, 392.0, 415.3, 466.16],  # Bb Mixolydian
                "chord_roots": [116.54, 155.56, 130.81, 174.61],  # Bb7, Eb7, C7, F7
                "chord_types": ["dom7", "dom7", "min7", "dom7"],
                "instrument": "rhodes_upright",
                "drum_style": "brushed_swing",
                "reverb_decay": 0.7,
            }

        if _match_any(["rock", "metal", "punk", "power", "grunge", "electric", "heavy", "workout", "rebel", "speed"]):
            return {
                "genre": "rock",
                "situation": "heavy_rock",
                "mood": "High-Voltage Rock Energy",
                "target_bpm": 140,
                "scale": [164.81, 196.0, 220.0, 246.94, 293.66, 329.63, 392.0, 440.0],  # E Minor
                "chord_roots": [82.41, 98.0, 130.81, 146.83],  # Em, G, C, D
                "chord_types": ["power", "power", "power", "power"],
                "instrument": "power_chords_bass",
                "drum_style": "driving_rock",
                "reverb_decay": 0.7,
            }

        if _match_any(["acoustic", "folk", "campfire", "unplugged", "guitar"]):
            return {
                "genre": "acoustic",
                "situation": "acoustic_folk",
                "mood": "Organic Acoustic Folk",
                "target_bpm": 96,
                "scale": [196.0, 220.0, 246.94, 261.63, 293.66, 329.63, 369.99, 392.0],  # G Major
                "chord_roots": [196.0, 164.81, 130.81, 146.83],  # G, Em, C, D
                "chord_types": ["major", "minor", "major", "major"],
                "instrument": "plucked_guitar_piano",
                "drum_style": "soft_acoustic_pop",
                "reverb_decay": 0.8,
            }

        # 11. Corporate / Upbeat / Tech / Inspiring / Showcase (Default)
        return {
            "genre": "corporate",
            "situation": "corporate_upbeat",
            "mood": "Inspiring Modern Showcase",
            "target_bpm": 120,
            "scale": [130.81, 146.83, 164.81, 174.61, 196.0, 220.0, 246.94, 261.63],  # C Major
            "chord_roots": [130.81, 196.0, 110.0, 174.61],  # C, G, Am, F
            "chord_types": ["major", "major", "minor", "major"],
            "instrument": "acoustic_piano_chimes",
            "drum_style": "pop_acoustic",
            "reverb_decay": 0.8,
        }

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
        """Generate situation-adapted acoustic stems and composite broadcast master."""
        start_time = time.time()

        if seed is not None:
            seed_val = seed
        else:
            seed_val = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16) % 1000000

        np.random.seed(seed_val)
        rng = np.random.RandomState(seed_val)

        # 1. Semantic Situation & Emotion Extraction
        ctx = self._analyze_situation(prompt)
        genre = ctx["genre"]
        situation = ctx["situation"]
        target_bpm = bpm or ctx["target_bpm"]
        scale = ctx["scale"]
        chord_roots = ctx["chord_roots"]
        reverb_decay = ctx["reverb_decay"]

        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
        beat_interval = 60.0 / target_bpm
        beat_samples = max(1, int(beat_interval * sample_rate))
        bar_samples = beat_samples * 4

        drums = np.zeros(num_samples, dtype=np.float32)
        bass = np.zeros(num_samples, dtype=np.float32)
        harmony = np.zeros(num_samples, dtype=np.float32)
        melody = np.zeros(num_samples, dtype=np.float32)

        # ── 2. Drum Synthesis Tailored to Emotional Situation ──
        drum_style = ctx["drum_style"]

        if drum_style == "taiko_march":
            # Resonant taiko thunder drum on bar 1 and 3 with deep sub rumble
            for pos in range(0, num_samples, bar_samples):
                k_len = min(pos + int(0.7 * sample_rate), num_samples)
                kt = np.arange(k_len - pos) / sample_rate
                # Acoustic taiko pitch-envelope sweep
                drums[pos:k_len] += (0.45 * np.sin(2 * np.pi * (95 * np.exp(-kt * 8) + 42) * kt) * np.exp(-kt * 4.2)).astype(np.float32)
                # Marching snare rim hit on beat 3
                b3 = pos + beat_samples * 2
                if b3 < num_samples:
                    r_len = min(b3 + int(0.12 * sample_rate), num_samples)
                    rt = np.arange(r_len - b3) / sample_rate
                    drums[b3:r_len] += (0.25 * np.sin(2 * np.pi * 175 * rt) * np.exp(-rt * 22)).astype(np.float32)

        elif drum_style == "808_trap":
            for bar_pos in range(0, num_samples, bar_samples):
                # 808 Sub Kick on beat 1
                k_len = min(bar_pos + int(0.3 * sample_rate), num_samples)
                kt = np.arange(k_len - bar_pos) / sample_rate
                drums[bar_pos:k_len] += (0.45 * np.sin(2 * np.pi * (135 * np.exp(-kt * 22) + 38) * kt) * np.exp(-kt * 6)).astype(np.float32)
                # Snappy trap snares on 2 and 4
                for b_idx in [1, 3]:
                    s_pos = bar_pos + b_idx * beat_samples
                    if s_pos < num_samples:
                        s_len = min(s_pos + int(0.1 * sample_rate), num_samples)
                        st = np.arange(s_len - s_pos) / sample_rate
                        drums[s_pos:s_len] += (rng.uniform(-0.25, 0.25, len(st)) * np.exp(-st * 24)).astype(np.float32)
                # Trap hi-hat triplets
                triplet = beat_samples // 3
                for tp in range(bar_pos, min(bar_pos + bar_samples, num_samples), triplet):
                    h_len = min(tp + int(0.02 * sample_rate), num_samples)
                    ht = np.arange(h_len - tp) / sample_rate
                    drums[tp:h_len] += (rng.uniform(-0.1, 0.1, len(ht)) * np.exp(-ht * 75)).astype(np.float32)

        elif drum_style in ("minimal_brush", "brushed_swing"):
            # Delicate brushed jazz / acoustic swing
            for pos in range(0, num_samples, beat_samples):
                r_len = min(pos + int(0.14 * sample_rate), num_samples)
                rt = np.arange(r_len - pos) / sample_rate
                drums[pos:r_len] += (rng.uniform(-0.07, 0.07, len(rt)) * np.exp(-rt * 11)).astype(np.float32)
                # Soft kick on beats 1 and 3
                if (pos // beat_samples) % 4 in (0, 2):
                    k_len = min(pos + int(0.06 * sample_rate), num_samples)
                    kt = np.arange(k_len - pos) / sample_rate
                    drums[pos:k_len] += (0.18 * np.sin(2 * np.pi * 65 * kt) * np.exp(-kt * 28)).astype(np.float32)

        elif drum_style == "ambient_swell":
            # No harsh percussion; slow organic swell
            swell_span = bar_samples * 2
            for pos in range(0, num_samples, swell_span):
                s_len = min(pos + swell_span, num_samples)
                st = np.arange(s_len - pos) / sample_rate
                env = np.sin(np.pi * np.linspace(0, 1, len(st))).astype(np.float32)
                noise = rng.uniform(-0.03, 0.03, len(st)).astype(np.float32)
                drums[pos:s_len] += noise * env

        else:
            # Studio Acoustic Pop / Driving Rock / Confident Corporate
            for pos in range(0, num_samples, beat_samples):
                b_num = (pos // beat_samples) % 4
                # Multi-layer acoustic kick on 1 and 3 (pitch-envelope drop + beater transient)
                if b_num in (0, 2):
                    k_len = min(pos + int(0.12 * sample_rate), num_samples)
                    kt = np.arange(k_len - pos) / sample_rate
                    f_sweep = 48.0 + 95.0 * np.exp(-kt * 32.0)
                    body = 0.40 * np.sin(2 * np.pi * f_sweep * kt) * np.exp(-kt * 18.0)
                    beater = rng.uniform(-0.08, 0.08, len(kt)) * np.exp(-kt * 95.0)
                    drums[pos:k_len] += (body + beater).astype(np.float32)
                # Multi-layer acoustic snare on 2 and 4 (wooden shell resonance + metal wire sizzle)
                if b_num in (1, 3):
                    s_len = min(pos + int(0.16 * sample_rate), num_samples)
                    st = np.arange(s_len - pos) / sample_rate
                    shell = 0.22 * np.sin(2 * np.pi * 185.0 * st) * np.exp(-st * 26.0)
                    wires = rng.uniform(-0.18, 0.18, len(st)) * np.exp(-st * 22.0)
                    drums[pos:s_len] += (shell + wires).astype(np.float32)
                # Studio acoustic hi-hat on eighth-note upbeat
                hh_pos = pos + beat_samples // 2
                if hh_pos < num_samples:
                    h_len = min(hh_pos + int(0.025 * sample_rate), num_samples)
                    ht = np.arange(h_len - hh_pos) / sample_rate
                    metal = rng.uniform(-0.06, 0.06, len(ht)) * np.exp(-ht * 80.0)
                    drums[hh_pos:h_len] += metal.astype(np.float32)

        # ── 3. Acoustic Bass Synthesis ──
        for bar_idx, bar_pos in enumerate(range(0, num_samples, bar_samples)):
            root_f = chord_roots[bar_idx % len(chord_roots)]
            b_end = min(bar_pos + bar_samples, num_samples)
            bar_len = b_end - bar_pos

            if situation in ("melancholy", "epic_triumph", "peaceful_nature"):
                # Warm deep acoustic cello / sub-octave bass note per bar with subtle human swell
                bt = np.arange(bar_len) / sample_rate
                f_bass = root_f * 0.5
                sub_tone = 0.3 * np.sin(2 * np.pi * f_bass * bt) + 0.15 * np.sin(4 * np.pi * f_bass * bt)
                swell = (0.75 + 0.25 * np.sin(np.pi * (bt / (bar_len / sample_rate)))).astype(np.float32)
                bass[bar_pos:b_end] += (sub_tone * swell).astype(np.float32)

            elif situation == "smooth_jazz":
                # Walking bass (quarter note stepping tones)
                walk_steps = [root_f * 0.5, root_f * 0.625, root_f * 0.75, root_f * 0.625]
                for q_idx, q_pos in enumerate(range(bar_pos, b_end, beat_samples)):
                    q_len = min(q_pos + int(beat_samples * 0.85), num_samples)
                    qt = np.arange(q_len - q_pos) / sample_rate
                    f_walk = walk_steps[q_idx % 4]
                    bass[q_pos:q_len] += (0.28 * np.sin(2 * np.pi * f_walk * qt) * np.exp(-qt * 3.8)).astype(np.float32)

            else:
                # Root-fifth acoustic rhythmic bass
                for s_idx, s_pos in enumerate(range(bar_pos, b_end, beat_samples)):
                    s_len = min(s_pos + int(beat_samples * 0.75), num_samples)
                    st = np.arange(s_len - s_pos) / sample_rate
                    p_bass = (root_f * 0.5) if s_idx % 2 == 0 else (root_f * 0.75)
                    bass[s_pos:s_len] += (0.28 * np.sin(2 * np.pi * p_bass * st) * np.exp(-st * 4.8)).astype(np.float32)

        # ── 4. Harmony Synthesis with Physical Modeling ──
        for bar_idx, bar_pos in enumerate(range(0, num_samples, bar_samples)):
            root_f = chord_roots[bar_idx % len(chord_roots)]
            h_end = min(bar_pos + bar_samples, num_samples)
            bar_len_s = (h_end - bar_pos) / sample_rate

            # Build harmonic triad / 7th voicing
            c_type = ctx["chord_types"][bar_idx % len(ctx["chord_types"])]
            if c_type == "minor":
                c_freqs = [root_f, root_f * 1.189, root_f * 1.498]
            elif c_type == "minor7":
                c_freqs = [root_f, root_f * 1.189, root_f * 1.498, root_f * 1.782]
            elif c_type == "major7":
                c_freqs = [root_f, root_f * 1.26, root_f * 1.498, root_f * 1.887]
            elif c_type == "dom7":
                c_freqs = [root_f, root_f * 1.26, root_f * 1.498, root_f * 1.782]
            else:
                c_freqs = [root_f, root_f * 1.26, root_f * 1.498]

            if ctx["instrument"] in ("piano_strings", "acoustic_piano_chimes"):
                # Real felt-hammer acoustic grand piano chord + subtle bowed cello underlay
                piano_block = _synthesize_piano_chord([f * 1.5 for f in c_freqs], bar_len_s, sample_rate, rng=rng)
                strings_block = _synthesize_bowed_strings([f * 0.75 for f in c_freqs], bar_len_s, sample_rate)
                harmony[bar_pos:h_end] += (piano_block + strings_block)[: (h_end - bar_pos)]

            elif ctx["instrument"] == "orchestral_brass_strings":
                # Full symphonic bowed string section with emotional crescendo
                strings_block = _synthesize_bowed_strings([f * 1.0 for f in c_freqs], bar_len_s, sample_rate)
                harmony[bar_pos:h_end] += strings_block[: (h_end - bar_pos)]

            elif ctx["instrument"] in ("acoustic_guitar_pad", "plucked_guitar_piano"):
                # Karplus-Strong physical plucked guitar arpeggio across bar
                step_pl = beat_samples
                for pl_idx, pl_pos in enumerate(range(bar_pos, h_end, step_pl)):
                    f_pl = c_freqs[pl_idx % len(c_freqs)] * 2.0
                    pl_audio = _karplus_strong_guitar(f_pl, 1.2, sample_rate, rng=rng)
                    p_end = min(pl_pos + len(pl_audio), num_samples)
                    harmony[pl_pos:p_end] += (pl_audio[: (p_end - pl_pos)] * 0.28).astype(np.float32)

            else:
                # Warm vintage Rhodes electric piano chords
                piano_block = _synthesize_piano_chord([f * 1.25 for f in c_freqs], bar_len_s, sample_rate, rng=rng)
                harmony[bar_pos:h_end] += piano_block[: (h_end - bar_pos)]

        # ── 5. Emotionally Expressive Melody Synthesis ──
        if situation == "melancholy":
            # Slow, emotional solo violin / piano phrase with pauses and weeping vibrato
            step_len = beat_samples * 2
            mel_pattern = [0, 2, 4, 3, 2, 0, 1, 0]
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                if s_idx % 4 != 3:  # Emotional breathing pause
                    s_len = min(s_pos + int(step_len * 0.92), num_samples)
                    mt = np.arange(s_len - s_pos) / sample_rate
                    f_note = scale[mel_pattern[s_idx % len(mel_pattern)]] * 2.0
                    # Expressive slow vibrato that deepens as the note sustains
                    vib = 1.0 + 0.016 * np.sin(2 * np.pi * 5.2 * mt) * np.minimum(mt / 0.5, 1.0)
                    tone = np.sin(2 * np.pi * (f_note * vib) * mt) + 0.3 * np.sin(4 * np.pi * (f_note * vib) * mt)
                    env = np.sin(np.pi * (mt / (len(mt) / sample_rate))) ** 0.8
                    melody[s_pos:s_len] += (tone * env * 0.18).astype(np.float32)

        elif situation == "epic_triumph":
            # Soaring heroic trumpet / violin fanfare with confident phrasing
            step_len = beat_samples
            mel_pattern = [0, 0, 2, 4, 7, 5, 4, 2]
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                s_len = min(s_pos + int(step_len * 0.85), num_samples)
                mt = np.arange(s_len - s_pos) / sample_rate
                f_note = scale[mel_pattern[s_idx % len(mel_pattern)]] * 2.5
                lead = (0.22 * np.sin(2 * np.pi * f_note * mt) * np.exp(-mt * 4.5)).astype(np.float32)
                melody[s_pos:s_len] += lead

        elif situation == "peaceful_nature":
            # Delicate pentatonic acoustic guitar notes
            step_len = beat_samples
            pattern = [0, 4, 2, 5, 3, 7, 4, 2]
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                if s_idx % 2 == 0:
                    f_note = scale[pattern[s_idx % len(pattern)]] * 2.0
                    pluck = _karplus_strong_guitar(f_note, 1.4, sample_rate, rng=rng)
                    p_end = min(s_pos + len(pluck), num_samples)
                    melody[s_pos:p_end] += (pluck[: (p_end - s_pos)] * 0.22).astype(np.float32)

        else:
            # Catchy acoustic lead hook with humanized dynamics
            step_len = beat_samples // 2
            pattern = [0, 2, 4, 7, 4, 2, 0, 4]
            for s_idx, s_pos in enumerate(range(0, num_samples, step_len)):
                s_len = min(s_pos + int(step_len * 0.72), num_samples)
                mt = np.arange(s_len - s_pos) / sample_rate
                f_note = scale[pattern[s_idx % len(pattern)]] * 2.5
                lead = (0.16 * np.sin(2 * np.pi * f_note * mt) * np.exp(-mt * 6.5)).astype(np.float32)
                melody[s_pos:s_len] += lead

        # ── 6. Studio Room Reverberation & Headroom ──
        harmony = _apply_studio_reverb(harmony, sample_rate, reverb_decay, rng=rng)
        melody = _apply_studio_reverb(melody, sample_rate, reverb_decay, rng=rng)

        # ── 7. Musical Bar-Aligned Ending ──
        fade_in = int(sample_rate * 0.05)
        fade_out = int(sample_rate * 0.35)
        for stem in [drums, bass, harmony, melody]:
            if fade_in > 0:
                stem[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
            if fade_out > 0:
                stem[-fade_out:] *= (np.linspace(1, 0, fade_out, dtype=np.float32) ** 2)

        # ── 8. Sum Stems into Mix with Lossless Summation Guarantee ──
        mix = drums + bass + harmony + melody
        peak = np.max(np.abs(mix))
        if peak > 0:
            scale_fac = 0.85 / peak
            mix *= scale_fac
            drums *= scale_fac
            bass *= scale_fac
            harmony *= scale_fac
            melody *= scale_fac

        # ── 9. Save Files ──
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
            model_name=f"mock-{genre.replace('-', '')}-engine",
            model_version="4.0.0",
            seed=seed,
            cost_estimate_usd=0.0,
            generation_time_seconds=round(generation_time, 3),
            warnings=[f"Composed for situational emotion: '{ctx['mood']}' ({genre.upper()})."],
            metadata={
                "stems_dir": str(stems_dir),
                "genre": genre,
                "situation": situation,
                "mood": ctx["mood"],
            },
        )

    def is_available(self) -> bool:
        return True

    def get_info(self) -> dict:
        return {
            "name": "MockMusicAdapter",
            "version": "4.0.0",
            "description": "Physical modeling synthesis engine (Karplus-Strong guitar, acoustic grand piano, bowed strings, room reverb) with emotional situation understanding.",
            "license": "MIT",
            "cost_per_minute": 0.0,
        }
