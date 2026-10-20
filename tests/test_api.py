"""
Tests for the AI Background Music Engine.
Run with: pytest tests/ -v
All tests use the mock adapter (no API keys or GPU needed).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

# Force mock adapter for tests
os.environ["MUSIC_ADAPTER"] = "mock"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_request():
    """Minimal valid music generation request."""
    return {
        "prompt": "calm acoustic guitar for a product video",
        "duration_seconds": 15.0,
    }


# ─── Health Endpoint Tests ───


class TestHealthEndpoints:
    """Integration Standard #8: /health and /ready endpoints."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_ready_returns_ok(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# ─── Job Creation Tests ───


class TestJobCreation:
    """Integration Standard #2, #3: POST /api/v1/music/jobs returns 202."""

    def test_create_job_returns_202(self, client, sample_request):
        response = client.post("/api/v1/music/jobs", json=sample_request)
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] in ["pending", "processing", "completed"]

    def test_create_job_with_all_fields(self, client):
        request = {
            "prompt": "energetic electronic music with synth pads",
            "duration_seconds": 30.0,
            "style": "electronic",
            "mood": "energetic",
            "bpm": 128,
            "energy": 0.8,
            "output_format": "wav",
            "loudness_lufs": -14.0,
            "include_stems": True,
        }
        response = client.post("/api/v1/music/jobs", json=request)
        assert response.status_code == 202
        data = response.json()
        assert data["request"]["duration_s"] == 30.0

    def test_create_job_with_brief_format(self, client):
        """Test the brief's exact JSON format with duration_s, tempo_bpm."""
        request = {
            "prompt": "upbeat corporate music for product demo",
            "duration_s": 20.0,
            "tempo_bpm": 120,
        }
        response = client.post("/api/v1/music/jobs", json=request)
        assert response.status_code == 202
        data = response.json()
        assert data["request"]["duration_s"] == 20.0

    def test_create_job_with_sections(self, client):
        request = {
            "prompt": "cinematic orchestral score",
            "duration_seconds": 45.0,
            "sections": [
                {"start_seconds": 0.0, "end_seconds": 15.0, "mood": "calm"},
                {"start_seconds": 15.0, "end_seconds": 30.0, "mood": "dramatic"},
                {"start_seconds": 30.0, "end_seconds": 45.0, "mood": "uplifting"},
            ],
        }
        response = client.post("/api/v1/music/jobs", json=request)
        assert response.status_code == 202

    def test_invalid_duration_rejected(self, client):
        request = {
            "prompt": "test",
            "duration_seconds": -5.0,
        }
        response = client.post("/api/v1/music/jobs", json=request)
        assert response.status_code == 422  # Validation error

    def test_missing_prompt_rejected(self, client):
        request = {
            "duration_seconds": 10.0,
        }
        response = client.post("/api/v1/music/jobs", json=request)
        assert response.status_code == 422

    def test_duration_over_300s_rejected(self, client):
        request = {
            "prompt": "test music",
            "duration_seconds": 500.0,
        }
        response = client.post("/api/v1/music/jobs", json=request)
        assert response.status_code == 422


# ─── Idempotency Tests ───


class TestIdempotency:
    """Integration Standard #4: Idempotency-Key header."""

    def test_same_key_returns_same_job(self, client, sample_request):
        key = "test-idempotency-key-001"
        headers = {"Idempotency-Key": key}

        r1 = client.post("/api/v1/music/jobs", json=sample_request, headers=headers)
        r2 = client.post("/api/v1/music/jobs", json=sample_request, headers=headers)

        assert r1.status_code == 202
        assert r2.status_code == 202
        assert r1.json()["job_id"] == r2.json()["job_id"]

    def test_different_keys_create_different_jobs(self, client, sample_request):
        r1 = client.post(
            "/api/v1/music/jobs",
            json=sample_request,
            headers={"Idempotency-Key": "key-a"},
        )
        r2 = client.post(
            "/api/v1/music/jobs",
            json=sample_request,
            headers={"Idempotency-Key": "key-b"},
        )

        assert r1.json()["job_id"] != r2.json()["job_id"]


# ─── Job Status Tests ───


class TestJobStatus:
    """Integration Standard #3: GET /api/v1/jobs/{job_id}."""

    def test_get_existing_job(self, client, sample_request):
        create_response = client.post("/api/v1/music/jobs", json=sample_request)
        job_id = create_response.json()["job_id"]

        status_response = client.get(f"/api/v1/music/jobs/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["job_id"] == job_id

    def test_get_nonexistent_job_returns_404(self, client):
        response = client.get("/api/v1/music/jobs/nonexistent-id")
        assert response.status_code == 404

    def test_list_jobs(self, client, sample_request):
        # Create a couple of jobs
        client.post("/api/v1/music/jobs", json=sample_request)
        client.post("/api/v1/music/jobs", json=sample_request)

        response = client.get("/api/v1/music/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert data["total"] >= 2


# ─── Correlation ID Tests ───


class TestCorrelationId:
    """Integration Standard #8: X-Correlation-Id in responses."""

    def test_correlation_id_returned(self, client):
        response = client.get("/health")
        assert "x-correlation-id" in response.headers

    def test_custom_correlation_id_echoed(self, client):
        custom_id = "my-trace-id-123"
        response = client.get(
            "/health",
            headers={"X-Correlation-Id": custom_id},
        )
        assert response.headers["x-correlation-id"] == custom_id


# ─── Error Format Tests ───


class TestErrorFormat:
    """Integration Standard #5: JSON error format."""

    def test_404_returns_json_error(self, client):
        response = client.get("/api/v1/music/jobs/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


# ─── Audio Processing Unit Tests ───


class TestLoudnessNormalization:
    """Must-have #5: Loudness at target LUFS."""

    def test_normalize_to_target(self):
        from app.audio.loudness import normalize_loudness, measure_loudness

        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        normalized = normalize_loudness(audio, sr, target_lufs=-16.0)
        result = measure_loudness(normalized, sr)

        # Should be within 1 LU of target
        assert abs(result["loudness_lufs"] - (-16.0)) <= 1.0

    def test_true_peak_below_limit(self):
        from app.audio.loudness import normalize_loudness, measure_loudness

        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        normalized = normalize_loudness(audio, sr, target_lufs=-16.0, max_true_peak_dbtp=-1.0)
        result = measure_loudness(normalized, sr)

        assert result["true_peak_dbtp"] <= -1.0


class TestExactLength:
    """Must-have #1: Exact length within 0.05s."""

    def test_trim_to_exact_length(self):
        from app.audio.length import trim_or_pad_to_exact_length

        sr = 48000
        audio = np.random.randn(sr * 20).astype(np.float32)  # 20s
        target_samples = sr * 15  # 15s

        result = trim_or_pad_to_exact_length(audio, target_samples, sr)
        assert len(result) == target_samples

    def test_pad_short_audio(self):
        from app.audio.length import trim_or_pad_to_exact_length

        sr = 48000
        audio = np.random.randn(sr * 5).astype(np.float32)  # 5s
        target_samples = sr * 10  # 10s

        result = trim_or_pad_to_exact_length(audio, target_samples, sr)
        assert len(result) == target_samples

    def test_musical_ending(self):
        from app.audio.length import trim_or_pad_to_exact_length

        sr = 48000
        audio = np.ones(sr * 10, dtype=np.float32)  # 10s of constant
        target_samples = sr * 10

        result = trim_or_pad_to_exact_length(audio, target_samples, sr, musical_ending=True)

        # Last sample should be near zero (faded out)
        assert abs(result[-1]) < 0.01


class TestStemVerification:
    """Must-have #2: Stems sum back to mix."""

    def test_stems_sum_equals_mix(self):
        from app.audio.stems import verify_stems_sum_to_mix

        sr = 48000
        duration = 2.0
        n = int(sr * duration)

        # Create stems that sum to a known mix
        stem1 = np.random.randn(n).astype(np.float32) * 0.3
        stem2 = np.random.randn(n).astype(np.float32) * 0.3
        stem3 = np.random.randn(n).astype(np.float32) * 0.3
        mix = stem1 + stem2 + stem3

        with tempfile.TemporaryDirectory() as tmpdir:
            mix_path = Path(tmpdir) / "mix.wav"
            s1_path = Path(tmpdir) / "stem1.wav"
            s2_path = Path(tmpdir) / "stem2.wav"
            s3_path = Path(tmpdir) / "stem3.wav"

            sf.write(str(mix_path), mix, sr, subtype="FLOAT")
            sf.write(str(s1_path), stem1, sr, subtype="FLOAT")
            sf.write(str(s2_path), stem2, sr, subtype="FLOAT")
            sf.write(str(s3_path), stem3, sr, subtype="FLOAT")

            result = verify_stems_sum_to_mix(
                mix_path, [s1_path, s2_path, s3_path]
            )

            assert result["passes"]
            assert result["difference_db"] < -40


class TestMockAdapter:
    """Test mock adapter produces valid audio."""

    def test_mock_generates_exact_duration(self):
        from app.adapters.mock_adapter import MockMusicAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = MockMusicAdapter(output_dir=tmpdir)
            result = adapter.generate(
                prompt="test music",
                duration_seconds=10.0,
                sample_rate=48000,
            )

            audio, sr = sf.read(str(result.audio_path))
            actual_duration = len(audio) / sr

            assert abs(actual_duration - 10.0) < 0.05

    def test_mock_is_available(self):
        from app.adapters.mock_adapter import MockMusicAdapter
        adapter = MockMusicAdapter()
        assert adapter.is_available()

    def test_mock_deterministic_with_seed(self):
        from app.adapters.mock_adapter import MockMusicAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = MockMusicAdapter(output_dir=tmpdir)
            r1 = adapter.generate("test", 5.0, seed=42, sample_rate=48000)
            r2 = adapter.generate("test", 5.0, seed=42, sample_rate=48000)

            a1, _ = sf.read(str(r1.audio_path))
            a2, _ = sf.read(str(r2.audio_path))
            np.testing.assert_array_almost_equal(a1, a2)


class TestAudioInspection:
    """Test POST /api/v1/audio/analyze endpoint."""

    def test_analyze_audio_endpoint(self, client):
        sr = 44100
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr, subtype="FLOAT")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as audio_file:
                response = client.post(
                    "/api/v1/audio/analyze",
                    files={"file": ("test_track.wav", audio_file, "audio/wav")},
                    data={"normalize": "false", "separate": "true"},
                )

            assert response.status_code == 200
            data = response.json()
            assert "upload_id" in data
            assert abs(data["duration_seconds"] - 2.0) < 0.05
            assert "drums" in data["stems"]
            assert "stem_verification" in data
            assert "difference_db" in data["stem_verification"]
            assert "passes" in data["stem_verification"]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_with_normalization(self, client):
        sr = 44100
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        audio = 0.8 * np.sin(2 * np.pi * 220 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr, subtype="FLOAT")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as audio_file:
                response = client.post(
                    "/api/v1/audio/analyze",
                    files={"file": ("test_norm.wav", audio_file, "audio/wav")},
                    data={"normalize": "true", "target_lufs": "-16.0", "separate": "false"},
                )

            assert response.status_code == 200
            data = response.json()
            assert data["normalization_applied"] is True
            assert abs(data["loudness_lufs"] - (-16.0)) <= 0.5
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_invalid_extension(self, client):
        response = client.post(
            "/api/v1/audio/analyze",
            files={"file": ("document.txt", b"plain text", "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported format" in response.json()["detail"]


class TestRemixEndpoint:
    """Tests for POST /api/v1/music/remix portion-based stem automation."""

    def test_remix_with_portions(self, client):
        resp = client.post(
            "/api/v1/music/jobs",
            json={"prompt": "dynamic upbeat corporate showcase", "duration_seconds": 6.0},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        import time
        for _ in range(20):
            time.sleep(0.3)
            st = client.get(f"/api/v1/music/jobs/{job_id}").json()
            if st["status"] == "completed":
                break

        remix_payload = {
            "job_id": job_id,
            "portions": [
                {
                    "name": "Intro",
                    "start_s": 0.0,
                    "end_s": 3.0,
                    "gains": {"drums": 1.0, "bass": 0.1, "melody": 0.8, "harmony": 0.5},
                },
                {
                    "name": "Drop",
                    "start_s": 3.0,
                    "end_s": 6.0,
                    "gains": {"drums": 1.0, "bass": 1.0, "melody": 0.7, "harmony": 0.6},
                },
            ],
        }
        remix_resp = client.post("/api/v1/music/remix", json=remix_payload)
        assert remix_resp.status_code == 200
        data = remix_resp.json()
        assert data["status"] == "completed"
        assert "custom_mix_url" in data
        assert data["portions_count"] == 2
        assert data["true_peak_dbtp"] <= -1.0
        assert -24.0 <= data["loudness_lufs"] <= -14.0


class TestNatifyAssistant:
    """Tests for POST /api/v1/natify/chat intelligent assistant."""

    def test_natify_theme_command(self, client):
        resp = client.post("/api/v1/natify/chat", json={"message": "Switch to cyber neon mode"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"]["type"] == "SET_THEME"
        assert data["action"]["payload"]["theme"] == "midnight"

    def test_natify_portion_command(self, client):
        resp = client.post("/api/v1/natify/chat", json={"message": "Make drums maximum and bass minimum"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"]["type"] == "SET_PORTION"
        assert data["action"]["payload"]["profile"] == "drumsMaxBassMin"

    def test_natify_music_creation_command(self, client):
        resp = client.post("/api/v1/natify/chat", json={"message": "Compose a high energy tech promo for 20 seconds at 128 bpm"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"]["type"] == "CREATE_MUSIC"
        assert data["action"]["payload"]["tempo_bpm"] == 128
        assert data["action"]["payload"]["duration_seconds"] == 20.0

    def test_natify_voiceover_command(self, client):
        resp = client.post(
            "/api/v1/natify/chat",
            json={"message": "generate a male senior voiceover saying 'Long ago, stories were told around fires'"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"]["type"] == "GENERATE_VOICE"
        assert data["action"]["payload"]["gender"] == "male"
        assert data["action"]["payload"]["age_category"] == "senior"


class TestVoiceoverEndpoint:
    """Tests for POST /api/v1/voice/generate AI voiceover generation."""

    def test_voice_generate_female_young_adult(self, client):
        payload = {
            "text": "Welcome to IncuBrix AI background music engine.",
            "gender": "female",
            "age_category": "young_adult",
        }
        resp = client.post("/api/v1/voice/generate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "audio_url" in data
        assert data["gender"] == "female"
        assert data["age_category"] == "young_adult"
        assert data["duration_seconds"] > 0

    def test_voice_generate_male_senior(self, client):
        payload = {
            "text": "Every legendary journey begins with a single bold step.",
            "gender": "male",
            "age_category": "senior",
        }
        resp = client.post("/api/v1/voice/generate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["gender"] == "male"
        assert data["age_category"] == "senior"
        assert "Christopher" in data["voice_name"]




