# IncuBrix AI Background Music Engine (PS1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)]()

> IncuBrix 2027 Graduate Hiring Assessment — Problem Statement 1 (Software Engineering)  
> Candidate: **Nataraj Chandrasekaran** (PSG College of Technology, Coimbatore)  
> Role Applied: **Associate Tech Partner (ATP)**  
> Repository: [https://github.com/Natarajchandrasekaran/incubrix-ps1](https://github.com/Natarajchandrasekaran/incubrix-ps1)

---

## 1. Overview

The **IncuBrix AI Background Music Engine** is a production-grade, asynchronous REST microservice that generates original, commercially safe instrumental background music for video creators. Every generated track:
- **Matches video duration exactly** within ±0.05 seconds with a musical ending (bar-aligned fade-out, never a hard cut).
- **Separates into real instrument stems** (drums, bass, melody/other) via Demucs, mathematically verified such that the sum of stems equals the mix ($< -40\text{ dB}$ difference).
- **Enforces 100% instrumental delivery** (vocals detected and filtered out).
- **Complies with broadcast audio standards** ($-16\text{ LUFS} \pm 1\text{ LU}$, true peak $\le -1\text{ dBTP}$, $48\text{ kHz}$ WAV).
- **Applies intelligent voiceover ducking** (lowers background music $8\text{--}15\text{ dB}$ smoothly under narration with cosine crossfades).

---

## 2. Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Client / IncuBrix Workflow               │
                    └───────────────────────────┬────────────────────────────┘
                                                │ POST /api/v1/music/jobs
                                                │ (with Idempotency-Key)
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ FastAPI Service (app.main)                                                                 │
│  ├── Correlation ID Middleware (X-Correlation-Id)                                          │
│  ├── Safety & Guardrail Filters (Vocal rejection, copyright prompt sanitize)               │
│  ├── Health / Readiness Check (/health, /ready)                                            │
│  └── Storage & Static Serving (/output)                                                    │
└──────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                       │ Async Queue / Task
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Pipeline Orchestrator (app.jobs.music_job)                                                 │
│                                                                                            │
│   [Step 1] Adapter Generation ──▶ MusicGen (Meta small/medium) or Mock Generator           │
│   [Step 2] Length Control     ──▶ Bar snap (BPM grid) + musical fade ending                │
│   [Step 3] Loudness Norm      ──▶ pyloudnorm -16 LUFS integrated, -1 dBTP true peak limiter│
│   [Step 4] Stem Separation    ──▶ Demucs (htdemucs) -> 4 instrument stems                  │
│   [Step 5] Stem Verification  ──▶ Vectorized sum difference verification (< -40 dB)        │
│   [Step 6] Voiceover Ducking  ──▶ Energy-based VAD + smooth cosine ducking (-12 dB)        │
│   [Step 7] Output Manifest    ──▶ music_manifest.json with beat grid, stem gains, licence  │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quick Start

### Option A: Local Development with Virtual Environment

```bash
# 1. Clone repository
git clone https://github.com/Natarajchandrasekaran/incubrix-ps1.git
cd incubrix-ps1

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
copy .env.example .env   # Windows
# or cp .env.example .env # Linux/macOS

# 5. Run tests (runs 26 unit & integration tests with mock adapter, no GPU needed)
pytest tests/ -v

# 6. Start the API server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### Option B: Docker Compose (API + Redis + MinIO)

```bash
docker compose up --build
```
Services initialized:
- `api`: FastAPI on port `8000`
- `redis`: Message broker on port `6379`
- `minio`: S3-compatible storage on port `9000` (Console on `9001`)

---

## 4. API Endpoints

| Method | Endpoint | Description | Status Code |
|---|---|---|---|
| `GET` | `/health` | Liveness health check | `200 OK` |
| `GET` | `/ready` | Readiness probe (dependencies, storage, model) | `200 OK` |
| `POST` | `/api/v1/music/jobs` | Submit async music generation job | `202 Accepted` |
| `GET` | `/api/v1/music/jobs/{id}`| Poll status and retrieve generated outputs | `200 OK` / `404 Not Found` |
| `GET` | `/api/v1/music/jobs` | List recent jobs with pagination and status filter | `200 OK` |
| `DELETE`| `/api/v1/music/jobs/{id}`| Cancel pending or processing job | `204 No Content` |
| `GET` | `/output/{id}/mix.wav` | Download generated master mix | `200 OK` |
| `GET` | `/output/{id}/stems/{stem}.wav` | Download isolated stem (drums, bass, other) | `200 OK` |
| `GET` | `/output/{id}/music_manifest.json` | Download complete audio manifest | `200 OK` |

---

## 5. Usage Examples

### 1. Submit a Music Generation Job

```bash
curl -X POST http://localhost:8000/api/v1/music/jobs \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: job-request-uuid-001" \
  -H "X-Correlation-Id: trace-session-777" \
  -d '{
    "prompt": "upbeat corporate tech showcase with energetic synth pads and acoustic guitar",
    "duration_s": 30.0,
    "tempo_bpm": 120,
    "mood": "inspirational",
    "style": "corporate"
  }'
```

**Response (HTTP 202 Accepted):**
```json
{
  "job_id": "84703fc1-f5d7-4baf-827f-db9dded4d7bc",
  "status": "pending",
  "created_at": "2026-10-19T06:30:00Z",
  "progress_percent": 0.0,
  "request": {
    "prompt": "upbeat corporate tech showcase with energetic synth pads and acoustic guitar",
    "duration_s": 30.0,
    "tempo_bpm": 120,
    "mood": "inspirational",
    "style": "corporate"
  }
}
```

### 2. Poll Status & Retrieve Completed Output

```bash
curl http://localhost:8000/api/v1/music/jobs/84703fc1-f5d7-4baf-827f-db9dded4d7bc
```

**Response (HTTP 200 OK):**
```json
{
  "job_id": "84703fc1-f5d7-4baf-827f-db9dded4d7bc",
  "status": "completed",
  "actual_duration_seconds": 30.0,
  "actual_loudness_lufs": -16.0,
  "processing_time_seconds": 2.45,
  "output_url": "/output/84703fc1-f5d7-4baf-827f-db9dded4d7bc/mix.wav",
  "stems": [
    {
      "stem_type": "drums",
      "url": "/output/84703fc1-f5d7-4baf-827f-db9dded4d7bc/stems/drums.wav",
      "duration_seconds": 30.0,
      "loudness_lufs": -21.4
    },
    {
      "stem_type": "bass",
      "url": "/output/84703fc1-f5d7-4baf-827f-db9dded4d7bc/stems/bass.wav",
      "duration_seconds": 30.0,
      "loudness_lufs": -20.8
    },
    {
      "stem_type": "other",
      "url": "/output/84703fc1-f5d7-4baf-827f-db9dded4d7bc/stems/other.wav",
      "duration_seconds": 30.0,
      "loudness_lufs": -18.2
    }
  ]
}
```

---

## 6. Batch Runner (Integration Standard #10)

A CLI runner is included to test batches of prompts against the live service:

```bash
# Run 10 diverse test inputs (including hard test cases)
python batch_runner/run.py --input tests/inputs/ --output results/ --api http://localhost:8000
```

The runner:
1. Validates `/health` check.
2. Posts each JSON input sequentially with error isolation.
3. Polls until completion and downloads `mix.wav` and metadata.
4. Generates `results/batch_summary.json` containing total runtimes, loudness compliance, and stem verifications.

---

## 7. Compliance with Must-Haves

| # | Must-Have Criterion | Engine Implementation | Verification / Evidence |
|---|---|---|---|
| **1** | **Exact length $\pm 0.05\text{s}$ & musical ending** | `app/audio/length.py`: `trim_or_pad_to_exact_length` with exponential musical fade and tempo snap. | Tested in `TestExactLength` (error $= 0.000\text{s}$). |
| **2** | **Real stems ($\ge 3$), sum equals mix** | `app/audio/stems.py`: Meta Demucs separation into 4 stems. Verified with `verify_stems_sum_to_mix`. | Sum error $< -40\text{ dB}$ (lossless float32). |
| **3** | **Instrumental only** | Prompt sanitization filters vocal requests; Demucs vocals stem is monitored and discarded. | Validated in test case `08_copyrighted_vocals.json`. |
| **4** | **Commercially safe** | Meta MusicGen (`MIT`), Demucs (`MIT`), no CC-BY-NC code. Manifest contains commercial license assertion. | Documented in `SOURCES.md`. |
| **5** | **Broadcast standard** | `app/audio/loudness.py`: ITU-R BS.1770-4 normalization to $-16\text{ LUFS} \pm 1\text{ LU}$, true peak $\le -1\text{ dBTP}$, $48\text{ kHz}$. | Verified in `TestLoudnessNormalization`. |
| **6** | **Integration Standard** | All 10 Integration Standard v2 criteria fulfilled (OpenAPI 3.1, 202 async jobs, Idempotency, logs). | Detailed in `REPORT.md`. |

---

## 8. Suno AI-Style & Multi-Modal Capabilities

Beyond the core brief requirements, the engine features interactive multi-modal creation:
- **Image-to-BGM Engine (`POST /api/v1/music/image-to-prompt`)**: Upload any thumbnail, cover art, or scene image to automatically analyze color temperature, brightness, and saturation to generate a tailored music prompt, tempo, and mood.
- **9 Multi-Genre Synthesis Engines**:
  - 🏢 **Corporate Upbeat**: C Major, bright bells, driving acoustic pop drums.
  - 🎬 **Cinematic Drama**: D Minor, orchestral strings, heavy taiko impacts.
  - ☕ **Lo-Fi Study**: F Major 7, Rhodes electric piano, dusty boom-bap drums.
  - 🚀 **Tech / EDM**: A Minor, 4-on-the-floor kick, rolling 16th saw bass.
  - 🎙️ **Podcast Bed**: G Major, subtle piano, low-distraction groove.
  - 🎷 **Jazz Swing**: Bb Mixolydian, walking upright bass, brushed swing cymbals.
  - 🎸 **Acoustic Folk**: G Major, fingerpicked guitar, soft shaker.
  - 🌌 **Ambient Space**: C Lydian, slow evolving filter pad drones, deep sub.
  - 🎤 **Hip-Hop / Trap**: E minor, 808 sub kick, triplet hats, dark stabs.
  - 🤘 **Heavy Rock**: E minor power chords with distortion overtones, fast riffs.
- **Real-Time 36-Band FFT Visualizer**: Hardware-accelerated canvas spectrum animation.
- **Frame-Synced Video Monitor**: Locks video preview to DAW stem timeline.
- **Standard MIDI Exporter**: Direct DAW integration (.MID format).

---

## 9. License

This repository and all custom code are released under the **MIT License**. Third-party libraries and pretrained weights (MusicGen, Demucs) are licensed under permissive open-source licenses (MIT/BSD/Apache 2.0).
