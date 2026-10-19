# SOURCES.md — Third-Party Dependencies and Licensing

> IncuBrix 2027 Graduate Hiring Assessment — Problem Statement 1 (PS1)  
> Candidate: **Nataraj Chandrasekaran** (PSG College of Technology)  
> Project: **AI Background Music Engine for Video**

---

## 1. Commercial Safety Statement

All software components, pretrained weights, and audio algorithms utilized in this project are strictly verified for **unrestricted commercial use**.
- **No Non-Commercial Licenses**: Zero dependencies licensed under CC BY-NC, CC BY-NC-SA, or similar restrictions.
- **No Copyleft Contamination**: Zero runtime linking against viral copyleft licenses (e.g. AGPL/GPL).
- **Original Output Generation**: Output music is generated purely via latent audio diffusion / autoregressive generation and signal processing. No copyrighted music or trademarked compositions are embedded or reproduced.

---

## 2. Machine Learning Models

| Component / Model | Provider / Authors | License | Repository / Origin | Commercial Rights & Usage |
|---|---|---|---|---|
| **MusicGen (small / medium)** | Meta AI Research (Audiocraft) | **MIT License** | [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) | Model architecture and open weights released under MIT; fully licensed for commercial inference and derivative generation. |
| **Demucs v4 (htdemucs)** | Meta AI Research | **MIT License** | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | State-of-the-art hybrid transformer source separation released under MIT. Permitted for commercial audio processing. |

---

## 3. Core Software Libraries

| Library | Version | License | Upstream Source | Primary Function |
|---|---|---|---|---|
| `fastapi` | 0.115.6 | **MIT** | [tiangolo/fastapi](https://github.com/tiangolo/fastapi) | Async REST API framework |
| `uvicorn` | 0.34.0 | **BSD 3-Clause** | [encode/uvicorn](https://github.com/encode/uvicorn) | High-performance ASGI web server |
| `pydantic` | 2.10.4 | **MIT** | [pydantic/pydantic](https://github.com/pydantic/pydantic) | Strict type validation and JSON schema generation |
| `pyloudnorm` | 0.1.1 | **MIT** | [csteinmetz1/pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | ITU-R BS.1770-4 compliant loudness normalization (LUFS) |
| `soundfile` | 0.13.1 | **BSD 3-Clause** | [bastibe/python-soundfile](https://github.com/bastibe/python-soundfile) | Uncompressed 48 kHz float32 / PCM audio I/O |
| `numpy` | 2.2.1 | **BSD 3-Clause** | [numpy/numpy](https://github.com/numpy/numpy) | Vectorized digital signal processing |
| `scipy` | 1.15.0 | **BSD 3-Clause** | [scipy/scipy](https://github.com/scipy/scipy) | Digital filtering (Butterworth, cos ducking, signal envelopes) |
| `structlog` | 24.4.0 | **MIT / Apache 2.0**| [hynek/structlog](https://github.com/hynek/structlog) | Production structured JSON telemetry & logging |
| `httpx` | 0.28.1 | **BSD 3-Clause** | [encode/httpx](https://github.com/encode/httpx) | Async HTTP client for webhook delivery & batch runner |
| `minio` | 7.2.20 | **Apache 2.0** | [minio/minio-py](https://github.com/minio/minio-py) | S3-compatible client SDK for artifact persistence |
| `pytest` | 8.3.4 | **MIT** | [pytest-dev/pytest](https://github.com/pytest-dev/pytest) | Automated regression test runner |

---

## 4. Container & Infrastructure Tooling

| Software | License | Role in Submission |
|---|---|---|
| **Docker Engine & Compose** | Apache 2.0 | Containerized deployment manifest (`Dockerfile`, `docker-compose.yml`) |
| **Redis** | BSD 3-Clause | Celery distributed task broker (optional distributed mode) |
| **FFmpeg** | LGPL 2.1+ | Underlying dynamic audio encoding utilities in container |
