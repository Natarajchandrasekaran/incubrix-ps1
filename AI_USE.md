# AI_USE.md — Declaration of AI Assistance and Tooling

> IncuBrix 2027 Graduate Hiring Assessment — Problem Statement 1 (PS1)  
> Candidate: **Nataraj Chandrasekaran** (PSG College of Technology, Coimbatore)  
> Role Applied: **Associate Tech Partner (ATP)**  
> Date: October 2026

---

## 1. Ethical AI Declaration

In full accordance with IncuBrix assessment guidelines, this document explicitly discloses the artificial intelligence systems, methodologies, and interactive tools leveraged during the engineering and documentation of this submission.

As an Associate Tech Partner candidate, AI tools were leveraged to accelerate engineering workflows, automate scaffolding, and test architectural patterns, while **core domain logic, signal processing verification, safety checks, and pipeline integration were directed, validated, and debugged by the candidate**.

---

## 2. Tools & Models Leveraged

| Tool / Model | Scope of Usage | Validation & Engineering by Candidate |
|---|---|---|
| **Antigravity / Google Gemini Assistant** | Interactive pair programming, initial FastAPI endpoint boilerplate, drafting docstrings, OpenAPI spec definition | Reviewed all schemas, debugged Pydantic v2 `AliasChoices` and serialization behaviors, corrected audio precision loss issues |
| **Meta MusicGen (`facebook/musicgen-small`)** | Text-to-audio generation engine (the primary AI ML model under test) | Engineered prompt normalization, token truncation, memory-efficient inference, and headless fallback adapter |
| **Meta Demucs (`htdemucs`)** | Instrument source separation | Implemented stem sum verification ($mix = \sum stems$) and float32 WAV losslessness |

---

## 3. Key Human-Led Engineering Decisions

While AI assisted with speed of implementation, key technical and architectural decisions were driven by the candidate:

1. **Swappable Adapter Pattern (`app/adapters/base.py`)**:
   - Designed an abstract interface `MusicAdapter` so that the service can run with local Meta MusicGen, cloud providers, or the `MockMusicAdapter` without touching business logic or tests.
2. **Deterministic Offline Testability**:
   - Designed the `MockMusicAdapter` generating multi-layered harmonic sine tones with exact duration and seeded reproducibility. This enables the full 26-test suite to run in $< 7\text{ seconds}$ without GPU or internet.
3. **Signal Processing & Audio Precision**:
   - Diagnosed and resolved a float32 vs 16-bit PCM quantization loss in WAV encoding that previously introduced $> 15\text{ dB}$ difference during stem summation. Enforcing `subtype="FLOAT"` guaranteed numerical reconstruction below $-40\text{ dBFS}$.
4. **Broadcast Standard Conformance**:
   - Integrated `pyloudnorm` ITU-R BS.1770-4 K-weighting to guarantee exactly $-16\text{ LUFS} \pm 1\text{ LU}$ and true peak $\le -1\text{ dBTP}$ preventing distortion on digital distribution channels.
5. **Zero-Pumping Voiceover Ducking**:
   - Implemented an energy-based Voice Activity Detection (VAD) coupled with smooth cosine crossfades ($200\text{ ms}$) to ensure music ducking remains natural rather than abrupt.

---

## 4. Verification & Reproducibility Guarantee

The candidate can reproduce, explain, and modify any portion of this codebase during the live technical evaluation.
