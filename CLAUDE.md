# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Production-grade FastAPI server wrapping `faster-whisper` (CTranslate2 backend) for speech-to-text transcription. Replaces the original single-file HuggingFace transformers pipeline with a 4x faster implementation featuring built-in Silero VAD, structured logging, rate limiting, API key auth, and Kubernetes-compatible health probes.

## Architecture

```
whisper_api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app factory, lifespan, middleware, uvicorn entry
│   ├── config.py            # pydantic-settings (env vars + .env)
│   ├── dependencies.py      # Bearer token auth dependency
│   ├── middleware.py         # Request ID + timing logging (structlog)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py        # /health/live, /health/ready, /health/gpu
│   │   └── transcription.py # POST /v1/audio/transcriptions (OpenAI-compatible)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio.py         # ffmpeg decode, ffprobe duration, upload handling
│   │   └── transcriber.py   # faster-whisper model, asyncio.Queue GPU worker
│   └── schemas/
│       ├── __init__.py
│       └── responses.py     # Pydantic response models (OpenAI-compatible)
├── .env.example             # All config vars with defaults
├── .gitignore               # Excludes model dirs, venv, .env, logs
├── .dockerignore
├── deploy.sh                # Non-invasive deployment for AutoDL / GPU cloud
├── Dockerfile               # nvidia/cuda container (NOT for AutoDL)
├── docker-compose.yml       # GPU reservation, healthcheck, model volume mount
├── requirements.txt
└── whisper_api.py           # Original single-file server (reference only)
```

## Key Design Decisions

| Concern | Choice | Rationale |
|---|---|---|
| Model serving | `faster-whisper` (CTranslate2) | 4x faster than HF transformers, built-in Silero VAD, better VRAM |
| Concurrency | `asyncio.Queue` + single GPU worker | Serializes GPU access, prevents OOM from concurrent inference |
| Config | `pydantic-settings` | `.env` + env vars, type-safe, single `settings` singleton |
| Logging | `structlog` | Structured JSON in prod, colored console in dev |
| Rate limiting | `slowapi` | In-memory, no Redis dependency |
| Auth | API key (Bearer token) | Simple, sufficient for API; disabled if no keys configured |
| Audio decode | ffmpeg subprocess | Same as original, battle-tested, wrapped in `asyncio.to_thread` |
| Health | Liveness + Readiness + GPU | K8s-compatible probes |

## Data Flow (Transcription)

```
POST /v1/audio/transcriptions
  │
  ├─ Middleware: X-Request-ID injection, timing
  ├─ Auth: Bearer token validation (skipped if no API_KEYS)
  ├─ Rate limit check (slowapi, per-IP)
  │
  ├─ Stream upload → temp file (1MB chunks, max 500MB)
  ├─ ffprobe → duration check (max 10 min)
  ├─ ffmpeg → 16kHz mono float32 numpy (asyncio.to_thread)
  │
  ├─ Push to asyncio.Queue → await Future
  │     │
  │     └─ GPU Worker (single, asyncio.to_thread)
  │           ├─ faster-whisper.transcribe(audio, vad_filter=True)
  │           ├─ beam_size=5, best_of=5, temperature=[0.0..1.0]
  │           ├─ torch.cuda.empty_cache()
  │           └─ Return segments + text
  │
  ├─ Cleanup temp file (finally block)
  └─ JSONResponse (TranscriptionResponse or VerboseTranscriptionResponse)
```

## Model

Fine-tuned `whisper-large-v3-turbo` (809M params, encoder-decoder, 32 encoder layers, 4 decoder layers, d_model=1280). Converted to CTranslate2 format for faster-whisper.

The model is **NOT stored in git** — it is excluded via `.gitignore` (`models/`, `whisper-large-v3-turbo-finetuned/`, `*.safetensors`, `*.bin`). Model setup is handled by `deploy.sh` or Docker volume mount.

Default model path: `models/whisper-large-v3-turbo-ct2` (configurable via `MODEL_PATH` env var).

## Configuration

All settings via environment variables or `.env` file. See `.env.example` for the full list. Key vars:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8080` | Server port |
| `MODEL_PATH` | `models/whisper-large-v3-turbo-ct2` | CTranslate2 model directory |
| `MODEL_DEVICE` | `cuda` | `cuda` or `cpu` |
| `MODEL_COMPUTE_TYPE` | `float16` | `float16`, `int8_float16`, `int8` |
| `DEFAULT_LANGUAGE` | `zh` | Fallback language for transcription |
| `MAX_UPLOAD_BYTES` | `524288000` | Max file size (500 MB) |
| `MAX_AUDIO_DURATION` | `600` | Max audio length in seconds |
| `VAD_ENABLED` | `true` | Enable Silero VAD filtering |
| `RATE_LIMIT_RPM` | `60` | Requests per minute per IP (0 = disabled) |
| `API_KEYS` | `""` | Comma-separated API keys (empty = no auth) |
| `LOG_FORMAT` | `json` | `json` for prod, `console` for dev |
| `MODEL_DOWNLOAD_URL` | `""` | URL to download model archive |
| `HF_MODEL_ID` | `""` | HuggingFace model ID for ct2 conversion |

## Running

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env — set MODEL_PATH, API_KEYS, etc.

# Start server
python -m app.main
# or: uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### AutoDL / GPU Cloud (Primary)

```bash
# First time setup
cp .env.example .env
# Edit .env — set MODEL_DOWNLOAD_URL or HF_MODEL_ID

# Deploy
bash deploy.sh start       # setup + start
bash deploy.sh stop        # graceful shutdown
bash deploy.sh restart     # stop + start
bash deploy.sh status      # health check
bash deploy.sh logs 100    # tail logs
```

AutoDL does **NOT** support Docker (no root, shared kernel). Use `deploy.sh` — it creates an isolated venv, does not touch system Python, and supports HTTP proxy for pip/model downloads.

### Docker (Standard GPU Servers Only)

```bash
cp .env.example .env
# Place model in ./models/whisper-large-v3-turbo-ct2/ or set MODEL_DOWNLOAD_URL
docker compose up -d
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health/live` | Liveness probe — always 200 |
| `GET` | `/health/ready` | Readiness probe — 503 if model/GPU not ready |
| `GET` | `/health/gpu` | GPU VRAM metrics |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible transcription |

### Transcription Request

```
POST /v1/audio/transcriptions
Authorization: Bearer <api_key>    # optional if no API_KEYS configured
Content-Type: multipart/form-data

file: <audio_file>                 # required, any ffmpeg-supported format
language: zh                       # optional, default from DEFAULT_LANGUAGE
response_format: json|verbose_json # optional, default "json"
```

## Key Dependencies

- `faster-whisper>=1.0.0` (CTranslate2 Whisper, built-in VAD)
- `fastapi>=0.110.0` + `uvicorn[standard]>=0.29.0`
- `pydantic-settings>=2.0.0` (configuration)
- `structlog>=24.0.0` (structured logging)
- `slowapi>=0.1.9` (rate limiting)
- `numpy>=1.24.0`
- `python-multipart>=0.0.9` (file uploads)
- `aiofiles>=23.0.0` (async file I/O)
- `ffmpeg` (system-level, for audio decoding)

## Git Conventions

- Model files are **never** committed (`.gitignore` excludes `models/`, `*.safetensors`, `*.bin`)
- `.env` is gitignored; `.env.example` is committed as a template
- Upstream: `https://github.com/revoice-resonance/asr.git`
