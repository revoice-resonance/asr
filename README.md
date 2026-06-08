# Whisper ASR API

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Production-grade speech-to-text API powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend). OpenAI-compatible transcription endpoint with built-in VAD, rate limiting, API key auth, and Kubernetes-ready health probes.

## Features

- **4x faster** than HuggingFace transformers via CTranslate2
- **Built-in Silero VAD** — filters silence and non-speech automatically
- **OpenAI-compatible** `POST /v1/audio/transcriptions` endpoint
- **GPU worker queue** — serializes inference to prevent OOM under concurrency
- **Structured JSON logging** via structlog (colored console in dev mode)
- **Rate limiting** per IP (in-memory, no Redis needed)
- **Bearer token auth** — optional, disabled when no keys configured
- **K8s health probes** — `/health/live`, `/health/ready`, `/health/gpu`
- **Non-blocking I/O** — ffmpeg/ffprobe offloaded to thread pool
- **Non-invasive deploy** — `deploy.sh` uses isolated venv, never touches system Python

## Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/revoice-resonance/asr.git
cd asr

cp .env.example .env
# Edit .env — at minimum set MODEL_DOWNLOAD_URL or HF_MODEL_ID
```

### 2. Deploy

**AutoDL / GPU cloud (recommended):**

```bash
bash deploy.sh start
```

**Local / standard GPU server:**

```bash
pip install -r requirements.txt
python -m app.main
```

**Docker (not for AutoDL):**

```bash
docker compose up -d
```

### 3. Verify

```bash
# Health check
curl http://localhost:8080/health/live

# GPU info
curl http://localhost:8080/health/gpu

# Transcribe
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "language=zh"
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe (model + GPU) |
| `GET` | `/health/gpu` | GPU VRAM metrics |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio |

### Transcription

```
POST /v1/audio/transcriptions
Authorization: Bearer <api_key>     # optional
Content-Type: multipart/form-data

file:             <audio_file>      # required (wav, mp3, m4a, ogg, flac, ...)
language:         zh                # optional (default: zh)
response_format:  json|verbose_json # optional (default: json)
```

**Response (`json`):**

```json
{ "text": "转录结果文本" }
```

**Response (`verbose_json`):**

```json
{
  "text": "转录结果文本",
  "language": "zh",
  "duration": 12.34,
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "第一段文本",
      "tokens": [...],
      "temperature": 0.0,
      "avg_logprob": -0.23,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01
    }
  ]
}
```

## Configuration

All settings via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8080` | Server port |
| `MODEL_PATH` | `models/whisper-large-v3-turbo-ct2` | CTranslate2 model dir |
| `MODEL_DEVICE` | `cuda` | `cuda` or `cpu` |
| `MODEL_COMPUTE_TYPE` | `float16` | `float16`, `int8_float16`, `int8` |
| `DEFAULT_LANGUAGE` | `zh` | Fallback language |
| `MAX_UPLOAD_BYTES` | `524288000` | Max file size (500 MB) |
| `MAX_AUDIO_DURATION` | `600` | Max audio length (seconds) |
| `VAD_ENABLED` | `true` | Silero VAD filtering |
| `RATE_LIMIT_RPM` | `60` | Requests/min per IP (0=off) |
| `API_KEYS` | `""` | Comma-separated keys (empty=no auth) |
| `LOG_FORMAT` | `json` | `json` or `console` |
| `MODEL_DOWNLOAD_URL` | `""` | Model archive download URL |
| `HF_MODEL_ID` | `""` | HuggingFace model ID for ct2 conversion |
| `HTTP_PROXY` | `""` | Proxy for pip/downloads |

## deploy.sh

```bash
bash deploy.sh setup       # Install deps + download model (don't start)
bash deploy.sh start       # Setup + start server
bash deploy.sh stop        # Graceful shutdown (SIGTERM → 30s → SIGKILL)
bash deploy.sh restart     # Stop + start
bash deploy.sh status      # Show PID, port, model, health check
bash deploy.sh logs 100    # Tail last N lines
```

## Architecture

```
POST /v1/audio/transcriptions
  │
  ├─ Middleware: X-Request-ID, timing
  ├─ Auth: Bearer token (skipped if no keys)
  ├─ Rate limit (slowapi, per-IP)
  ├─ Upload → temp file (1MB chunks)
  ├─ ffprobe → duration check
  ├─ ffmpeg → 16kHz mono float32
  │
  ├─ asyncio.Queue → GPU Worker (single)
  │     └─ faster-whisper.transcribe()
  │           ├─ beam_size=5, best_of=5
  │           ├─ Silero VAD filter
  │           └─ torch.cuda.empty_cache()
  │
  └─ JSONResponse
```

## Model

Uses `whisper-large-v3-turbo` (809M params) converted to CTranslate2 format. The model is **not** stored in this repo — it is downloaded/converted by `deploy.sh` or mounted as a Docker volume.

## Requirements

- Python 3.10+
- CUDA-capable GPU (or CPU with degraded performance)
- ffmpeg (system-level)
- See `requirements.txt` for Python dependencies

## License

MIT
