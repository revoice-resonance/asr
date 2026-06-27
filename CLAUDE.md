# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Production-grade FastAPI server wrapping `faster-whisper` (CTranslate2 backend) for speech-to-text transcription. Part of the **Revoice Resonance** platform (构音障碍语音 AI) — a three-service system where the **main backend** orchestrates calls to this ASR service and a separate TTS service. `db.md` documents the inter-service architecture.

**Two operating modes**, gated by `DATABASE_URL`:
- **Stateless** (default, `DATABASE_URL` empty): OpenAI-compatible transcription endpoint, no persistence. Backward-compatible with existing deployments.
- **DB-backed** (`DATABASE_URL` set): Full corpus management with MD5 dedup, async task lifecycle tracking (PENDING → PROCESSING → SUCCESS/FAILED), persistent file storage, and HTTP callback notification to the main backend.

## Architecture

```
asr/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app factory, lifespan (DB + worker + scheduler), uvicorn entry
│   ├── config.py            # pydantic-settings — all config via env vars / .env
│   ├── database.py           # Async SQLAlchemy engine + session (no-op when DATABASE_URL empty)
│   ├── dependencies.py      # DI: get_db session, get_settings
│   ├── middleware.py         # Request ID injection + structured timing logging (structlog)
│   ├── worker.py             # TaskScheduler: DB-polling loop → claims PENDING tasks → GPU worker → callback
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py           # SQLAlchemy DeclarativeBase + TimestampMixin
│   │   ├── corpus.py         # Corpus ORM (file metadata, MD5, upload status, business context)
│   │   └── asr_task.py       # AsrTask ORM (per-engine task, status, confidence, result_detail)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py         # /health/live, /health/ready, /health/gpu
│   │   ├── transcription.py  # POST /v1/audio/transcriptions (OpenAI-compatible, stateless)
│   │   └── corpus.py         # /api/v1/asr/corpus, /api/v1/asr/tasks (DB-backed CRUD)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio.py          # ffmpeg decode, ffprobe, upload streaming, file storage
│   │   ├── transcriber.py    # faster-whisper model + asyncio.Queue GPU worker
│   │   ├── corpus.py         # Corpus business logic (MD5 hashing, dedup, CRUD)
│   │   └── callback.py       # HTTP callback to main backend with exponential retry
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── responses.py      # Pydantic response models (OpenAI + corpus/task)
│   │   └── requests.py       # Pydantic request schemas
│   └── migrations/
│       ├── __init__.py
│       └── 001_init.sql      # ASR schema: corpus + asr_tasks tables, indexes, triggers
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_audio.py
│   ├── test_worker.py
│   ├── test_corpus_service.py
│   ├── test_task_scheduler.py
│   └── test_corpus_routes.py
├── .env.example
├── .gitignore
├── .dockerignore
├── deploy.sh                 # Non-invasive deployment for AutoDL / GPU cloud (venv + nohup)
├── Dockerfile                # nvidia/cuda container (NOT for AutoDL)
├── docker-compose.yml        # GPU reservation, model volume mount
├── requirements.txt
├── pyproject.toml            # pytest config (asyncio_mode = "auto")
└── whisper_api.py            # Original single-file server (reference only)
```

## Key Design Decisions

| Concern | Choice | Rationale |
|---|---|---|
| ASR engine | `faster-whisper` (CTranslate2) | 4x faster than HF transformers, built-in Silero VAD |
| GPU concurrency | `asyncio.Queue` + single worker thread | Serializes GPU access, prevents OOM |
| ORM | SQLAlchemy 2.0 async + asyncpg | Connection pooling, async session, industry standard |
| Database migration | Clean SQL file (`001_init.sql`) | 2 tables — simpler than Alembic for now |
| Task claiming | `UPDATE … WHERE status='PENDING' … RETURNING *` | Atomic row lock, safe for horizontal scaling |
| Stateless backward compat | `DATABASE_URL` empty → all DB code is no-op | Zero-config for existing deployments |
| File storage | Local filesystem at `STORAGE_PATH` | Configurable; S3/MinIO gated behind future config |
| Callback | HTTP POST with exponential backoff | Reliable async notification to main orchestrator |
| Auth | Handled upstream by API gateway | No auth at this layer (intranet trust) |
| Config | `pydantic-settings` with `extra="forbid"` | Type-safe, catches config typos |
| Logging | `structlog` | Structured JSON in prod, colored console in dev |
| Rate limiting | `slowapi` (in-memory) | No Redis dependency |

## Dual-Mode Data Flow

### Stateless flow (`POST /v1/audio/transcriptions`)
```
Upload → save temp → ffprobe duration → ffmpeg decode → GPU worker queue → return text
                                                                                    │
                                          TranscriptionWorker (asyncio.Queue + asyncio.to_thread)
                                            └─ faster-whisper.transcribe(audio, vad_filter=True)
                                                 beam_size=5, best_of=5, temperature fallback
                                                 torch.cuda.empty_cache() after each job
```

### DB-backed flow (`POST /api/v1/asr/corpus`)
```
Upload → save temp → compute MD5 → dedup check
  ├─ MD5 match + SUCCESS task → return cached result (cached: true)
  └─ New file:
       ├─ INSERT corpus (status=UPLOADING) → move file to STORAGE_PATH → UPDATE (status=UPLOADED)
       ├─ INSERT asr_tasks (status=PENDING)
       └─ Return {corpus_id, task_id, cached: false}
            │
            └─ TaskScheduler (background, polls every TASK_POLL_INTERVAL seconds)
                 ├─ UPDATE task status=PROCESSING, started_at=now()  (atomic claim)
                 ├─ Load + decode audio from storage
                 ├─ Submit to same TranscriptionWorker queue
                 ├─ UPDATE task status=SUCCESS, result_text, confidence, result_detail
                 ├─ UPDATE corpus.text_content (denormalized for query convenience)
                 └─ POST callback to MAIN_BACKEND_CALLBACK_URL (with retry)
```

## API Endpoints

| Method | Path | DB Required | Description |
|---|---|---|---|
| `GET` | `/health/live` | No | K8s liveness — always 200 |
| `GET` | `/health/ready` | No | K8s readiness — 503 if model/GPU not ready |
| `GET` | `/health/gpu` | No | GPU VRAM metrics |
| `POST` | `/v1/audio/transcriptions` | No | OpenAI-compatible, stateless transcription |
| `POST` | `/api/v1/asr/corpus` | **Yes** | Upload audio → corpus + task; supports MD5 dedup |
| `GET` | `/api/v1/asr/corpus` | **Yes** | List corpora (filter by business_id/type/status) |
| `GET` | `/api/v1/asr/corpus/{id}` | **Yes** | Corpus details with embedded task summaries |
| `GET` | `/api/v1/asr/tasks` | **Yes** | List tasks (filter by status/corpus_id) |
| `GET` | `/api/v1/asr/tasks/{id}` | **Yes** | Full task details including result_detail |

DB-required endpoints return **503** with `"detail": "Database not configured. Set DATABASE_URL to enable."` when running in stateless mode.

## Database

The ASR service owns the **`ASR`** PostgreSQL database (see `db.md` for cross-service architecture). Two tables:

- **`corpus`** — Audio file registry. MD5 unique constraint enables dedup. `status`: UPLOADING → UPLOADED → (FAILED). `text_content` is a denormalized copy of the latest successful ASR result. `business_id`/`business_type` link to upstream domain (e.g., patient records).
- **`asr_tasks`** — Per-engine ASR task journal. `status`: PENDING → PROCESSING → SUCCESS/FAILED. Stores `result_text`, `confidence` (0-1, 4 decimal places), `result_detail` (JSON with timestamps/word-level data), `processing_time` (ms).

Initial schema is in `app/migrations/001_init.sql` — a fixed version of `fuck_u_qiniu.sql` (corrected trigger column names, removed cross-DB contamination).

## Configuration

All settings via environment variables or `.env` file. See `.env.example` for the full list.

Key vars by concern:

| Concern | Key Vars |
|---|---|
| Server | `HOST`, `PORT`, `CORS_ORIGINS` |
| Model | `MODEL_PATH`, `MODEL_DEVICE`, `MODEL_COMPUTE_TYPE`, `MODEL_DEVICE_INDEX` |
| Model download | `MODEL_DOWNLOAD_URL`, `HF_MODEL_ID` |
| Audio limits | `MAX_UPLOAD_BYTES`, `MAX_AUDIO_DURATION`, `DEFAULT_LANGUAGE` |
| VAD | `VAD_ENABLED`, `VAD_THRESHOLD`, `VAD_MIN_SILENCE_DURATION_MS` |
| Rate limit | `RATE_LIMIT_RPM`, `RATE_LIMIT_BURST` |
| Logging | `LOG_LEVEL`, `LOG_FORMAT` (`json`/`console`) |
| Temp | `TEMP_DIR` |
| **Database** | `DATABASE_URL`, `DATABASE_POOL_SIZE`, `DATABASE_POOL_OVERFLOW` |
| **Storage** | `STORAGE_PATH` |
| **Scheduler** | `TASK_POLL_INTERVAL`, `TASK_MAX_RETRIES`, `MAX_CONCURRENT_TASKS` |
| **Callback** | `MAIN_BACKEND_CALLBACK_URL`, `CALLBACK_MAX_RETRIES`, `CALLBACK_RETRY_BASE_DELAY` |
| Proxy | `HTTP_PROXY`, `HTTPS_PROXY` |

`DATABASE_URL` is the mode switch — empty = stateless, set = DB-backed.

## Running

### Local Development
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set MODEL_PATH; optionally DATABASE_URL for DB mode
python -m app.main
```

### AutoDL / GPU Cloud
```bash
bash deploy.sh start       # isolated venv + nohup uvicorn
bash deploy.sh stop        # SIGTERM + 30s grace
bash deploy.sh restart
bash deploy.sh status
bash deploy.sh logs 100
```
AutoDL does **NOT** support Docker. `deploy.sh` creates an isolated venv, supports HTTP proxy.

### Docker
```bash
docker compose up -d
```

## Testing

```bash
pytest -v                          # all tests
pytest tests/test_config.py -v     # config only
pytest tests/test_worker.py -v     # worker lifecycle
```
Tests run without GPU/DB — ffmpeg and GPU calls are mocked or use monkeypatched env. `pyproject.toml` sets `asyncio_mode = "auto"`.

## Key Dependencies

- `faster-whisper` (CTranslate2 Whisper, built-in Silero VAD)
- `fastapi` + `uvicorn[standard]`
- `sqlalchemy[asyncio]` + `asyncpg` (DB layer)
- `httpx` (async HTTP client for main-backend callbacks)
- `pydantic-settings` (configuration)
- `structlog` (structured logging)
- `slowapi` (rate limiting)
- `numpy`, `python-multipart`, `aiofiles`
- `ffmpeg` / `ffprobe` (system-level, for audio decode)

## Git Conventions

- Model files are **never** committed (`.gitignore` excludes `models/`, `*.safetensors`, `*.bin`)
- `.env` is gitignored; `.env.example` is committed as a template
- Upstream: `https://github.com/revoice-resonance/asr.git`
