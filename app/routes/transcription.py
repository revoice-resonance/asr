"""Transcription endpoint — OpenAI-compatible POST /v1/audio/transcriptions."""

from __future__ import annotations

from typing import Literal

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.schemas.responses import (
    ErrorResponse,
    TranscriptionResponse,
    TranscriptionSegment,
    VerboseTranscriptionResponse,
)
from app.services.audio import (
    cleanup_temp_file,
    decode_audio_ffmpeg_async,
    get_audio_duration_async,
    save_upload_to_temp,
)

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["transcription"])

# Rate limiter — keyed by client IP
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse | VerboseTranscriptionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        413: {"model": ErrorResponse, "description": "File too large"},
        429: {"model": ErrorResponse, "description": "Rate limited"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
@limiter.limit(
    f"{settings.rate_limit_rpm}/minute" if settings.rate_limit_rpm > 0 else "100000/minute"
)
async def transcribe(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form(
        default="",
        description="Language code (e.g. 'zh', 'en'). Empty for auto-detect.",
    ),
    temperature: float = Form(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0-1). Lower = more deterministic.",
    ),
    response_format: Literal["json", "verbose_json"] = Form(
        default="json",
        description="Response format: 'json' for text only, 'verbose_json' for segments.",
    ),
) -> TranscriptionResponse | VerboseTranscriptionResponse:
    """Transcribe audio to text.

    OpenAI-compatible endpoint. Accepts multipart form data with an audio file.
    Supports all formats that ffmpeg can decode (wav, mp3, m4a, ogg, flac, etc.).

    Authentication is handled by the upstream API gateway — no Bearer token
    validation is performed at this layer.
    """
    # Validate language
    lang = language.strip()

    # B-3: Pre-check Content-Length to reject oversized uploads before reading body
    content_length = request.headers.get("Content-Length")
    if content_length:
        try:
            cl = int(content_length)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail="Invalid Content-Length header",
            )
        if cl < 0:
            raise HTTPException(
                status_code=400,
                detail="Content-Length must not be negative",
            )
        if cl > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: Content-Length {cl} exceeds "
                        f"limit of {settings.max_upload_bytes} bytes",
            )

    # Save upload to temp file
    tmp_path = await save_upload_to_temp(file)

    try:
        # Check audio duration (non-blocking via asyncio.to_thread)
        duration = await get_audio_duration_async(tmp_path)
        if duration > settings.max_audio_duration:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long: {duration:.1f}s exceeds limit of "
                        f"{settings.max_audio_duration}s",
            )

        if duration < 0.1:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short: {duration:.2f}s",
            )

        # Decode to numpy array (non-blocking via asyncio.to_thread)
        audio = await decode_audio_ffmpeg_async(tmp_path)

        # Submit to GPU worker
        worker = request.app.state.worker
        client_id = request.client.host if request.client else "unknown"
        try:
            result = await worker.submit(
                audio, language=lang, temperature=temperature, client_id=client_id
            )
        except RuntimeError as exc:
            # Worker is shutting down or not running — map to 503 (C-1)
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            )

        # Build response
        if response_format == "verbose_json":
            segments = [
                TranscriptionSegment(
                    id=seg["id"],
                    seek=seg["seek"],
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    tokens=seg["tokens"],
                    temperature=seg["temperature"],
                    avg_logprob=seg["avg_logprob"],
                    compression_ratio=seg["compression_ratio"],
                    no_speech_prob=seg["no_speech_prob"],
                )
                for seg in result.segments
            ]
            return VerboseTranscriptionResponse(
                text=result.text,
                language=result.language,
                duration=result.duration,
                segments=segments,
            )
        else:
            return TranscriptionResponse(text=result.text)

    finally:
        # Always clean up temp file
        cleanup_temp_file(tmp_path)
