"""Transcription endpoint — OpenAI-compatible POST /v1/audio/transcriptions."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.dependencies import verify_api_key
from app.schemas.responses import (
    ErrorDetail,
    ErrorResponse,
    TranscriptionResponse,
    TranscriptionSegment,
    VerboseTranscriptionResponse,
)
from app.services.audio import (
    cleanup_temp_file,
    decode_audio_ffmpeg,
    get_audio_duration,
    save_upload_to_temp,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["transcription"])

# Rate limiter — keyed by client IP
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse | VerboseTranscriptionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
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
    response_format: Literal["json", "verbose_json"] = Form(
        default="json",
        description="Response format: 'json' for text only, 'verbose_json' for segments.",
    ),
    _auth: None = Depends(verify_api_key),
) -> TranscriptionResponse | VerboseTranscriptionResponse:
    """Transcribe audio to text.

    OpenAI-compatible endpoint. Accepts multipart form data with an audio file.
    Supports all formats that ffmpeg can decode (wav, mp3, m4a, ogg, flac, etc.).

    Rate limited per IP. Requires Bearer token if API_KEYS is configured.
    """
    # Validate language
    lang = language.strip() or settings.default_language

    # Save upload to temp file
    tmp_path = await save_upload_to_temp(file)

    try:
        # Check audio duration
        duration = get_audio_duration(tmp_path)
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

        # Decode to numpy array
        audio = decode_audio_ffmpeg(tmp_path)

        # Submit to GPU worker
        worker = request.app.state.worker
        result = await worker.submit(audio, language=lang)

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
