"""Audio upload pipeline — unified upload → validate → save → probe → decode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from fastapi import HTTPException, Request, UploadFile

from app.config import Settings
from app.services.audio import (
    cleanup_temp_file,
    decode_audio_ffmpeg_async,
    get_audio_duration_async,
    save_upload_to_temp,
)

logger = structlog.get_logger(__name__)


@dataclass
class DecodedAudio:
    """Result of processing an audio upload through the pipeline."""
    audio: np.ndarray
    temp_path: Path
    duration: float


def validate_content_length(request: Request, max_size: int | None = None) -> None:
    """Validate the Content-Length header against max upload size.

    Raises HTTPException(413) if the upload is too large, or 400 if the header is invalid.
    """
    content_length = request.headers.get("Content-Length")
    if not content_length:
        return

    limit = max_size if max_size is not None else Settings.resolve(None).max_upload_bytes

    try:
        cl = int(content_length)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid Content-Length header")
    if cl < 0:
        raise HTTPException(status_code=400, detail="Content-Length must not be negative")
    if cl > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: Content-Length {cl} exceeds limit of {limit} bytes",
        )


async def process_upload(
    upload: UploadFile,
    max_size: int | None = None,
    max_duration: int | None = None,
    cleanup: bool = True,
) -> DecodedAudio:
    """Process an audio upload end-to-end.

    1. Save to temp file with size validation
    2. Probe duration and validate bounds
    3. Decode to float32 numpy array

    Args:
        upload: FastAPI UploadFile.
        max_size: Max upload size in bytes. Defaults to settings.max_upload_bytes.
        max_duration: Max audio duration in seconds. Defaults to settings.max_audio_duration.
        cleanup: Whether to clean up the temp file after decoding. Set False to keep
                 the temp file (e.g., for MD5 hashing and permanent storage).

    Returns:
        DecodedAudio with audio array, temp path, and duration.

    Raises:
        HTTPException: On validation failure or decode error.
    """
    size_limit = max_size if max_size is not None else Settings.resolve(None).max_upload_bytes
    duration_limit = max_duration if max_duration is not None else Settings.resolve(None).max_audio_duration

    # Step 1: Save to temp
    tmp_path = await save_upload_to_temp(upload, max_size=size_limit)

    try:
        # Step 2: Probe duration
        duration = await get_audio_duration_async(tmp_path)
        if duration > duration_limit:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long: {duration:.1f}s exceeds limit of {duration_limit}s",
            )
        if duration < 0.1:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short: {duration:.2f}s",
            )

        # Step 3: Decode
        audio = await decode_audio_ffmpeg_async(tmp_path)

        result = DecodedAudio(audio=audio, temp_path=tmp_path, duration=duration)

        if cleanup:
            cleanup_temp_file(tmp_path)

        return result
    except Exception:
        cleanup_temp_file(tmp_path)
        raise