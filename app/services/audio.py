"""Audio processing utilities — ffmpeg decode, ffprobe duration, upload handling."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from fastapi import HTTPException, UploadFile

from app.config import settings


# --- Audio Decoding ---


def decode_audio_ffmpeg(file_path: str | Path, sr: int = 16000) -> np.ndarray:
    """Decode any audio file to 16kHz mono float32 numpy array via ffmpeg.

    Args:
        file_path: Path to the audio file.
        sr: Target sample rate (default 16000 for Whisper).

    Returns:
        Float32 numpy array of shape (n_samples,).

    Raises:
        HTTPException: If ffmpeg fails or produces empty output.
    """
    file_path = str(file_path)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-loglevel", "error",
        "-",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Audio decoding timed out",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found — ensure ffmpeg is installed on the server",
        )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode audio: {stderr or 'unknown ffmpeg error'}",
        )

    if len(proc.stdout) == 0:
        raise HTTPException(
            status_code=400,
            detail="Decoded audio is empty — file may be corrupt or silent",
        )

    # Convert s16le bytes → float32 normalized to [-1, 1]
    raw = np.frombuffer(proc.stdout, dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


# --- Duration Check ---


def get_audio_duration(file_path: str | Path) -> float:
    """Get audio duration in seconds via ffprobe.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration in seconds as a float.

    Raises:
        HTTPException: If ffprobe fails or cannot determine duration.
    """
    file_path = str(file_path)

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Audio duration check timed out",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffprobe not found — ensure ffmpeg is installed on the server",
        )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to probe audio: {stderr or 'unknown ffprobe error'}",
        )

    try:
        duration = float(proc.stdout.decode("utf-8").strip())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Could not determine audio duration",
        )

    if duration <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio duration: {duration:.2f}s",
        )

    return duration


# --- Upload Handling ---


async def save_upload_to_temp(upload: UploadFile, max_size: int | None = None) -> Path:
    """Stream an uploaded file to a temporary location with size limiting.

    Args:
        upload: FastAPI UploadFile object.
        max_size: Maximum allowed size in bytes. Defaults to settings.max_upload_bytes.

    Returns:
        Path to the saved temporary file.

    Raises:
        HTTPException: If the upload exceeds max_size or has no filename.
    """
    if max_size is None:
        max_size = settings.max_upload_bytes

    if not upload.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided in upload",
        )

    # Ensure temp directory exists
    temp_dir = settings.temp_dir_resolved
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create a temp file with the original extension to help ffmpeg detect format
    suffix = Path(upload.filename).suffix or ".tmp"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=str(temp_dir))
    tmp_path = Path(tmp_path)

    try:
        total = 0
        with os.fdopen(fd, "wb") as f:
            while chunk := await upload.read(1024 * 1024):  # 1 MB chunks
                total += len(chunk)
                if total > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large: {total} bytes exceeds limit of {max_size} bytes",
                    )
                f.write(chunk)

        if total == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

    except HTTPException:
        # Clean up temp file on error
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file",
        )

    return tmp_path


def cleanup_temp_file(file_path: str | Path) -> None:
    """Safely remove a temporary file, ignoring errors."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception:
        pass
