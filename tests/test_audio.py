"""Tests for audio processing utilities.

Covers: ffmpeg decode, ffprobe duration, upload save/cleanup, B-3 validation order.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

from app.services.audio import (
    cleanup_temp_file,
    decode_audio_ffmpeg,
    get_audio_duration,
    save_upload_to_temp,
)


class TestDecodeAudioFfmpeg:
    """Test ffmpeg audio decoding."""

    def test_decode_returns_float32_array(self, monkeypatch):
        """Decoded audio should be float32 numpy array."""
        # Create mock subprocess that returns valid s16le data
        mock_proc = mock.MagicMock()
        mock_proc.returncode = 0
        # 100 samples of silence as s16le
        mock_proc.stdout = b"\x00\x00" * 100

        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_proc)

        result = decode_audio_ffmpeg("/fake/path.wav")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 100

    def test_decode_ffmpeg_not_found(self, monkeypatch):
        """Should raise 500 when ffmpeg is not installed."""
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
        )
        with pytest.raises(HTTPException) as exc_info:
            decode_audio_ffmpeg("/fake/path.wav")
        assert exc_info.value.status_code == 500
        assert "not found" in exc_info.value.detail.lower()

    def test_decode_ffmpeg_error(self, monkeypatch):
        """Should raise 400 when ffmpeg returns non-zero."""
        mock_proc = mock.MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = b"Invalid data found"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_proc)

        with pytest.raises(HTTPException) as exc_info:
            decode_audio_ffmpeg("/fake/path.wav")
        assert exc_info.value.status_code == 400

    def test_decode_empty_output(self, monkeypatch):
        """Should raise 400 when ffmpeg produces empty output."""
        mock_proc = mock.MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b""
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_proc)

        with pytest.raises(HTTPException) as exc_info:
            decode_audio_ffmpeg("/fake/path.wav")
        assert exc_info.value.status_code == 400


class TestGetAudioDuration:
    """Test ffprobe duration extraction."""

    def test_returns_float(self, monkeypatch):
        mock_proc = mock.MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"123.456\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_proc)

        duration = get_audio_duration("/fake/path.wav")
        assert duration == 123.456

    def test_invalid_duration_raises(self, monkeypatch):
        mock_proc = mock.MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"not_a_number\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_proc)

        with pytest.raises(HTTPException) as exc_info:
            get_audio_duration("/fake/path.wav")
        assert exc_info.value.status_code == 400

    def test_zero_duration_raises(self, monkeypatch):
        mock_proc = mock.MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"0.0\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_proc)

        with pytest.raises(HTTPException) as exc_info:
            get_audio_duration("/fake/path.wav")
        assert exc_info.value.status_code == 400


class TestSaveUploadToTemp:
    """Test upload streaming with size validation (B-3)."""

    @pytest.mark.asyncio
    async def test_saves_small_file(self, monkeypatch):
        """Small upload should be saved successfully."""
        # Create a mock UploadFile
        content = b"fake audio data here"

        async def mock_read(size):
            if not hasattr(mock_read, "called"):
                mock_read.called = True
                return content
            return b""

        upload = mock.MagicMock(spec=UploadFile)
        upload.filename = "test.wav"
        upload.read = mock_read

        # Mock disk_usage to return plenty of space
        mock_usage = mock.MagicMock()
        mock_usage.free = 10 * 1024 * 1024 * 1024  # 10 GB
        monkeypatch.setattr("shutil.disk_usage", lambda path: mock_usage)

        result = await save_upload_to_temp(upload, max_size=1024 * 1024)
        assert result.exists()
        assert result.suffix == ".wav"

        # Cleanup
        result.unlink()

    @pytest.mark.asyncio
    async def test_rejects_oversized_upload(self, monkeypatch):
        """B-3: Upload exceeding max_size should be rejected BEFORE writing to disk."""
        chunk_size = 1024 * 1024  # 1 MB
        max_size = 500_000  # 500 KB — smaller than one chunk

        chunks_sent = []

        async def mock_read(size):
            chunks_sent.append(size)
            return b"x" * chunk_size

        upload = mock.MagicMock(spec=UploadFile)
        upload.filename = "large.wav"
        upload.read = mock_read

        # Mock disk_usage to return plenty of space
        mock_usage = mock.MagicMock()
        mock_usage.free = 10 * 1024 * 1024 * 1024
        monkeypatch.setattr("shutil.disk_usage", lambda path: mock_usage)

        with pytest.raises(HTTPException) as exc_info:
            await save_upload_to_temp(upload, max_size=max_size)
        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    async def test_rejects_empty_upload(self, monkeypatch):
        """Empty upload should be rejected."""
        upload = mock.MagicMock(spec=UploadFile)
        upload.filename = "empty.wav"

        async def mock_read(size):
            return b""

        upload.read = mock_read

        mock_usage = mock.MagicMock()
        mock_usage.free = 10 * 1024 * 1024 * 1024
        monkeypatch.setattr("shutil.disk_usage", lambda path: mock_usage)

        with pytest.raises(HTTPException) as exc_info:
            await save_upload_to_temp(upload, max_size=1024 * 1024)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_no_filename(self, monkeypatch):
        """Upload without filename should be rejected."""
        upload = mock.MagicMock(spec=UploadFile)
        upload.filename = None

        mock_usage = mock.MagicMock()
        mock_usage.free = 10 * 1024 * 1024 * 1024
        monkeypatch.setattr("shutil.disk_usage", lambda path: mock_usage)

        with pytest.raises(HTTPException) as exc_info:
            await save_upload_to_temp(upload)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_disk_full_rejected(self, monkeypatch):
        """B-3: When disk is nearly full, reject with 503."""
        upload = mock.MagicMock(spec=UploadFile)
        upload.filename = "test.wav"

        # Mock disk_usage to return very little free space
        mock_usage = mock.MagicMock()
        mock_usage.free = 1024  # Only 1 KB free
        monkeypatch.setattr("shutil.disk_usage", lambda path: mock_usage)

        with pytest.raises(HTTPException) as exc_info:
            await save_upload_to_temp(upload, max_size=1024 * 1024)
        assert exc_info.value.status_code == 503


class TestCleanupTempFile:
    """Test temp file cleanup."""

    def test_removes_existing_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = Path(f.name)

        assert path.exists()
        cleanup_temp_file(path)
        assert not path.exists()

    def test_ignores_missing_file(self):
        """Should not raise if file doesn't exist."""
        cleanup_temp_file("/nonexistent/path/xyz.tmp")
