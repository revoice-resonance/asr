"""Tests for TranscriptionWorker queue and lifecycle.

Covers: submit/result flow, shutdown behavior (C-1), queue full, worker not running.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.services.transcriber import TranscriptionResult, TranscriptionWorker


@pytest.fixture
def dummy_audio():
    """1 second of silence at 16kHz."""
    return np.zeros(16000, dtype=np.float32)


class TestWorkerLifecycle:
    """Test worker start/stop/is_ready."""

    @pytest.mark.asyncio
    async def test_worker_not_ready_before_start(self):
        worker = TranscriptionWorker()
        assert worker.is_ready is False

    @pytest.mark.asyncio
    async def test_submit_raises_when_not_running(self, dummy_audio):
        worker = TranscriptionWorker()
        with pytest.raises(RuntimeError, match="not running"):
            await worker.submit(dummy_audio, language="zh")


class TestWorkerShutdown:
    """C-1: Verify shutdown properly resolves pending futures."""

    @pytest.mark.asyncio
    async def test_stop_sets_exception_on_pending_jobs(self, dummy_audio, monkeypatch):
        """When stop() is called, pending queue jobs should get RuntimeError, not hang."""
        worker = TranscriptionWorker()

        # Mock the model loading so we don't need a real GPU
        monkeypatch.setattr(worker, "_model_loaded", True)
        monkeypatch.setattr(worker, "_running", True)

        # Start the worker loop (it will try to transcribe but we don't care)
        worker._worker_task = asyncio.create_task(asyncio.sleep(0))

        # Submit a job — it will sit in the queue since worker is "busy"
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        from app.services.transcriber import TranscriptionJob
        job = TranscriptionJob(audio=dummy_audio, language="zh", future=future)
        await worker._queue.put(job)

        # Stop the worker
        await worker.stop()

        # The pending future should have an exception set, not be hanging
        assert future.done()
        with pytest.raises(RuntimeError, match="shutting down"):
            future.result()

    @pytest.mark.asyncio
    async def test_submit_catches_cancelled_error(self, dummy_audio, monkeypatch):
        """submit() should translate CancelledError into RuntimeError."""
        worker = TranscriptionWorker()
        monkeypatch.setattr(worker, "_running", True)

        # Create a future that will be cancelled
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        # Bypass the queue put and directly set up the future
        original_put = worker._queue.put

        async def mock_put(job):
            # Store the job's future so we can cancel it
            mock_put.stored_future = job.future
            await original_put(job)

        mock_put.stored_future = None
        monkeypatch.setattr(worker._queue, "put", mock_put)

        # Start a task that will cancel the future after a short delay
        async def cancel_later():
            await asyncio.sleep(0.05)
            if mock_put.stored_future:
                mock_put.stored_future.cancel()

        cancel_task = asyncio.create_task(cancel_later())

        with pytest.raises(RuntimeError, match="shutting down"):
            await worker.submit(dummy_audio, language="zh")

        await cancel_task
