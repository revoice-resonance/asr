"""Transcription worker — faster-whisper model wrapped in an asyncio.Queue worker.

Single GPU worker pattern:
- One WhisperModel instance loaded at startup
- asyncio.Queue serializes GPU access (prevents OOM)
- Each job gets an asyncio.Future, resolved when transcription completes
- GPU memory cleaned after each job via torch.cuda.empty_cache()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog
import torch
from faster_whisper import WhisperModel

from app.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class TranscriptionJob:
    """A single transcription job in the queue."""

    audio: np.ndarray
    language: str
    future: asyncio.Future


@dataclass
class TranscriptionResult:
    """Result of a transcription job."""

    text: str
    language: str
    duration: float
    segments: list[dict]


class TranscriptionWorker:
    """Manages the faster-whisper model and GPU worker queue.

    Usage:
        worker = TranscriptionWorker()
        await worker.start()
        result = await worker.submit(audio, language="zh")
        await worker.stop()
    """

    def __init__(self) -> None:
        self._model: Optional[WhisperModel] = None
        self._queue: asyncio.Queue[TranscriptionJob] = asyncio.Queue(
            maxsize=settings.rate_limit_burst * 2 or 20
        )
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._model_loaded = False

    # --- Lifecycle ---

    async def start(self) -> None:
        """Load the model and start the background worker task."""
        if self._running:
            return

        logger.info(
            "Loading faster-whisper model",
            model_path=settings.model_path,
            device=settings.model_device,
            compute_type=settings.model_compute_type,
        )

        model_path = str(settings.model_path_resolved)

        # Load model in a thread to avoid blocking the event loop
        self._model = await asyncio.to_thread(
            WhisperModel,
            model_path,
            device=settings.model_device,
            device_index=settings.model_device_index,
            compute_type=settings.model_compute_type,
            cpu_threads=4,
            num_workers=1,
        )

        self._model_loaded = True
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

        logger.info("Transcription worker started")

    async def stop(self) -> None:
        """Gracefully stop the worker and clean up GPU resources."""
        self._running = False

        # Cancel pending jobs with explicit exception so callers get a
        # proper error instead of hanging forever (C-1 fix).
        while not self._queue.empty():
            try:
                job = self._queue.get_nowait()
                if not job.future.done():
                    job.future.set_exception(
                        RuntimeError("Server is shutting down — please retry")
                    )
            except asyncio.QueueEmpty:
                break

        # Wait for worker to finish current job
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # Clean up GPU
        if self._model is not None:
            del self._model
            self._model = None

        if settings.model_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model_loaded = False
        logger.info("Transcription worker stopped")

    # --- Job Submission ---

    async def submit(
        self,
        audio: np.ndarray,
        language: str = "",
    ) -> TranscriptionResult:
        """Submit an audio array for transcription.

        Args:
            audio: Float32 numpy array of 16kHz mono audio.
            language: Language code (e.g. "zh", "en") or "" for auto-detect.

        Returns:
            TranscriptionResult with text, language, duration, and segments.

        Raises:
            RuntimeError: If the worker is not running or is shutting down.
            asyncio.TimeoutError: If the queue is full and timeout expires.
        """
        if not self._running:
            raise RuntimeError("Transcription worker is not running")

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        job = TranscriptionJob(audio=audio, language=language, future=future)

        # Put in queue with timeout
        try:
            await asyncio.wait_for(self._queue.put(job), timeout=30.0)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Transcription queue is full (depth={self._queue.qsize()}). "
                "Try again later."
            )

        # Wait for result — translate CancelledError / shutdown RuntimeError
        # into a proper error so the client gets a 503 instead of hanging (C-1).
        try:
            return await future
        except asyncio.CancelledError:
            raise RuntimeError("Server is shutting down — please retry")
        except RuntimeError:
            # Re-raise RuntimeError from shutdown so route can map to 503
            raise

    # --- Properties ---

    @property
    def queue_depth(self) -> int:
        """Current number of jobs waiting in the queue."""
        return self._queue.qsize()

    @property
    def is_ready(self) -> bool:
        """Whether the worker is ready to accept jobs."""
        return self._running and self._model_loaded

    # --- Internal ---

    async def _worker_loop(self) -> None:
        """Main worker loop: dequeue jobs, run inference, resolve futures."""
        try:
            while self._running:
                try:
                    # Wait for a job with a timeout so we can check _running
                    job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if job.future.done():
                    continue

                try:
                    start = time.monotonic()
                    result = await self._transcribe(job.audio, job.language)
                    elapsed = time.monotonic() - start

                    logger.info(
                        "Transcription complete",
                        language=result.language,
                        duration=result.duration,
                        elapsed=round(elapsed, 2),
                        queue_depth=self._queue.qsize(),
                    )

                    job.future.set_result(result)

                except Exception as exc:
                    logger.error(
                        "Transcription failed",
                        error=str(exc),
                        queue_depth=self._queue.qsize(),
                    )
                    if not job.future.done():
                        job.future.set_exception(exc)
        except asyncio.CancelledError:
            # Worker is being stopped — drain remaining queued futures so
            # callers don't hang (C-1).
            while not self._queue.empty():
                try:
                    job = self._queue.get_nowait()
                    if not job.future.done():
                        job.future.set_exception(
                            RuntimeError("Worker stopped — please retry")
                        )
                except asyncio.QueueEmpty:
                    break
            raise

    async def _transcribe(
        self,
        audio: np.ndarray,
        language: str,
    ) -> TranscriptionResult:
        """Run faster-whisper transcription in a thread.

        Args:
            audio: Float32 numpy array of 16kHz mono audio.
            language: Language code or "" for auto-detect.

        Returns:
            TranscriptionResult.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Resolve language
        lang = language or settings.default_language or None

        # Run inference in a thread to avoid blocking the event loop
        segments, info = await asyncio.to_thread(
            self._model.transcribe,
            audio,
            language=lang,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4],
            vad_filter=settings.vad_enabled,
            vad_parameters=dict(
                threshold=settings.vad_threshold,
                min_silence_duration_ms=settings.vad_min_silence_duration_ms,
            ) if settings.vad_enabled else None,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            word_timestamps=False,
        )

        # Collect segments
        segment_list: list[dict] = []
        full_text_parts: list[str] = []

        for seg in segments:
            segment_list.append({
                "id": seg.id,
                "seek": seg.seek,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "tokens": seg.tokens,
                "temperature": round(seg.temperature, 2),
                "avg_logprob": round(seg.avg_logprob, 4),
                "compression_ratio": round(seg.compression_ratio, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
            })
            full_text_parts.append(seg.text.strip())

        # GPU cleanup after each job
        if settings.model_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return TranscriptionResult(
            text="".join(full_text_parts),
            language=info.language,
            duration=round(info.duration, 2),
            segments=segment_list,
        )
