"""Transcription worker — faster-whisper model wrapped in an asyncio.Queue worker.

Single GPU worker pattern:
- One WhisperModel instance loaded at startup
- asyncio.Queue serializes GPU access (prevents OOM)
- Each job gets an asyncio.Future, resolved when transcription completes
- GPU memory cleaned after each job via torch.cuda.empty_cache()
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog
import torch
from faster_whisper import WhisperModel

from app.config import settings

logger = structlog.get_logger(__name__)


def _safe_float(value: float, default: float = 0.0) -> float:
    """Coerce a float to a finite value, replacing NaN/Inf with a default.

    faster-whisper can emit NaN/Inf segment fields with corrupted models or
    certain cuDNN versions; such values produce invalid JSON (RFC 8259) and
    break client parsers. Sanitize at the source rather than at serialization.
    """
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return value


@dataclass
class TranscriptionJob:
    """A single transcription job in the queue."""

    audio: np.ndarray
    language: str
    future: asyncio.Future
    temperature: float = 0.0
    client_id: str = "unknown"


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
        # Per-client fairness: cap in-flight jobs per client so one client cannot
        # saturate the single-GPU queue with max-duration audio (DoS).
        self._active_jobs_per_client: dict[str, int] = {}
        self._max_jobs_per_client: int = max(1, settings.rate_limit_burst)

    def _decrement_client(self, client_id: str) -> None:
        """Release one in-flight slot for a client. Called via future callback."""
        active = self._active_jobs_per_client.get(client_id, 0)
        if active <= 1:
            self._active_jobs_per_client.pop(client_id, None)
        else:
            self._active_jobs_per_client[client_id] = active - 1

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
        temperature: float = 0.0,
        client_id: str = "unknown",
    ) -> TranscriptionResult:
        """Submit an audio array for transcription.

        Args:
            audio: Float32 numpy array of 16kHz mono audio.
            language: Language code (e.g. "zh", "en") or "" for auto-detect.
            temperature: Sampling temperature (0-1).
            client_id: Identifier (typically client IP) for per-client fairness.

        Returns:
            TranscriptionResult with text, language, duration, and segments.

        Raises:
            RuntimeError: If the worker is not running or is shutting down.
            asyncio.TimeoutError: If the queue is full and timeout expires.
        """
        if not self._running:
            raise RuntimeError("Transcription worker is not running")

        # Defense-in-depth: validate audio array size against max_audio_duration.
        # ffprobe reads container metadata which can be spoofed; this guards the
        # memory held by TranscriptionJob while waiting in the queue.
        max_audio_bytes = settings.max_audio_duration * 16000 * 4  # 16kHz float32
        if audio.nbytes > max_audio_bytes * 2:  # 2x safety margin
            raise ValueError(
                f"Audio array too large: {audio.nbytes} bytes exceeds limit "
                f"of {max_audio_bytes * 2} bytes"
            )

        # Per-client fairness: reject if this client already has too many jobs
        # in-flight, so one client cannot monopolize the single-GPU queue.
        active = self._active_jobs_per_client.get(client_id, 0)
        if active >= self._max_jobs_per_client:
            raise RuntimeError(
                f"Too many concurrent jobs from this client ({active}). "
                "Please wait for existing jobs to complete."
            )

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        job = TranscriptionJob(
            audio=audio,
            language=language,
            future=future,
            temperature=temperature,
            client_id=client_id,
        )

        # Reserve a slot now; release it when the future settles (resolved,
        # failed, cancelled, or rejected on shutdown) via a done callback.
        self._active_jobs_per_client[client_id] = active + 1
        future.add_done_callback(
            lambda _f, cid=client_id: self._decrement_client(cid)
        )

        # Put in queue with timeout
        try:
            await asyncio.wait_for(self._queue.put(job), timeout=30.0)
        except asyncio.TimeoutError:
            # Never queued — undo the reservation we just made.
            self._decrement_client(client_id)
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
                    result = await self._transcribe(
                        job.audio, job.language, job.temperature
                    )
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
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """Run faster-whisper transcription in a thread.

        Args:
            audio: Float32 numpy array of 16kHz mono audio.
            language: Language code or "" for auto-detect.
            temperature: Sampling temperature (0-1).

        Returns:
            TranscriptionResult.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Resolve language: if explicitly empty, try default; if default also empty, use None for auto-detect
        _lang = language.strip() if language else ""
        lang = _lang if _lang else (settings.default_language or None)

        # Run inference in a thread to avoid blocking the event loop
        segments, info = await asyncio.to_thread(
            self._model.transcribe,
            audio,
            language=lang,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=(
                [temperature, min(temperature + 0.2, 1.0), min(temperature + 0.4, 1.0)]
                if temperature < 1.0 else [temperature]
            ),
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
            # Sanitize float fields: NaN/Inf would produce invalid JSON and
            # break clients. Replace with sensible defaults.
            segment_list.append({
                "id": seg.id,
                "seek": seg.seek,
                "start": round(_safe_float(seg.start, 0.0), 2),
                "end": round(_safe_float(seg.end, 0.0), 2),
                "text": seg.text,
                "tokens": seg.tokens,
                "temperature": round(_safe_float(seg.temperature, 0.0), 2),
                "avg_logprob": round(_safe_float(seg.avg_logprob, 0.0), 4),
                "compression_ratio": round(_safe_float(seg.compression_ratio, 0.0), 4),
                "no_speech_prob": round(_safe_float(seg.no_speech_prob, 1.0), 4),
            })
            full_text_parts.append(seg.text)

        # GPU cleanup after each job
        if settings.model_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return TranscriptionResult(
            text="".join(full_text_parts),
            language=info.language,
            duration=round(_safe_float(info.duration, 0.0), 2),
            segments=segment_list,
        )
