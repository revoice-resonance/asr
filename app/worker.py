"""Task scheduler — polls the database for PENDING ASR tasks and processes them.

The TaskScheduler runs as a background asyncio task. It periodically queries
the database for PENDING tasks, atomically claims them, loads and decodes the
audio, submits to the GPU worker, stores results, and triggers callbacks.

This sits ABOVE the TranscriptionWorker — the worker remains a pure GPU queue.
The scheduler is the DB-aware orchestrator that feeds it.
"""

from __future__ import annotations

import asyncio
import os
import time

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.config import settings
from app.models.corpus import Corpus
from app.services.audio import decode_audio_ffmpeg
from app.services.callback import notify_main_backend
from app.services.corpus import (
    claim_pending_task,
    complete_task,
    fail_task,
    sync_text_content,
)
from app.services.transcriber import TranscriptionWorker

logger = structlog.get_logger(__name__)


class TaskScheduler:
    """Background task that polls the DB for PENDING ASR tasks.

    For each pending task found:
    1. Atomically claim (UPDATE status='PROCESSING', started_at=NOW())
    2. Load the audio file from permanent storage
    3. Decode to numpy via ffmpeg (thread)
    4. Submit to TranscriptionWorker queue → await result
    5. Store result in asr_tasks + sync text_content to corpus
    6. POST callback to main backend
    7. On failure: mark FAILED, retry up to task_max_retries

    The scheduler respects max_concurrent_tasks — it won't claim more tasks
    than the configured limit.

    Usage:
        scheduler = TaskScheduler(session_factory, worker)
        await scheduler.start()
        ...
        await scheduler.stop()
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        worker: TranscriptionWorker,
    ) -> None:
        self._session_factory = session_factory
        self._worker = worker
        self._running = False
        self._task: asyncio.Task | None = None
        self._active_count = 0
        self._max_concurrent = max(1, settings.max_concurrent_tasks)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def active_count(self) -> int:
        return self._active_count

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Task scheduler started",
            poll_interval=settings.task_poll_interval,
            max_concurrent=self._max_concurrent,
        )

    async def stop(self) -> None:
        """Gracefully stop the scheduler. Waits for in-flight tasks to complete."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Task scheduler stopped", remaining_tasks=self._active_count)

    async def _poll_loop(self) -> None:
        """Main loop: poll DB, claim tasks, process them."""
        while self._running:
            try:
                # Check for PENDING tasks if we have capacity
                while self._active_count < self._max_concurrent:
                    task_claimed = await self._try_claim_and_process()
                    if not task_claimed:
                        break  # No more PENDING tasks

                await asyncio.sleep(settings.task_poll_interval)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler poll error — will retry")
                await asyncio.sleep(settings.task_poll_interval)

    async def _try_claim_and_process(self) -> bool:
        """Try to claim one PENDING task and start processing it.

        Returns True if a task was claimed, False if none available.
        """
        try:
            async with self._session_factory() as db:
                task = await claim_pending_task(db)
                if task is None:
                    await db.commit()
                    return False

                # Load corpus to get audio file path
                corpus = await db.get(Corpus, task.corpus_id)
                if corpus is None or corpus.file_path is None:
                    await fail_task(db, task.id, "Corpus record missing or file_path is null")
                    await db.commit()
                    await self._notify(task.id, task.corpus_id, "FAILED", error="Corpus missing")
                    return True

                task_id = task.id
                corpus_id = corpus.id
                audio_path = corpus.file_path
                await db.commit()

            # Process outside the DB session (avoids holding a connection during GPU inference)
            self._active_count += 1
            asyncio.create_task(self._process_task(task_id, corpus_id, audio_path))
            return True

        except Exception:
            logger.exception("Failed to claim task")
            return False

    async def _process_task(
        self,
        task_id: int,
        corpus_id: int,
        audio_path: str,
    ) -> None:
        """Load audio, submit to GPU worker, store results, trigger callback."""
        try:
            # Load and decode audio
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            audio = await asyncio.to_thread(decode_audio_ffmpeg, audio_path)

            # Submit to GPU worker (reuses the same asyncio.Queue)
            start = time.monotonic()
            result = await self._worker.submit(audio)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            # Extract confidence from segment avg_logprobs
            confidence = None
            if result.segments:
                logprobs = [s.get("avg_logprob", 0.0) for s in result.segments]
                logprobs = [lp for lp in logprobs if lp != 0.0]
                if logprobs:
                    confidence = round(sum(logprobs) / len(logprobs), 4)

            # Store results in a fresh session
            async with self._session_factory() as db:
                await complete_task(
                    db,
                    task_id=task_id,
                    result_text=result.text,
                    confidence=confidence,
                    result_detail={
                        "language": result.language,
                        "duration": result.duration,
                        "segments": result.segments,
                    },
                    processing_time=elapsed_ms,
                )
                await sync_text_content(db, corpus_id, result.text)
                await db.commit()

            await self._notify(
                task_id, corpus_id, "SUCCESS",
                result_text=result.text,
                confidence=confidence,
            )

        except Exception as exc:
            logger.exception("Task processing failed", task_id=task_id)
            error_msg = f"{type(exc).__name__}: {exc}"
            try:
                async with self._session_factory() as db:
                    await fail_task(db, task_id=task_id, error_message=error_msg)
                    await db.commit()
            except Exception:
                logger.exception("Failed to update task as FAILED", task_id=task_id)

            await self._notify(task_id, corpus_id, "FAILED", error=error_msg)

        finally:
            self._active_count -= 1

    async def _notify(
        self,
        task_id: int,
        corpus_id: int,
        status: str,
        result_text: str | None = None,
        confidence: float | None = None,
        error: str | None = None,
    ) -> None:
        """Fire-and-forget callback to main backend."""
        if not settings.main_backend_callback_url.strip():
            return
        try:
            await notify_main_backend(
                task_id=task_id,
                corpus_id=corpus_id,
                status=status,
                result_text=result_text,
                confidence=confidence,
                error_message=error,
            )
        except Exception:
            logger.exception("Callback failed unexpectedly", task_id=task_id)
