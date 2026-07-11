"""Task scheduler — polls the database for PENDING ASR tasks and processes them.

The TaskScheduler runs as a background asyncio task. It periodically queries
the database for PENDING tasks, atomically claims them, and delegates the full
processing pipeline to TaskProcessor.

This sits ABOVE the TranscriptionWorker — the worker remains a pure GPU queue.
The scheduler is the DB-aware orchestrator that feeds it.
"""

from __future__ import annotations

import asyncio

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.config import Settings
from app.services.task_processor import TaskProcessor
from app.services.transcriber import TranscriptionWorker

logger = structlog.get_logger(__name__)


class TaskScheduler:
    """Background task that polls the DB for PENDING ASR tasks.

    For each pending task found:
    1. Atomically claim (UPDATE status='PROCESSING', started_at=NOW())
    2. Delegate to TaskProcessor for audio load → decode → transcribe → store → callback
    3. On failure: mark FAILED, retry up to task_max_retries

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
        settings: Settings | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._worker = worker
        self._settings = Settings.resolve(settings)
        self._processor = TaskProcessor(session_factory, worker, settings)
        self._running = False
        self._task: asyncio.Task | None = None
        self._active_count = 0
        self._max_concurrent = max(1, self._settings.max_concurrent_tasks)

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
            poll_interval=self._settings.task_poll_interval,
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

                await asyncio.sleep(self._settings.task_poll_interval)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler poll error — will retry")
                await asyncio.sleep(self._settings.task_poll_interval)

    async def _try_claim_and_process(self) -> bool:
        """Try to claim one PENDING task and start processing it.

        Returns True if a task was claimed, False if none available.
        """
        claimed = await self._processor.try_claim()
        if claimed is None:
            return False

        task_id, corpus_id, audio_path = claimed
        self._active_count += 1
        asyncio.create_task(self._process_and_track(task_id, corpus_id, audio_path))
        return True

    async def _process_and_track(
        self,
        task_id: int,
        corpus_id: int,
        audio_path: str,
    ) -> None:
        """Delegate to TaskProcessor and manage active_count tracking."""
        try:
            await self._processor.process(task_id, corpus_id, audio_path)
        finally:
            self._active_count -= 1
