"""GPU worker — async priority queue consumer driving the Flux engine."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import torch
from sqlalchemy import select, func

from app.database import async_session_factory
from app.engine.flux_engine import FluxEngine
from app.models.enums import ImageFormat, JobStatus
from app.models.job import BatchJob, Job
from app.utils.image_utils import save_image
from app.utils.storage import get_image_path, image_url_path

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _QueueItem:
    """Priority queue item. Lower priority value = higher urgency."""

    priority: int
    timestamp: float = field(compare=True)
    job_id: str = field(compare=False)


@dataclass
class GPUWorker:
    """Consumes jobs from an async priority queue and runs inference."""

    engine: FluxEngine
    _queue: asyncio.PriorityQueue[_QueueItem] = field(default_factory=asyncio.PriorityQueue)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _task: asyncio.Task | None = field(default=None, repr=False)
    _jobs_processed: int = 0
    _total_inference_time: float = 0.0

    # ── Queue interface ──────────────────────────────────────────────

    async def enqueue(self, job_id: str, priority: int = 1) -> None:
        item = _QueueItem(priority=priority, timestamp=time.time(), job_id=job_id)
        await self._queue.put(item)
        logger.debug("Enqueued job=%s priority=%d queue_depth=%d", job_id, priority, self.queue_depth)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> asyncio.Task:
        """Start the worker loop as a background task."""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="gpu-worker")
        logger.info("GPU worker started.")
        return self._task

    async def shutdown(self) -> None:
        """Signal the worker to stop and wait for it to finish the current job."""
        logger.info("Shutting down GPU worker (queue_depth=%d)...", self.queue_depth)
        self._stop_event.set()
        if self._task and not self._task.done():
            # Unblock a pending queue.get() by pushing a sentinel
            await self._queue.put(_QueueItem(priority=999, timestamp=time.time(), job_id="__stop__"))
            await self._task
        logger.info("GPU worker stopped. Processed %d jobs total.", self._jobs_processed)

    # ── Resume support ───────────────────────────────────────────────

    async def resume_pending_jobs(self) -> int:
        """Re-enqueue jobs left in PENDING or PROCESSING state (crash recovery)."""
        async with async_session_factory() as session:
            result = await session.execute(
                select(Job).where(Job.status.in_(["pending", "processing"]))
            )
            stale_jobs = result.scalars().all()

            for job in stale_jobs:
                job.status = "pending"
                await self.enqueue(job.id, job.priority)

            await session.commit()

        if stale_jobs:
            logger.info("Resumed %d pending/stale jobs.", len(stale_jobs))
        return len(stale_jobs)

    # ── Main loop ────────────────────────────────────────────────────

    async def _run(self) -> None:
        logger.info("GPU worker loop running.")
        while not self._stop_event.is_set():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if item.job_id == "__stop__":
                break

            await self._process_job(item.job_id)
            self._queue.task_done()

        logger.info("GPU worker loop exited.")

    async def _process_job(self, job_id: str) -> None:
        async with async_session_factory() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job is None:
                logger.warning("Job %s not found in DB, skipping.", job_id)
                return

            # Skip cancelled jobs
            if job.status == "cancelled":
                logger.info("Job %s is cancelled, skipping.", job_id)
                return

            # Mark as processing
            job.status = "processing"
            job.started_at = datetime.now(timezone.utc)
            await session.commit()

        try:
            # Run inference in a thread to avoid blocking asyncio
            gen_result = await asyncio.to_thread(
                self.engine.generate,
                prompt=job.prompt,
                width=job.width,
                height=job.height,
                num_steps=job.num_steps,
                guidance_scale=job.guidance_scale,
                seed=job.seed,
            )

            # Save image to disk
            fmt = ImageFormat(job.format)
            file_path = get_image_path(job.id, job.format, batch_id=job.batch_id)
            save_image(
                gen_result.image,
                file_path,
                fmt,
                prompt=job.prompt,
                seed=gen_result.seed,
            )

            # Update job as completed
            async with async_session_factory() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one()
                job.status = "completed"
                job.file_path = str(file_path)
                job.seed = gen_result.seed
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()

                # Update batch counters if applicable
                if job.batch_id:
                    await self._update_batch_count(session, job.batch_id)

            self._jobs_processed += 1
            self._total_inference_time += gen_result.inference_time

        except torch.cuda.OutOfMemoryError:
            logger.error("OOM on job %s (%dx%d). Clearing cache.", job_id, job.width, job.height)
            torch.cuda.empty_cache()
            await self._fail_job(job_id, "GPU out of memory for this resolution")

        except Exception:
            logger.exception("Error processing job %s", job_id)
            await self._fail_job(job_id, "Internal generation error")

    async def _fail_job(self, job_id: str, error: str) -> None:
        async with async_session_factory() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.error_message = error
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()

                if job.batch_id:
                    await self._update_batch_count(session, job.batch_id)

    async def _update_batch_count(self, session, batch_id: str) -> None:
        """Recompute batch counters from individual jobs."""
        result = await session.execute(select(BatchJob).where(BatchJob.id == batch_id))
        batch = result.scalar_one_or_none()
        if not batch:
            return

        # Count completed and failed
        completed = (
            await session.execute(
                select(func.count()).where(Job.batch_id == batch_id, Job.status == "completed")
            )
        ).scalar()
        failed = (
            await session.execute(
                select(func.count()).where(Job.batch_id == batch_id, Job.status == "failed")
            )
        ).scalar()

        batch.completed_count = completed or 0
        batch.failed_count = failed or 0

        done = (completed or 0) + (failed or 0)
        if done >= batch.total_count:
            batch.status = "completed" if (failed or 0) == 0 else "failed"
            batch.completed_at = datetime.now(timezone.utc)
        elif done > 0:
            batch.status = "processing"

        await session.commit()

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        avg = (
            self._total_inference_time / self._jobs_processed
            if self._jobs_processed > 0
            else 0.0
        )
        return {
            "queue_depth": self.queue_depth,
            "jobs_processed": self._jobs_processed,
            "total_inference_time_s": round(self._total_inference_time, 2),
            "avg_inference_time_s": round(avg, 2),
        }
