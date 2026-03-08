"""Orchestrates image generation: validate → persist → enqueue."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import JobPriority
from app.models.job import BatchJob, Job
from app.schemas.generate import ImageRequest

if TYPE_CHECKING:
    from app.worker.gpu_worker import GPUWorker


async def create_single_job(
    session: AsyncSession,
    request: ImageRequest,
    worker: "GPUWorker",
    priority: int = JobPriority.REALTIME,
) -> list[str]:
    """Create generation job(s) for the request. Returns list of job_ids.

    Creates one job per requested image (num_images). Each job gets a unique
    seed derived from the base seed (if provided) to ensure varied outputs.
    """
    job_ids: list[str] = []

    for i in range(request.num_images):
        # Derive per-image seed: if user provided a seed, offset it per image
        seed = None
        if request.seed is not None:
            seed = (request.seed + i) % (2**32)

        job = Job(
            id=str(uuid.uuid4()),
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            seed=seed,
            format=request.format.value,
            priority=priority,
            status="pending",
        )
        session.add(job)
        job_ids.append(job.id)

    await session.commit()

    for jid in job_ids:
        await worker.enqueue(jid, priority)

    return job_ids


async def create_batch(
    session: AsyncSession,
    name: str | None,
    prompts: list[ImageRequest],
    worker: "GPUWorker",
) -> str:
    """Create a batch of generation jobs. Returns batch_id."""
    batch_id = str(uuid.uuid4())
    batch = BatchJob(
        id=batch_id,
        name=name,
        total_count=len(prompts),
        status="pending",
    )
    session.add(batch)

    jobs: list[Job] = []
    for req in prompts:
        job = Job(
            id=str(uuid.uuid4()),
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_steps=req.num_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            format=req.format.value,
            priority=JobPriority.BATCH,
            status="pending",
            batch_id=batch_id,
        )
        jobs.append(job)

    session.add_all(jobs)
    await session.commit()

    # Enqueue all jobs
    for job in jobs:
        await worker.enqueue(job.id, JobPriority.BATCH)

    return batch_id
