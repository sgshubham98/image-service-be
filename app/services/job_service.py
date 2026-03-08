"""Job and batch CRUD operations."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import JobStatus
from app.models.job import BatchJob, Job
from app.schemas.job import JobDetail
from app.utils.storage import image_url_path


def _job_to_detail(job: Job) -> JobDetail:
    return JobDetail(
        id=job.id,
        prompt=job.prompt,
        negative_prompt=job.negative_prompt,
        width=job.width,
        height=job.height,
        num_steps=job.num_steps,
        guidance_scale=job.guidance_scale,
        seed=job.seed,
        status=job.status,
        priority=job.priority,
        format=job.format,
        file_path=job.file_path,
        image_url=image_url_path(job.id) if job.status == "completed" else None,
        error_message=job.error_message,
        batch_id=job.batch_id,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


async def get_job(session: AsyncSession, job_id: str) -> JobDetail | None:
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        return None
    return _job_to_detail(job)


async def list_jobs(
    session: AsyncSession,
    status: str | None = None,
    batch_id: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[JobDetail], int]:
    query = select(Job).order_by(Job.created_at.desc())
    count_query = select(func.count()).select_from(Job)

    if status:
        query = query.where(Job.status == status)
        count_query = count_query.where(Job.status == status)
    if batch_id:
        query = query.where(Job.batch_id == batch_id)
        count_query = count_query.where(Job.batch_id == batch_id)

    total = (await session.execute(count_query)).scalar() or 0
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(query)
    jobs = [_job_to_detail(j) for j in result.scalars().all()]
    return jobs, total


async def cancel_job(session: AsyncSession, job_id: str) -> JobDetail | None:
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        return None
    if job.status in ("completed", "failed", "cancelled"):
        return _job_to_detail(job)

    job.status = "cancelled"
    job.completed_at = datetime.now(timezone.utc)
    await session.commit()
    return _job_to_detail(job)


async def get_batch_progress(session: AsyncSession, batch_id: str) -> dict | None:
    result = await session.execute(select(BatchJob).where(BatchJob.id == batch_id))
    batch = result.scalar_one_or_none()
    if not batch:
        return None

    cancelled = (
        await session.execute(
            select(func.count()).where(Job.batch_id == batch_id, Job.status == "cancelled")
        )
    ).scalar() or 0

    pending = batch.total_count - batch.completed_count - batch.failed_count - cancelled

    return {
        "batch_id": batch.id,
        "name": batch.name,
        "total": batch.total_count,
        "completed": batch.completed_count,
        "failed": batch.failed_count,
        "cancelled": cancelled,
        "pending": pending,
        "status": batch.status,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
    }


async def cancel_batch(session: AsyncSession, batch_id: str) -> dict | None:
    result = await session.execute(select(BatchJob).where(BatchJob.id == batch_id))
    batch = result.scalar_one_or_none()
    if not batch:
        return None

    # Cancel all pending jobs in this batch
    pending_result = await session.execute(
        select(Job).where(Job.batch_id == batch_id, Job.status == "pending")
    )
    now = datetime.now(timezone.utc)
    for job in pending_result.scalars().all():
        job.status = "cancelled"
        job.completed_at = now

    batch.status = "cancelled"
    batch.completed_at = now
    await session.commit()

    return await get_batch_progress(session, batch_id)


async def retry_failed_in_batch(session: AsyncSession, batch_id: str) -> list[str]:
    """Re-set failed jobs in a batch back to pending. Returns list of job IDs."""
    result = await session.execute(
        select(Job).where(Job.batch_id == batch_id, Job.status == "failed")
    )
    job_ids = []
    for job in result.scalars().all():
        job.status = "pending"
        job.error_message = None
        job.completed_at = None
        job_ids.append(job.id)

    if job_ids:
        # Reset batch status
        batch_result = await session.execute(select(BatchJob).where(BatchJob.id == batch_id))
        batch = batch_result.scalar_one_or_none()
        if batch:
            batch.status = "processing"
            batch.completed_at = None
            batch.failed_count = 0

        await session.commit()

    return job_ids
