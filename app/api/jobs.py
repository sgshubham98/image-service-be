"""Job management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.schemas.job import JobDetail, JobList
from app.services.job_service import cancel_job, get_job, list_jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=JobList)
async def list_all_jobs(
    status: str | None = Query(None, description="Filter by status"),
    batch_id: str | None = Query(None, description="Filter by batch ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    jobs, total = await list_jobs(session, status=status, batch_id=batch_id, page=page, page_size=page_size)
    return JobList(jobs=jobs, total=total, page=page, page_size=page_size)


@router.get("/{job_id}", response_model=JobDetail)
async def get_job_detail(
    job_id: str,
    session: AsyncSession = Depends(get_session),
):
    job = await get_job(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/{job_id}", response_model=JobDetail)
async def cancel_job_endpoint(
    job_id: str,
    session: AsyncSession = Depends(get_session),
):
    job = await cancel_job(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
