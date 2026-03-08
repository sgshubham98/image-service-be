"""Image generation endpoints — async and sync modes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.job import Job
from app.schemas.generate import ImageRequest, ImageResponse
from app.services.generation_service import create_single_job
from app.services.job_service import get_job
from app.utils.storage import image_url_path

router = APIRouter(tags=["generate"])


def _get_worker(request: Request):
    return request.app.state.worker


@router.post("/generate", response_model=ImageResponse)
async def generate_async(
    body: ImageRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Submit an image generation request (1–4 images). Returns immediately with job IDs."""
    worker = _get_worker(request)
    job_ids = await create_single_job(session, body, worker)
    return ImageResponse(job_ids=job_ids, status="pending")


@router.post("/generate/sync", response_model=ImageResponse)
async def generate_sync(
    body: ImageRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Submit and wait for image(s) to be generated (up to SYNC_TIMEOUT seconds)."""
    worker = _get_worker(request)
    job_ids = await create_single_job(session, body, worker)

    # Poll until all jobs complete or timeout
    deadline = asyncio.get_event_loop().time() + settings.SYNC_TIMEOUT
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.3)
        details = [await get_job(session, jid) for jid in job_ids]
        if all(d and d.status in ("completed", "failed", "cancelled") for d in details):
            break
    else:
        raise HTTPException(status_code=408, detail="Generation timed out")

    failed = [d for d in details if d and d.status == "failed"]
    if failed and len(failed) == len(details):
        raise HTTPException(
            status_code=500,
            detail=failed[0].error_message or "Generation failed",
        )

    return ImageResponse(
        job_ids=job_ids,
        status="completed",
        image_urls=[d.image_url if d and d.status == "completed" else None for d in details],
    )


@router.get("/jobs/{job_id}/image")
async def serve_image(
    job_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Serve the generated image file for a completed job."""
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.file_path:
        raise HTTPException(status_code=404, detail="Image not available")

    path = Path(job.file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image file missing from disk")

    media_types = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "webp": "image/webp",
    }
    ext = path.suffix.lstrip(".")
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(path, media_type=media_type, filename=path.name)
