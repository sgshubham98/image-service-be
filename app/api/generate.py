"""Image generation endpoints — async and sync modes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.job import Job
from app.schemas.generate import ImageRequest, ImageResponse
from app.services.generation_service import create_single_job
from app.services.job_service import get_job
from app.services.moderation_service import ModerationError, check_prompt_safety
from app.utils.storage import image_url_path

router = APIRouter(tags=["generate"])

_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


def _get_worker(request: Request):
    return request.app.state.worker


@router.post("/generate", response_model=ImageResponse)
async def generate_async(
    body: ImageRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Submit an image generation request (1–4 images). Returns immediately with job IDs."""
    try:
        is_safe, reason = await check_prompt_safety(body.prompt)
    except ModerationError:
        raise HTTPException(status_code=503, detail="Content moderation service unavailable")
    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt contains unsafe or explicit content: {reason}",
        )

    worker = _get_worker(request)
    job_ids = await create_single_job(session, body, worker)
    return ImageResponse(job_ids=job_ids, status="pending")


@router.get("/generate/stream")
async def stream_generate_progress(
    request: Request,
    job_id: list[str] = Query(..., description="One or more job IDs to monitor"),
    session: AsyncSession = Depends(get_session),
):
    """SSE stream for one or more generate job IDs until all reach terminal state."""
    unique_job_ids = list(dict.fromkeys(job_id))

    # Fail fast if any requested job ID does not exist.
    checks = [await get_job(session, jid) for jid in unique_job_ids]
    missing = [jid for jid, detail in zip(unique_job_ids, checks, strict=False) if detail is None]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {', '.join(missing)}",
        )

    async def event_generator() -> AsyncGenerator[dict, None]:
        while True:
            if await request.is_disconnected():
                break

            # Expire identity map so concurrent worker updates are visible.
            session.expire_all()
            details = [await get_job(session, jid) for jid in unique_job_ids]

            if any(d is None for d in details):
                yield {
                    "event": "error",
                    "data": json.dumps({"detail": "Job no longer exists"}),
                }
                break

            jobs = [d.model_dump() for d in details if d is not None]
            total = len(jobs)
            completed = sum(1 for d in details if d and d.status == "completed")
            failed = sum(1 for d in details if d and d.status == "failed")
            cancelled = sum(1 for d in details if d and d.status == "cancelled")
            terminal = sum(1 for d in details if d and d.status in _TERMINAL_JOB_STATUSES)

            payload = {
                "total": total,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "pending": total - terminal,
                "jobs": jobs,
            }
            yield {"event": "progress", "data": json.dumps(payload)}

            if terminal == total:
                yield {"event": "done", "data": json.dumps(payload)}
                break

            await asyncio.sleep(1.0)

    return EventSourceResponse(event_generator())


@router.post("/generate/sync", response_model=ImageResponse)
async def generate_sync(
    body: ImageRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Submit and wait for image(s) to be generated (up to SYNC_TIMEOUT seconds)."""
    try:
        is_safe, reason = await check_prompt_safety(body.prompt)
    except ModerationError:
        raise HTTPException(status_code=503, detail="Content moderation service unavailable")
    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt contains unsafe or explicit content: {reason}",
        )

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
