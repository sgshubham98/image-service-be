"""Batch / dataset generation endpoints with SSE progress streaming."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import zipfile
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.job import Job
from app.models.enums import ImageFormat
from app.schemas.batch import BatchProgress, BatchRequest
from app.schemas.generate import ImageRequest
from app.services.generation_service import create_batch
from app.services.job_service import (
    cancel_batch,
    get_batch_progress,
    retry_failed_in_batch,
)

router = APIRouter(prefix="/batch", tags=["batch"])


def _get_worker(request: Request):
    return request.app.state.worker


@router.post("", response_model=BatchProgress)
async def create_batch_job(
    body: BatchRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Create a batch of image generation jobs from a list of prompts."""
    worker = _get_worker(request)
    batch_id = await create_batch(session, body.name, body.prompts, worker)
    progress = await get_batch_progress(session, batch_id)
    return BatchProgress(**progress)


@router.post("/from-file", response_model=BatchProgress)
async def create_batch_from_file(
    request: Request,
    file: UploadFile = File(...),
    name: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Create a batch from a CSV or JSON file upload.

    CSV format: prompt,width,height,num_steps,seed,format
    JSON format: array of ImageRequest objects
    """
    worker = _get_worker(request)
    content = await file.read()

    filename = file.filename or ""
    if filename.endswith(".json"):
        prompts = _parse_json(content)
    elif filename.endswith(".csv"):
        prompts = _parse_csv(content)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Use .csv or .json",
        )

    if not prompts:
        raise HTTPException(status_code=400, detail="No valid prompts found in file")

    batch_id = await create_batch(session, name or filename, prompts, worker)
    progress = await get_batch_progress(session, batch_id)
    return BatchProgress(**progress)


@router.get("/{batch_id}", response_model=BatchProgress)
async def get_batch_status(
    batch_id: str,
    session: AsyncSession = Depends(get_session),
):
    progress = await get_batch_progress(session, batch_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Batch not found")
    return BatchProgress(**progress)


@router.get("/{batch_id}/stream")
async def stream_batch_progress(
    batch_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """SSE endpoint streaming batch progress updates until completion."""
    progress = await get_batch_progress(session, batch_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Batch not found")

    async def event_generator() -> AsyncGenerator[dict, None]:
        while True:
            if await request.is_disconnected():
                break

            async with _fresh_session() as sess:
                p = await get_batch_progress(sess, batch_id)

            if p is None:
                break

            # Calculate ETA
            eta = _estimate_eta(p, request.app.state.worker)

            yield {
                "event": "progress",
                "data": json.dumps({**p, "estimated_remaining_seconds": eta}),
            }

            if p["status"] in ("completed", "failed", "cancelled"):
                yield {"event": "done", "data": json.dumps(p)}
                break

            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


@router.get("/{batch_id}/download")
async def download_batch_images(
    batch_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Download all completed images in a batch as a ZIP archive."""
    progress = await get_batch_progress(session, batch_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Batch not found")

    result = await session.execute(
        select(Job).where(Job.batch_id == batch_id, Job.status == "completed")
    )
    jobs = result.scalars().all()

    if not jobs:
        raise HTTPException(status_code=404, detail="No completed images in this batch")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for job in jobs:
            if not job.file_path:
                continue
            file_path = Path(job.file_path)
            if file_path.is_file():
                zf.write(file_path, file_path.name)
    buf.seek(0)

    batch_name = progress.get("name") or batch_id[:12]
    safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in batch_name).strip()
    filename = f"{safe_name}.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{batch_id}", response_model=BatchProgress)
async def cancel_batch_job(
    batch_id: str,
    session: AsyncSession = Depends(get_session),
):
    result = await cancel_batch(session, batch_id)
    if not result:
        raise HTTPException(status_code=404, detail="Batch not found")
    return BatchProgress(**result)


@router.post("/{batch_id}/retry-failed")
async def retry_failed_jobs(
    batch_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Re-enqueue all failed jobs in a batch."""
    worker = _get_worker(request)
    job_ids = await retry_failed_in_batch(session, batch_id)
    if not job_ids:
        raise HTTPException(status_code=404, detail="No failed jobs found in batch")

    for jid in job_ids:
        await worker.enqueue(jid, priority=5)

    return {"retried": len(job_ids), "job_ids": job_ids}


# ── File parsing helpers ─────────────────────────────────────────


def _parse_json(content: bytes) -> list[ImageRequest]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="JSON must be an array of objects")

    prompts = []
    for i, item in enumerate(data):
        try:
            prompts.append(ImageRequest(**item))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid entry at index {i}: {e}")
    return prompts


def _parse_csv(content: bytes) -> list[ImageRequest]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")

    reader = csv.DictReader(io.StringIO(text))
    prompts = []
    for i, row in enumerate(reader):
        try:
            kwargs: dict = {"prompt": row["prompt"]}
            if "width" in row and row["width"]:
                kwargs["width"] = int(row["width"])
            if "height" in row and row["height"]:
                kwargs["height"] = int(row["height"])
            if "num_steps" in row and row["num_steps"]:
                kwargs["num_steps"] = int(row["num_steps"])
            if "seed" in row and row["seed"]:
                kwargs["seed"] = int(row["seed"])
            if "format" in row and row["format"]:
                kwargs["format"] = ImageFormat(row["format"].strip().lower())
            prompts.append(ImageRequest(**kwargs))
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f"Row {i + 1}: 'prompt' column is required"
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Row {i + 1}: {e}")
    return prompts


def _estimate_eta(progress: dict, worker) -> float | None:
    """Estimate remaining seconds based on average inference time."""
    stats = worker.get_stats()
    avg_time = stats.get("avg_inference_time_s", 0)
    if avg_time <= 0:
        return None
    remaining = progress.get("pending", 0)
    return round(remaining * avg_time, 1)


# Helper to get fresh session for SSE generator
from app.database import async_session_factory as _fresh_session  # noqa: E402
