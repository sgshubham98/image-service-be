"""Tests for generation API endpoints."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.job import Job


@pytest.mark.asyncio
async def test_generate_async_returns_job_ids(client):
    resp = await client.post("/generate", json={"prompt": "a beautiful sunset"})
    assert resp.status_code == 200
    data = resp.json()
    assert "job_ids" in data
    assert len(data["job_ids"]) == 1
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_generate_multiple_images(client, mock_worker):
    resp = await client.post("/generate", json={"prompt": "test", "num_images": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["job_ids"]) == 3
    assert mock_worker.enqueue.call_count == 3


@pytest.mark.asyncio
async def test_generate_max_4_images(client):
    resp = await client.post("/generate", json={"prompt": "test", "num_images": 4})
    assert resp.status_code == 200
    assert len(resp.json()["job_ids"]) == 4


@pytest.mark.asyncio
async def test_generate_rejects_more_than_4_images(client):
    resp = await client.post("/generate", json={"prompt": "test", "num_images": 5})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_generate_async_enqueues_work(client, mock_worker):
    resp = await client.post("/generate", json={"prompt": "test prompt"})
    assert resp.status_code == 200
    mock_worker.enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_generate_validates_resolution_multiple_of_8(client):
    resp = await client.post(
        "/generate",
        json={"prompt": "test", "width": 513, "height": 1024},
    )
    assert resp.status_code == 422  # validation error


@pytest.mark.asyncio
async def test_generate_rejects_empty_prompt(client):
    resp = await client.post("/generate", json={"prompt": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_generate_rejects_oversized_resolution(client):
    resp = await client.post(
        "/generate",
        json={"prompt": "test", "width": 4096, "height": 4096},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_generate_default_values(client, mock_worker):
    resp = await client.post("/generate", json={"prompt": "a cat"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_serve_image_404_for_missing_job(client):
    resp = await client.get("/jobs/nonexistent-id/image")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_generate_stream_emits_done_for_completed_jobs(client, db_engine):
    create = await client.post("/generate", json={"prompt": "test", "num_images": 2})
    assert create.status_code == 200
    job_ids = create.json()["job_ids"]

    # Mark jobs as completed so the SSE stream terminates immediately.
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        result = await session.execute(select(Job).where(Job.id.in_(job_ids)))
        jobs = result.scalars().all()
        for job in jobs:
            job.status = "completed"
            job.file_path = f"output/images/{job.id}.png"
        await session.commit()

    resp = await client.get(
        "/generate/stream",
        params=[("job_id", job_ids[0]), ("job_id", job_ids[1])],
    )
    assert resp.status_code == 200
    text = resp.text
    assert "event: progress" in text
    assert "event: done" in text
    assert '"completed": 2' in text


@pytest.mark.asyncio
async def test_generate_stream_returns_404_for_unknown_job(client):
    resp = await client.get("/generate/stream", params={"job_id": "unknown-job"})
    assert resp.status_code == 404
