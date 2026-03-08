"""Tests for generation API endpoints."""

import pytest


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
