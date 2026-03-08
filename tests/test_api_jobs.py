"""Tests for job management endpoints."""

import pytest


@pytest.mark.asyncio
async def test_list_jobs_empty(client):
    resp = await client.get("/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["jobs"] == []


@pytest.mark.asyncio
async def test_get_job_after_create(client, mock_worker):
    # Create a job via generate
    gen_resp = await client.post("/generate", json={"prompt": "test job"})
    job_id = gen_resp.json()["job_ids"][0]

    # Fetch it
    resp = await client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == job_id
    assert data["prompt"] == "test job"
    assert data["status"] == "pending"
    assert data["width"] == 1024
    assert data["height"] == 1024


@pytest.mark.asyncio
async def test_cancel_job(client, mock_worker):
    gen_resp = await client.post("/generate", json={"prompt": "to cancel"})
    job_id = gen_resp.json()["job_ids"][0]

    resp = await client.delete(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_list_jobs_with_status_filter(client, mock_worker):
    await client.post("/generate", json={"prompt": "j1"})
    await client.post("/generate", json={"prompt": "j2"})

    resp = await client.get("/jobs?status=pending")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2


@pytest.mark.asyncio
async def test_get_nonexistent_job(client):
    resp = await client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "vram" in data
    assert "worker" in data
