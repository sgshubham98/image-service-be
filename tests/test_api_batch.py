"""Tests for batch API endpoints."""

import pytest


@pytest.mark.asyncio
async def test_create_batch(client, mock_worker):
    resp = await client.post(
        "/batch",
        json={
            "name": "test-batch",
            "prompts": [
                {"prompt": "a red car"},
                {"prompt": "a blue house"},
                {"prompt": "a green tree"},
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert data["status"] == "pending"
    assert data["batch_id"]
    # Worker should have been called 3 times
    assert mock_worker.enqueue.call_count == 3


@pytest.mark.asyncio
async def test_get_batch_404(client):
    resp = await client.get("/batch/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_batch(client, mock_worker):
    # Create batch first
    resp = await client.post(
        "/batch",
        json={
            "name": "cancel-test",
            "prompts": [{"prompt": "p1"}, {"prompt": "p2"}],
        },
    )
    batch_id = resp.json()["batch_id"]

    # Cancel it
    resp = await client.delete(f"/batch/{batch_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"


@pytest.mark.asyncio
async def test_batch_requires_prompts(client):
    resp = await client.post("/batch", json={"name": "empty", "prompts": []})
    assert resp.status_code == 422
