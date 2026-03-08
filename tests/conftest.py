"""Shared test fixtures."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.engine.flux_engine import GenerationResult
from app.models.job import Base


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_engine(tmp_path):
    """In-memory SQLite for tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest.fixture
def mock_engine():
    """A mock FluxEngine that returns a tiny test image."""
    engine = MagicMock()
    engine.is_loaded = True
    engine.get_vram_stats.return_value = {"available": True, "allocated_gb": 12.0}
    engine.get_vram_usage_gb.return_value = 12.0

    dummy_image = Image.new("RGB", (64, 64), color="red")
    engine.generate.return_value = GenerationResult(
        image=dummy_image, seed=42, inference_time=0.5
    )
    return engine


@pytest.fixture
def mock_worker(mock_engine):
    """A mock GPUWorker that tracks enqueued job IDs."""
    worker = MagicMock()
    worker.queue_depth = 0
    worker.enqueue = AsyncMock()
    worker.get_stats.return_value = {
        "queue_depth": 0,
        "jobs_processed": 0,
        "total_inference_time_s": 0.0,
        "avg_inference_time_s": 0.0,
    }
    return worker


@pytest.fixture
async def client(mock_engine, mock_worker, db_engine, tmp_path):
    """Test client with mocked engine/worker and in-memory DB."""
    from app.database import get_session
    from app.models.job import Base

    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_session():
        async with factory() as session:
            yield session

    # Patch the database session factory used by the worker
    with patch("app.worker.gpu_worker.async_session_factory", factory), \
         patch("app.database.async_session_factory", factory):

        from app.main import app

        app.dependency_overrides[get_session] = override_get_session
        app.state.engine = mock_engine
        app.state.worker = mock_worker

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

        app.dependency_overrides.clear()
