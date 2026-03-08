"""Tests for the GPU worker queue and processing logic."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import pytest
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.engine.flux_engine import GenerationResult
from app.models.job import Base, Job
from app.worker.gpu_worker import GPUWorker, _QueueItem


class TestQueuePriority:
    def test_realtime_before_batch(self):
        """REALTIME (priority=1) items should sort before BATCH (priority=5)."""
        rt = _QueueItem(priority=1, timestamp=100.0, job_id="rt")
        batch = _QueueItem(priority=5, timestamp=99.0, job_id="batch")
        assert rt < batch

    def test_same_priority_fifo(self):
        """Same priority: earlier timestamp wins."""
        a = _QueueItem(priority=1, timestamp=100.0, job_id="a")
        b = _QueueItem(priority=1, timestamp=200.0, job_id="b")
        assert a < b


class TestWorkerEnqueue:
    @pytest.mark.asyncio
    async def test_enqueue_increases_depth(self):
        engine = MagicMock()
        worker = GPUWorker(engine=engine)
        await worker.enqueue("job-1", priority=1)
        await worker.enqueue("job-2", priority=5)
        assert worker.queue_depth == 2


class TestWorkerStats:
    def test_initial_stats(self):
        engine = MagicMock()
        worker = GPUWorker(engine=engine)
        stats = worker.get_stats()
        assert stats["jobs_processed"] == 0
        assert stats["queue_depth"] == 0
        assert stats["avg_inference_time_s"] == 0.0
