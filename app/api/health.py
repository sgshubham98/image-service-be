"""Health and status endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])

_start_time: float = 0.0


def set_start_time() -> None:
    global _start_time
    _start_time = time.time()


@router.get("/health")
async def health_check(request: Request):
    engine = request.app.state.engine
    worker = request.app.state.worker

    uptime = time.time() - _start_time if _start_time else 0

    return {
        "status": "ok",
        "model_loaded": engine.is_loaded,
        "vram": engine.get_vram_stats(),
        "worker": worker.get_stats(),
        "uptime_seconds": round(uptime, 1),
    }
