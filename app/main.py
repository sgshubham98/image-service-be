"""FastAPI application — lifespan manages model loading, worker, and DB."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import batch, generate, health, jobs
from app.config import settings
from app.database import init_db
from app.engine.flux_engine import FluxEngine
from app.utils.moderation import get_moderation_engine
from app.utils.storage import ensure_storage_dirs
from app.worker.gpu_worker import GPUWorker


def _setup_logging() -> None:
    """Configure logging to both console and a time-rotating file."""
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(log_format))
    root.addHandler(console)

    # Time-rotating file handler
    file_handler = TimedRotatingFileHandler(
        filename=settings.LOG_DIR / "flux-service.log",
        when=settings.LOG_ROTATION_WHEN,
        backupCount=settings.LOG_RETENTION_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.suffix = "%Y-%m-%d"  # rotated files: flux-service.log.2026-03-08
    root.addHandler(file_handler)


_setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────
    logger.info("Starting Flux Image Service...")

    # Storage directories
    ensure_storage_dirs()

    # Database
    await init_db()
    logger.info("Database initialized.")

    # Flux engine
    engine = FluxEngine()
    engine.load_model()
    engine.warmup()
    app.state.engine = engine

    # Content moderation engine
    moderation_engine = get_moderation_engine()
    moderation_engine.load(settings.MODERATION_MODEL_ID)
    app.state.moderation_engine = moderation_engine

    # GPU worker
    worker = GPUWorker(engine=engine)
    worker.start()
    resumed = await worker.resume_pending_jobs()
    if resumed:
        logger.info("Resumed %d pending jobs from previous session.", resumed)
    app.state.worker = worker

    # Start time for health endpoint
    health.set_start_time()

    logger.info("Flux Image Service ready. Queue depth: %d", worker.queue_depth)

    yield

    # ── Shutdown ─────────────────────────────────────────────────
    logger.info("Shutting down Flux Image Service...")
    await worker.shutdown()
    engine.unload()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Flux Image Service",
    description="Local image generation API powered by Flux.1 Schnell 12B",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(generate.router)
app.include_router(batch.router)
app.include_router(jobs.router)
app.include_router(health.router)
