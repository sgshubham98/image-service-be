"""Local filesystem storage management for generated images."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


def ensure_storage_dirs() -> None:
    """Create the output directory tree if it doesn't exist."""
    (settings.STORAGE_DIR / "images").mkdir(parents=True, exist_ok=True)
    (settings.STORAGE_DIR / "batches").mkdir(parents=True, exist_ok=True)


def get_image_path(job_id: str, fmt: str, batch_id: str | None = None) -> Path:
    """Return the storage path for a generated image.

    Layout:
        output/batches/{batch_id}/{job_id}.{ext}   (if part of a batch)
        output/images/{YYYY-MM-DD}/{job_id}.{ext}   (standalone)
    """
    ext = fmt.lower()
    if ext == "jpeg":
        ext = "jpg"

    if batch_id:
        directory = settings.STORAGE_DIR / "batches" / batch_id
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        directory = settings.STORAGE_DIR / "images" / date_str

    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{job_id}.{ext}"


def image_url_path(job_id: str) -> str:
    """Return the API URL path to serve this image."""
    return f"/jobs/{job_id}/image"
