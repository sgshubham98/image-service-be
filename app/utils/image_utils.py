"""Image saving, format conversion, and metadata embedding."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, PngImagePlugin

from app.models.enums import ImageFormat


def save_image(
    image: Image.Image,
    path: Path,
    fmt: ImageFormat,
    quality: int = 95,
    prompt: str | None = None,
    seed: int | None = None,
) -> None:
    """Save a PIL Image in the requested format with embedded metadata."""
    if fmt == ImageFormat.PNG:
        meta = PngImagePlugin.PngInfo()
        if prompt:
            meta.add_text("prompt", prompt)
        if seed is not None:
            meta.add_text("seed", str(seed))
        image.save(path, format="PNG", pnginfo=meta)

    elif fmt == ImageFormat.JPEG:
        # JPEG doesn't support alpha
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        image.save(path, format="JPEG", quality=quality)

    elif fmt == ImageFormat.WEBP:
        image.save(path, format="WEBP", quality=quality)

    else:
        image.save(path, format="PNG")
