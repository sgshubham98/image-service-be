"""Resolution presets and engine configuration."""

PRESET_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "square_sm": (512, 512),
    "square": (1024, 1024),
    "landscape": (1344, 768),
    "portrait": (768, 1344),
    "wide": (1536, 640),
}

# Approximate peak VRAM (GB) by max dimension for FP8 Flux Schnell.
# Used as a safety guard — not an exact model; conservative estimates.
VRAM_ESTIMATES: dict[int, float] = {
    512: 13.5,
    768: 14.5,
    1024: 16.0,
    1536: 18.5,
    2048: 20.0,
}


def estimated_vram_gb(width: int, height: int) -> float:
    """Return conservative VRAM estimate for a given resolution."""
    max_dim = max(width, height)
    for threshold in sorted(VRAM_ESTIMATES.keys()):
        if max_dim <= threshold:
            return VRAM_ESTIMATES[threshold]
    return VRAM_ESTIMATES[max(VRAM_ESTIMATES.keys())]
