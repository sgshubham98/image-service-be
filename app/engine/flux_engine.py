"""Flux.1 Schnell inference engine — model loading, generation, VRAM management."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# Dtype mapping
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float32": torch.float32,
}


@dataclass
class GenerationResult:
    image: Image.Image
    seed: int
    inference_time: float  # seconds


@dataclass
class FluxEngine:
    """Singleton engine wrapping the Flux.1 Schnell diffusion pipeline."""

    _pipeline: object | None = field(default=None, repr=False)
    _loaded: bool = False
    _device: str = "cuda"
    _dtype: torch.dtype = torch.bfloat16

    def load_model(self) -> None:
        """Load the Flux pipeline onto the GPU."""
        if self._loaded:
            logger.info("Model already loaded, skipping.")
            return

        from diffusers import FluxPipeline

        self._device = settings.DEVICE
        self._dtype = _DTYPE_MAP.get(settings.DTYPE, torch.bfloat16)

        logger.info(
            "Loading Flux.1 Schnell model=%s device=%s dtype=%s",
            settings.MODEL_ID,
            self._device,
            self._dtype,
        )

        load_kwargs: dict = {
            "torch_dtype": self._dtype,
        }

        # For FP8 on H100, use the fp8 variant if available
        if self._dtype == torch.float8_e4m3fn:
            load_kwargs["variant"] = "fp8"

        self._pipeline = FluxPipeline.from_pretrained(
            settings.MODEL_ID,
            **load_kwargs,
        )
        self._pipeline.to(self._device)
        self._pipeline.set_progress_bar_config(disable=True)

        # torch.compile for H100 Tensor Core acceleration
        if settings.ENABLE_TORCH_COMPILE and self._device == "cuda":
            logger.info("Compiling transformer with torch.compile (mode=max-autotune-no-cudagraphs)...")
            self._pipeline.transformer = torch.compile(
                self._pipeline.transformer,
                mode="max-autotune-no-cudagraphs",
            )

        self._loaded = True
        logger.info("Model loaded. VRAM: %.2f GB", self.get_vram_usage_gb())

    def warmup(self) -> None:
        """Run a single throwaway inference to trigger torch.compile tracing."""
        if not self._loaded:
            raise RuntimeError("Model not loaded — call load_model() first")

        logger.info("Running warm-up inference (first run compiles CUDA kernels)...")
        start = time.perf_counter()
        self.generate(prompt="warmup", width=512, height=512, num_steps=1, seed=0)
        elapsed = time.perf_counter() - start
        logger.info("Warm-up completed in %.1fs", elapsed)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate a single image. Runs synchronously on the GPU."""
        if not self._loaded:
            raise RuntimeError("Model not loaded — call load_model() first")

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        generator = torch.Generator(device=self._device).manual_seed(seed)

        start = time.perf_counter()

        output = self._pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

        elapsed = time.perf_counter() - start
        image: Image.Image = output.images[0]

        logger.info(
            "Generated %dx%d in %.2fs (steps=%d, seed=%d)",
            width,
            height,
            elapsed,
            num_steps,
            seed,
        )

        return GenerationResult(image=image, seed=seed, inference_time=elapsed)

    def get_vram_usage_gb(self) -> float:
        """Return current GPU memory allocated in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(self._device) / (1024**3)

    def get_vram_stats(self) -> dict:
        """Return detailed VRAM statistics."""
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "allocated_gb": round(torch.cuda.memory_allocated(self._device) / (1024**3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved(self._device) / (1024**3), 2),
            "max_allocated_gb": round(
                torch.cuda.max_memory_allocated(self._device) / (1024**3), 2
            ),
            "total_gb": round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2),
            "device_name": torch.cuda.get_device_name(0),
        }

    def unload(self) -> None:
        """Free the model and clear CUDA cache."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        logger.info("Model unloaded, CUDA cache cleared.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded
