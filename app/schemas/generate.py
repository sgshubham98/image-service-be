from pydantic import BaseModel, Field, model_validator

from app.config import settings
from app.models.enums import ImageFormat


class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str | None = None
    width: int = Field(default=1024, ge=settings.MIN_RESOLUTION, le=settings.MAX_RESOLUTION)
    height: int = Field(default=1024, ge=settings.MIN_RESOLUTION, le=settings.MAX_RESOLUTION)
    num_steps: int = Field(default=settings.DEFAULT_NUM_STEPS, ge=1, le=50)
    guidance_scale: float = Field(default=settings.DEFAULT_GUIDANCE_SCALE, ge=0.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    num_images: int = Field(default=1, ge=1, le=settings.MAX_IMMEDIATE_IMAGES)
    format: ImageFormat = ImageFormat.PNG

    @model_validator(mode="after")
    def validate_resolution(self) -> "ImageRequest":
        if self.width % 8 != 0:
            raise ValueError("width must be a multiple of 8")
        if self.height % 8 != 0:
            raise ValueError("height must be a multiple of 8")
        return self


class ImageResponse(BaseModel):
    job_ids: list[str]
    status: str
    image_urls: list[str | None] = []


PRESET_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "square_sm": (512, 512),
    "square": (1024, 1024),
    "landscape": (1344, 768),
    "portrait": (768, 1344),
    "wide": (1536, 640),
}
