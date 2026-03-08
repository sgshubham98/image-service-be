from pydantic import BaseModel, Field

from app.schemas.generate import ImageRequest


class BatchRequest(BaseModel):
    name: str | None = None
    prompts: list[ImageRequest] = Field(..., min_length=1, max_length=50000)


class BatchProgress(BaseModel):
    batch_id: str
    name: str | None
    total: int
    completed: int
    failed: int
    cancelled: int
    pending: int
    status: str
    estimated_remaining_seconds: float | None = None
    created_at: str
    completed_at: str | None = None
