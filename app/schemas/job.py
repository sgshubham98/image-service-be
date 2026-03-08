from pydantic import BaseModel


class JobDetail(BaseModel):
    id: str
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    num_steps: int
    guidance_scale: float
    seed: int | None
    status: str
    priority: int
    format: str
    file_path: str | None
    image_url: str | None
    error_message: str | None
    batch_id: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None


class JobList(BaseModel):
    jobs: list[JobDetail]
    total: int
    page: int
    page_size: int
