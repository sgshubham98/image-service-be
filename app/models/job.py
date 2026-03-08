import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text, nullable=True)
    width = Column(Integer, nullable=False, default=1024)
    height = Column(Integer, nullable=False, default=1024)
    num_steps = Column(Integer, nullable=False, default=4)
    guidance_scale = Column(Float, nullable=False, default=0.0)
    seed = Column(Integer, nullable=True)
    status = Column(
        Enum("pending", "processing", "completed", "failed", "cancelled", name="job_status"),
        nullable=False,
        default="pending",
    )
    priority = Column(Integer, nullable=False, default=1)
    format = Column(
        Enum("png", "jpeg", "webp", name="image_format"),
        nullable=False,
        default="png",
    )
    file_path = Column(String(512), nullable=True)
    error_message = Column(Text, nullable=True)
    batch_id = Column(String(36), ForeignKey("batch_jobs.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    batch = relationship("BatchJob", back_populates="jobs")


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=True)
    total_count = Column(Integer, nullable=False, default=0)
    completed_count = Column(Integer, nullable=False, default=0)
    failed_count = Column(Integer, nullable=False, default=0)
    status = Column(
        Enum("pending", "processing", "completed", "failed", "cancelled", name="batch_status"),
        nullable=False,
        default="pending",
    )
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    jobs = relationship("Job", back_populates="batch", lazy="selectin")
