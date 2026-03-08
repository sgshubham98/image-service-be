import enum


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, enum.Enum):
    REALTIME = 1
    BATCH = 5


class ImageFormat(str, enum.Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
