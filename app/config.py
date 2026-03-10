from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ── Model ──
    MODEL_ID: str = "black-forest-labs/FLUX.1-schnell"
    DEVICE: str = "cuda"
    DTYPE: str = "float8_e4m3fn"
    ENABLE_TORCH_COMPILE: bool = True

    # ── Generation defaults ──
    DEFAULT_NUM_STEPS: int = 4
    DEFAULT_GUIDANCE_SCALE: float = 0.0
    MAX_RESOLUTION: int = 2048
    MIN_RESOLUTION: int = 64

    # ── Storage ──
    STORAGE_DIR: Path = Path("./output")

    # ── Logging ──
    LOG_DIR: Path = Path("./logs")
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION_WHEN: str = "midnight"  # midnight, h, m, s, etc.
    LOG_RETENTION_COUNT: int = 30  # number of rotated files to keep

    # ── Queue ──
    MAX_QUEUE_SIZE: int = 10000

    # ── Server ──
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Immediate generation ──
    MAX_IMMEDIATE_IMAGES: int = 4

    # ── Sync generation timeout (seconds) ──
    SYNC_TIMEOUT: float = 60.0

    # ── Content moderation ──
    MODERATION_ENABLED: bool = False
    MODERATION_API_KEY: str = ""
    MODERATION_API_URL: str = "https://api.openai.com/v1/moderations"
    MODERATION_TIMEOUT: float = 10.0


settings = Settings()
