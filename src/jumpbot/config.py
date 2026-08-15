from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    secret_key: str = "development-only"
    database_url: str = "sqlite+aiosqlite:///./jumpbot.db"
    redis_url: str = "redis://localhost:6379/0"
    telegram_bot_token: str = ""
    storage_dir: Path = Path("storage")
    max_video_mb: int = Field(default=150, ge=1)
    max_video_seconds: int = Field(default=15, ge=1)
    free_analyses_per_week: int = Field(default=3, ge=0)
    keep_source_video_days: int = Field(default=7, ge=0)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
