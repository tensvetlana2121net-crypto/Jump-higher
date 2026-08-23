from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    secret_key: str = "development-only"
    api_key: str = "development-only-api-key"
    database_url: str = "sqlite+aiosqlite:///./jumpbot.db"
    redis_url: str = "redis://localhost:6379/0"
    telegram_bot_token: str = ""
    telegram_init_data_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    miniapp_static_dir: Path = Path(__file__).parent / "miniapp_static"
    storage_dir: Path = Path("storage")
    max_video_mb: int = Field(default=20, ge=1, le=20)
    max_video_seconds: int = Field(default=15, ge=1)
    free_quota_enabled: bool = False
    free_analyses_per_week: int = Field(default=3, ge=0)
    keep_source_video_days: int = Field(default=0, ge=0)
    log_level: str = "INFO"
    pose_backend: str = "rtmpose"
    pose_tracking_roi_enabled: bool = True
    pose_camera_stabilization_enabled: bool = True

    @model_validator(mode="after")
    def validate_pose_backend(self) -> "Settings":
        if self.pose_backend.lower() not in {"rtmpose", "mediapipe"}:
            raise ValueError("POSE_BACKEND must be 'rtmpose' or 'mediapipe'")
        return self

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        if self.app_env.lower() == "production":
            weak = {"", "change-me", "development-only", "development-only-api-key"}
            if self.secret_key in weak or len(self.secret_key) < 32:
                raise ValueError("SECRET_KEY must be at least 32 characters in production")
            if self.api_key in weak or len(self.api_key) < 32:
                raise ValueError("API_KEY must be at least 32 characters in production")
            if not self.telegram_bot_token:
                raise ValueError("TELEGRAM_BOT_TOKEN is required in production")
            if not self.database_url.startswith("postgresql+asyncpg://"):
                raise ValueError("Production must use PostgreSQL")
            if self.keep_source_video_days != 0:
                raise ValueError(
                    "Production source-video retention must be 0 until cleanup is implemented"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
