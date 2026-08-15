import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from jumpbot.db.models import AnalysisStatus


class UserCreate(BaseModel):
    telegram_user_id: int
    username: str | None = None
    first_name: str | None = None
    height_cm: Decimal | None = Field(default=None, ge=100, le=250)


class UserRead(UserCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime


class JumpRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    status: AnalysisStatus
    height_flight_cm: Decimal | None
    height_displacement_cm: Decimal | None
    takeoff_velocity_mps: Decimal | None
    confidence_score: Decimal | None
    quality_flags: list[str]
    created_at: datetime


class HealthRead(BaseModel):
    status: str
    version: str
