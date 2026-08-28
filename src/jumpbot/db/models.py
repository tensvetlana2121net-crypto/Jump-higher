import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from jumpbot.db.base import Base, TimestampMixin


class Plan(StrEnum):
    FREE = "free"
    PRO = "pro"


class SubscriptionStatus(StrEnum):
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    EXPIRED = "expired"


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    username: Mapped[str | None] = mapped_column(String(64))
    first_name: Mapped[str | None] = mapped_column(String(128))
    language_code: Mapped[str] = mapped_column(String(10), default="ru", nullable=False)
    height_cm: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    weight_kg: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    consent_version: Mapped[str | None] = mapped_column(String(32))
    consented_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    subscriptions: Mapped[list["Subscription"]] = relationship(back_populates="user")
    jumps: Mapped[list["JumpHistory"]] = relationship(back_populates="user")


class Subscription(TimestampMixin, Base):
    __tablename__ = "subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    plan: Mapped[Plan] = mapped_column(Enum(Plan, name="subscription_plan"), nullable=False)
    status: Mapped[SubscriptionStatus] = mapped_column(
        Enum(SubscriptionStatus, name="subscription_status"), nullable=False
    )
    provider: Mapped[str | None] = mapped_column(String(32))
    provider_subscription_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    current_period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_at_period_end: Mapped[bool] = mapped_column(default=False, nullable=False)

    user: Mapped[User] = relationship(back_populates="subscriptions")


class JumpHistory(TimestampMixin, Base):
    __tablename__ = "jump_history"
    __table_args__ = (
        Index("ix_jump_history_user_created", "user_id", "created_at"),
        Index("ix_jump_history_status_created", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus, name="analysis_status"), default=AnalysisStatus.QUEUED
    )
    source_file_key: Mapped[str | None] = mapped_column(Text)
    source_file_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    annotated_file_key: Mapped[str | None] = mapped_column(Text)
    source_fps: Mapped[Decimal | None] = mapped_column(Numeric(8, 3))
    frame_count: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    jump_type: Mapped[str] = mapped_column(String(32), default="countermovement")
    published_to_app: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    calibration_method: Mapped[str | None] = mapped_column(String(32))
    start_frame: Mapped[int | None] = mapped_column(Integer)
    takeoff_frame: Mapped[int | None] = mapped_column(Integer)
    apex_frame: Mapped[int | None] = mapped_column(Integer)
    landing_frame: Mapped[int | None] = mapped_column(Integer)
    flight_time_ms: Mapped[int | None] = mapped_column(Integer)
    height_flight_cm: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    height_displacement_cm: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    takeoff_velocity_mps: Mapped[Decimal | None] = mapped_column(Numeric(7, 3))
    max_propulsion_mps: Mapped[Decimal | None] = mapped_column(Numeric(7, 3))
    max_angular_velocity_dps: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 3))
    quality_flags: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    phase_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metric_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="jumps")


class UsageCounter(Base):
    __tablename__ = "usage_counters"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    analyses_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    __table_args__ = (Index("uq_usage_user_period", "user_id", "period_start", unique=True),)
