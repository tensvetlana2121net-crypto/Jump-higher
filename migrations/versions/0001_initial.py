"""Initial schema."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    plan = postgresql.ENUM("FREE", "PRO", name="subscription_plan", create_type=False)
    subscription_status = postgresql.ENUM(
        "TRIALING", "ACTIVE", "PAST_DUE", "CANCELED", "EXPIRED",
        name="subscription_status", create_type=False
    )
    analysis_status = postgresql.ENUM(
        "QUEUED", "PROCESSING", "COMPLETED", "FAILED", "REJECTED",
        name="analysis_status", create_type=False
    )
    plan.create(op.get_bind())
    subscription_status.create(op.get_bind())
    analysis_status.create(op.get_bind())

    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(64)),
        sa.Column("first_name", sa.String(128)),
        sa.Column("language_code", sa.String(10), nullable=False),
        sa.Column("height_cm", sa.Numeric(5, 2)),
        sa.Column("weight_kg", sa.Numeric(5, 2)),
        sa.Column("consent_version", sa.String(32)),
        sa.Column("consented_at", sa.DateTime(timezone=True)),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("telegram_user_id"),
    )
    op.create_table(
        "subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("plan", plan, nullable=False),
        sa.Column("status", subscription_status, nullable=False),
        sa.Column("provider", sa.String(32)),
        sa.Column("provider_subscription_id", sa.String(255)),
        sa.Column("current_period_start", sa.DateTime(timezone=True)),
        sa.Column("current_period_end", sa.DateTime(timezone=True)),
        sa.Column("cancel_at_period_end", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_subscription_id"),
    )
    op.create_index("ix_subscriptions_user_id", "subscriptions", ["user_id"])
    op.create_table(
        "jump_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("status", analysis_status, nullable=False),
        sa.Column("source_file_key", sa.Text()),
        sa.Column("source_file_sha256", sa.String(64)),
        sa.Column("annotated_file_key", sa.Text()),
        sa.Column("source_fps", sa.Numeric(8, 3)),
        sa.Column("frame_count", sa.Integer()),
        sa.Column("duration_ms", sa.Integer()),
        sa.Column("jump_type", sa.String(32)),
        sa.Column("calibration_method", sa.String(32)),
        sa.Column("start_frame", sa.Integer()),
        sa.Column("takeoff_frame", sa.Integer()),
        sa.Column("apex_frame", sa.Integer()),
        sa.Column("landing_frame", sa.Integer()),
        sa.Column("flight_time_ms", sa.Integer()),
        sa.Column("height_flight_cm", sa.Numeric(6, 2)),
        sa.Column("height_displacement_cm", sa.Numeric(6, 2)),
        sa.Column("takeoff_velocity_mps", sa.Numeric(7, 3)),
        sa.Column("max_propulsion_mps", sa.Numeric(7, 3)),
        sa.Column("max_angular_velocity_dps", sa.Numeric(8, 2)),
        sa.Column("confidence_score", sa.Numeric(4, 3)),
        sa.Column("quality_flags", postgresql.JSONB(), nullable=False),
        sa.Column("phase_data", postgresql.JSONB()),
        sa.Column("metric_data", postgresql.JSONB()),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_message", sa.Text()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jump_history_user_created", "jump_history", ["user_id", "created_at"])
    op.create_index("ix_jump_history_status_created", "jump_history", ["status", "created_at"])
    op.create_index("ix_jump_history_source_file_sha256", "jump_history", ["source_file_sha256"])
    op.create_table(
        "usage_counters",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("analyses_count", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("uq_usage_user_period", "usage_counters", ["user_id", "period_start"], unique=True)


def downgrade() -> None:
    op.drop_table("usage_counters")
    op.drop_table("jump_history")
    op.drop_table("subscriptions")
    op.drop_table("users")
    sa.Enum(name="analysis_status").drop(op.get_bind())
    sa.Enum(name="subscription_status").drop(op.get_bind())
    sa.Enum(name="subscription_plan").drop(op.get_bind())
