"""Gate bot results before publishing them to the Mini App."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_publish_results_to_app"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "jump_history",
        sa.Column(
            "published_to_app",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    op.drop_column("jump_history", "published_to_app")
