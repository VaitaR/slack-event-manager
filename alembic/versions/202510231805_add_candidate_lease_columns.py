"""Add lease tracking columns to event_candidates."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "202510231805"
down_revision = "b4e8c6d0f1d7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add processing timestamps and lease attempts."""

    op.add_column(
        "event_candidates",
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "event_candidates",
        sa.Column(
            "lease_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.execute(
        "UPDATE event_candidates SET lease_attempts = 0 WHERE lease_attempts IS NULL"
    )


def downgrade() -> None:
    """Remove lease tracking columns."""

    op.drop_column("event_candidates", "lease_attempts")
    op.drop_column("event_candidates", "processing_started_at")
