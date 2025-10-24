"""Add source tracking and LLM cache columns."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "202510251200"
down_revision = "202510231805"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add source_id to events and response_json to llm_calls."""

    op.add_column(
        "events",
        sa.Column("source_id", sa.String(), nullable=True, server_default="slack"),
    )
    op.execute("UPDATE events SET source_id = COALESCE(source_id, 'slack')")
    op.alter_column("events", "source_id", nullable=False, server_default=None)

    op.add_column(
        "llm_calls",
        sa.Column("response_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Remove source and cache columns."""

    op.drop_column("llm_calls", "response_json")
    op.drop_column("events", "source_id")
