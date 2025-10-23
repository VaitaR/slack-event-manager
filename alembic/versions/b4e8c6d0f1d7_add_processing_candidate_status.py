"""Allow processing status for event_candidates."""

from __future__ import annotations

from alembic import op

# Revision identifiers, used by Alembic.
revision = "b4e8c6d0f1d7"
down_revision = "20251020145129"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Expand candidate status check constraint."""

    op.drop_constraint("status_check", "event_candidates", type_="check")
    op.create_check_constraint(
        "status_check",
        "event_candidates",
        "status IN ('new', 'processing', 'llm_ok', 'llm_fail')",
    )


def downgrade() -> None:
    """Restore original candidate status constraint."""

    op.drop_constraint("status_check", "event_candidates", type_="check")
    op.create_check_constraint(
        "status_check",
        "event_candidates",
        "status IN ('new', 'llm_ok', 'llm_fail')",
    )
