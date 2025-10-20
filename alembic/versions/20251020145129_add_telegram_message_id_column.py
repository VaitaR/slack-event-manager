"""Add last_processed_message_id to ingestion_state_telegram table

Revision ID: 20251020145129
Revises: 4bfe4eac39b1
Create Date: 2025-10-20 14:51:29.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "20251020145129"
down_revision = "4bfe4eac39b1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add last_processed_message_id column to ingestion_state_telegram table."""
    op.add_column(
        "ingestion_state_telegram",
        sa.Column("last_processed_message_id", sa.String(length=50), nullable=True),
    )


def downgrade() -> None:
    """Remove last_processed_message_id column from ingestion_state_telegram table."""
    op.drop_column("ingestion_state_telegram", "last_processed_message_id")
