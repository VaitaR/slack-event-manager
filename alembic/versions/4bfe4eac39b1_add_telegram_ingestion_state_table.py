"""Add ingestion_state_telegram table

Revision ID: 4bfe4eac39b1
Revises: d00fe85d0b43
Create Date: 2025-10-19T21:10:47.787489

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "4bfe4eac39b1"
down_revision = "d00fe85d0b43"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ingestion_state_telegram table."""
    op.create_table(
        "ingestion_state_telegram",
        sa.Column("channel_id", sa.String(length=100), nullable=False),
        sa.Column("last_processed_ts", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("channel_id"),
    )


def downgrade() -> None:
    """Drop ingestion_state_telegram table."""
    op.drop_table("ingestion_state_telegram")
