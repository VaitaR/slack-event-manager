"""Rename last_processed_ts to max_processed_ts in ingestion_state_telegram

Revision ID: rename_telegram_ts
Revises:
Create Date: 2025-10-31 11:40:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "rename_telegram_ts"
down_revision = "202510251200"


def upgrade() -> None:
    """Rename last_processed_ts to max_processed_ts in ingestion_state_telegram."""
    op.execute("""
        ALTER TABLE ingestion_state_telegram
        RENAME COLUMN last_processed_ts TO max_processed_ts;
    """)


def downgrade() -> None:
    """Revert column name back to last_processed_ts."""
    op.execute("""
        ALTER TABLE ingestion_state_telegram
        RENAME COLUMN max_processed_ts TO last_processed_ts;
    """)
