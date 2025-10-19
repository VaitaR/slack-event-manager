"""Add raw_telegram_messages table

Revision ID: d00fe85d0b43
Revises: fe9b455692bd
Create Date: 2025-10-19T21:10:42.391593

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "d00fe85d0b43"
down_revision = "fe9b455692bd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create raw_telegram_messages table."""
    op.create_table(
        "raw_telegram_messages",
        sa.Column("message_id", sa.String(length=100), nullable=False),
        sa.Column("channel", sa.String(length=100), nullable=False),
        sa.Column("message_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sender_id", sa.String(length=100), nullable=True),
        sa.Column("sender_name", sa.String(length=200), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("text_norm", sa.Text(), nullable=True),
        sa.Column("forward_from_channel", sa.String(length=100), nullable=True),
        sa.Column("forward_from_message_id", sa.String(length=100), nullable=True),
        sa.Column("media_type", sa.String(length=50), nullable=True),
        sa.Column("links_raw", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("links_norm", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("anchors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("views", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reply_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reactions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("post_url", sa.Text(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("message_id"),
    )


def downgrade() -> None:
    """Drop raw_telegram_messages table."""
    op.drop_table("raw_telegram_messages")
