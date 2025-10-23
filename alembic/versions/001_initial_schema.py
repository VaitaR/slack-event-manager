"""Initial PostgreSQL schema

Revision ID: 001
Revises:
Create Date: 2025-10-16 22:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial schema for Slack Event Manager."""

    # 1. raw_slack_messages table (matching SQLite schema)
    op.create_table(
        "raw_slack_messages",
        sa.Column("message_id", sa.String(length=50), nullable=False),
        sa.Column("channel_id", sa.String(length=50), nullable=False),
        sa.Column("ts", sa.String(length=50), nullable=False),
        sa.Column("ts_dt", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user", sa.String(length=50), nullable=True),  # Quoted in queries
        sa.Column("user_real_name", sa.String(length=200), nullable=True),
        sa.Column("user_display_name", sa.String(length=200), nullable=True),
        sa.Column("user_email", sa.String(length=200), nullable=True),
        sa.Column("user_profile_image", sa.Text(), nullable=True),
        sa.Column("is_bot", sa.Boolean(), nullable=True),
        sa.Column("subtype", sa.String(length=50), nullable=True),
        sa.Column("text_raw", sa.Text(), nullable=True),
        sa.Column("blocks_text", sa.Text(), nullable=True),
        sa.Column("text_norm", sa.Text(), nullable=True),
        sa.Column("links_raw", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("links_norm", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("anchors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "attachments_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("files_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reactions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("total_reactions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reply_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("permalink", sa.Text(), nullable=True),
        sa.Column("edited_ts", sa.String(length=50), nullable=True),
        sa.Column("edited_user", sa.String(length=50), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("message_id"),
    )
    op.create_index(
        "idx_raw_messages_channel_ts", "raw_slack_messages", ["channel_id", "ts"]
    )

    # 2. event_candidates table (matching SQLite schema)
    op.create_table(
        "event_candidates",
        sa.Column("message_id", sa.String(length=50), nullable=False),
        sa.Column("channel", sa.String(length=50), nullable=False),
        sa.Column("ts_dt", sa.DateTime(timezone=True), nullable=False),
        sa.Column("text_norm", sa.Text(), nullable=True),
        sa.Column("links_norm", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("anchors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column(
            "features_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.CheckConstraint(
            "status IN ('new', 'processing', 'llm_ok', 'llm_fail')",
            name="status_check",
        ),
        sa.PrimaryKeyConstraint("message_id"),
    )

    # 3. events table (matching SQLite schema)
    op.create_table(
        "events",
        sa.Column("event_id", sa.String(length=100), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("message_id", sa.String(length=50), nullable=False),
        sa.Column("source_msg_event_idx", sa.Integer(), nullable=True),
        sa.Column("dedup_key", sa.String(length=64), nullable=True),
        sa.Column("event_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("category", sa.String(length=50), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "impact_area", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("links", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("anchors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "source_channels", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("dedup_key"),
    )
    op.create_index("idx_events_dedup_key", "events", ["dedup_key"])
    op.create_index("idx_events_date", "events", ["event_date"])

    # 4. llm_calls table
    op.create_table(
        "llm_calls",
        sa.Column("call_id", sa.String(length=100), nullable=False),
        sa.Column("message_id", sa.String(length=50), nullable=True),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_msg", sa.Text(), nullable=True),
        sa.Column(
            "response_cached", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("call_id"),
    )

    # 5. channel_watermarks table
    op.create_table(
        "channel_watermarks",
        sa.Column("channel_id", sa.String(length=50), nullable=False),
        sa.Column("ts", sa.String(length=50), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("channel_id"),
    )

    # 6. ingestion_state table
    op.create_table(
        "ingestion_state",
        sa.Column("channel_id", sa.String(length=50), nullable=False),
        sa.Column("last_processed_ts", sa.String(length=50), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("channel_id"),
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("ingestion_state")
    op.drop_table("channel_watermarks")
    op.drop_table("llm_calls")
    op.drop_index("idx_events_date", table_name="events")
    op.drop_index("idx_events_dedup_key", table_name="events")
    op.drop_table("events")
    op.drop_table("event_candidates")
    op.drop_index("idx_raw_messages_channel_ts", table_name="raw_slack_messages")
    op.drop_table("raw_slack_messages")
