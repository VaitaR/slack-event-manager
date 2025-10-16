"""Initial schema for Slack Event Manager.

Revision ID: 001
Revises:
Create Date: 2025-10-16

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables for Slack Event Manager."""

    # Table: raw_slack_messages
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_slack_messages (
            message_id TEXT PRIMARY KEY,
            channel TEXT NOT NULL,
            ts TEXT NOT NULL,
            ts_dt TIMESTAMP WITH TIME ZONE NOT NULL,
            "user" TEXT,
            user_real_name TEXT,
            user_display_name TEXT,
            user_email TEXT,
            user_profile_image TEXT,
            is_bot BOOLEAN,
            subtype TEXT,
            text TEXT,
            blocks_text TEXT,
            text_norm TEXT,
            links_raw JSONB,
            links_norm JSONB,
            anchors JSONB,
            attachments_count INTEGER DEFAULT 0,
            files_count INTEGER DEFAULT 0,
            reactions JSONB,
            total_reactions INTEGER DEFAULT 0,
            reply_count INTEGER DEFAULT 0,
            permalink TEXT,
            edited_ts TEXT,
            edited_user TEXT,
            ingested_at TIMESTAMP WITH TIME ZONE
        )
    """
    )

    # Table: event_candidates
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS event_candidates (
            message_id TEXT PRIMARY KEY,
            channel TEXT NOT NULL,
            ts_dt TIMESTAMP WITH TIME ZONE NOT NULL,
            text_norm TEXT,
            links_norm JSONB,
            anchors JSONB,
            score REAL,
            status TEXT CHECK(status IN ('new', 'llm_ok', 'llm_fail')),
            features_json JSONB
        )
    """
    )

    # Table: events
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            version INTEGER DEFAULT 1,
            message_id TEXT NOT NULL,
            source_msg_event_idx INTEGER,
            dedup_key TEXT UNIQUE,
            event_date TIMESTAMP WITH TIME ZONE NOT NULL,
            event_end TIMESTAMP WITH TIME ZONE,
            category TEXT,
            title TEXT,
            summary TEXT,
            impact_area JSONB,
            tags JSONB,
            links JSONB,
            anchors JSONB,
            confidence REAL,
            source_channels JSONB,
            ingested_at TIMESTAMP WITH TIME ZONE
        )
    """
    )

    # Create indexes for events table
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_events_dedup_key
        ON events(dedup_key)
    """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_events_date
        ON events(event_date)
    """
    )

    # Table: llm_calls
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id SERIAL PRIMARY KEY,
            message_id TEXT,
            prompt_hash TEXT,
            model TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            cost_usd REAL,
            latency_ms INTEGER,
            cached BOOLEAN,
            response_json TEXT,
            ts TIMESTAMP WITH TIME ZONE
        )
    """
    )

    # Table: channel_watermarks
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS channel_watermarks (
            channel TEXT PRIMARY KEY,
            processing_ts TIMESTAMP WITH TIME ZONE,
            committed_ts TEXT
        )
    """
    )

    # Table: ingestion_state
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_state (
            channel_id TEXT PRIMARY KEY,
            last_ts REAL NOT NULL
        )
    """
    )


def downgrade() -> None:
    """Drop all tables."""
    op.execute("DROP TABLE IF EXISTS ingestion_state")
    op.execute("DROP TABLE IF EXISTS channel_watermarks")
    op.execute("DROP TABLE IF EXISTS llm_calls")
    op.execute("DROP INDEX IF EXISTS idx_events_date")
    op.execute("DROP INDEX IF EXISTS idx_events_dedup_key")
    op.execute("DROP TABLE IF EXISTS events")
    op.execute("DROP TABLE IF EXISTS event_candidates")
    op.execute("DROP TABLE IF EXISTS raw_slack_messages")
