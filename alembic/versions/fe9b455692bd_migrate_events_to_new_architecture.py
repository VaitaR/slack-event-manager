"""migrate_events_to_new_architecture

Revision ID: fe9b455692bd
Revises: 001_initial_schema
Create Date: 2025-10-18

Migrates events table from old structure (title, version, event_date, event_end)
to new architecture (action, object_name_raw, status, time fields, importance).

This is a breaking change that requires data migration or table recreation.
For production, backup data before running this migration.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fe9b455692bd"
down_revision: str | None = "001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Migrate events table to new architecture.

    WARNING: This drops and recreates the events table.
    All existing event data will be lost.
    For production, implement data migration logic.
    """
    # Drop old events table
    op.drop_table("events")

    # Create new events table with comprehensive structure
    op.create_table(
        "events",
        # Identification
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("source_channels", postgresql.JSONB(), nullable=False),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=False),
        # Title Slots (source of truth)
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("object_id", sa.String(), nullable=True),
        sa.Column("object_name_raw", sa.String(), nullable=False),
        sa.Column("qualifiers", postgresql.JSONB(), nullable=True),
        sa.Column("stroke", sa.String(), nullable=True),
        sa.Column("anchor", sa.String(), nullable=True),
        # Classification & Lifecycle
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("change_type", sa.String(), nullable=False),
        sa.Column("environment", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=True),
        # Time Fields
        sa.Column("planned_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("planned_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("time_source", sa.String(), nullable=False),
        sa.Column("time_confidence", sa.Float(), nullable=False),
        # Content & Links
        sa.Column("summary", sa.String(), nullable=False),
        sa.Column("why_it_matters", sa.String(), nullable=True),
        sa.Column("links", postgresql.JSONB(), nullable=True),
        sa.Column("anchors", postgresql.JSONB(), nullable=True),
        sa.Column("impact_area", postgresql.JSONB(), nullable=True),
        sa.Column("impact_type", postgresql.JSONB(), nullable=True),
        # Quality & Importance
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("importance", sa.Integer(), nullable=False),
        # Clustering
        sa.Column("cluster_key", sa.String(), nullable=False),
        sa.Column("dedup_key", sa.String(), nullable=False),
        # Constraints
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("dedup_key"),
    )

    # Create indexes for common queries
    op.create_index(
        "idx_events_dedup_key",
        "events",
        ["dedup_key"],
        unique=True,
    )
    op.create_index(
        "idx_events_time_fields",
        "events",
        [sa.text("COALESCE(actual_start, actual_end, planned_start, planned_end)")],
    )
    op.create_index(
        "idx_events_cluster_key",
        "events",
        ["cluster_key"],
    )


def downgrade() -> None:
    """Revert to old events table structure.

    WARNING: This drops and recreates the events table.
    All existing event data will be lost.
    """
    # Drop new events table
    op.drop_index("idx_events_cluster_key", table_name="events")
    op.drop_index("idx_events_time_fields", table_name="events")
    op.drop_index("idx_events_dedup_key", table_name="events")
    op.drop_table("events")

    # Recreate old events table structure
    op.create_table(
        "events",
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("source_msg_event_idx", sa.Integer(), nullable=False),
        sa.Column("dedup_key", sa.String(), nullable=False),
        sa.Column("event_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("event_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=True),
        sa.Column("impact_area", postgresql.JSONB(), nullable=True),
        sa.Column("tags", postgresql.JSONB(), nullable=True),
        sa.Column("links", postgresql.JSONB(), nullable=True),
        sa.Column("anchors", postgresql.JSONB(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source_channels", postgresql.JSONB(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("dedup_key"),
    )

    # Recreate old indexes
    op.create_index("idx_events_dedup_key", "events", ["dedup_key"], unique=True)
    op.create_index("idx_events_event_date", "events", ["event_date"])
