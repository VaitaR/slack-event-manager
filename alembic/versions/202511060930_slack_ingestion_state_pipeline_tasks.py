"""Normalize Slack ingestion state and add pipeline tasks queue."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "202511060930"
down_revision = "rename_telegram_ts"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    """Check if a table exists."""

    return table_name in inspector.get_table_names()


def _get_columns(
    inspector: sa.Inspector, table_name: str
) -> dict[str, dict[str, object]]:
    """Return mapped column metadata keyed by column name."""

    return {column["name"]: column for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Rename Slack ingestion state, add resume fields, and create pipeline tasks."""

    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "slack_ingestion_state") and _has_table(
        inspector, "ingestion_state"
    ):
        op.rename_table("ingestion_state", "slack_ingestion_state")
        inspector = sa.inspect(bind)

    if _has_table(inspector, "slack_ingestion_state"):
        columns = _get_columns(inspector, "slack_ingestion_state")

        if "last_processed_ts" in columns:
            op.alter_column(
                "slack_ingestion_state",
                "last_processed_ts",
                new_column_name="max_processed_ts",
                existing_type=columns["last_processed_ts"]["type"],
                existing_nullable=False,
            )
            inspector = sa.inspect(bind)
            columns = _get_columns(inspector, "slack_ingestion_state")

        if "max_processed_ts" not in columns:
            op.add_column(
                "slack_ingestion_state",
                sa.Column("max_processed_ts", sa.String(length=50), nullable=True),
            )
            op.execute(
                sa.text(
                    "UPDATE slack_ingestion_state SET max_processed_ts = '0' "
                    "WHERE max_processed_ts IS NULL"
                )
            )
            op.alter_column(
                "slack_ingestion_state",
                "max_processed_ts",
                existing_type=sa.String(length=50),
                nullable=False,
            )
            columns = _get_columns(inspector, "slack_ingestion_state")
        else:
            op.alter_column(
                "slack_ingestion_state",
                "max_processed_ts",
                existing_type=columns["max_processed_ts"]["type"],
                nullable=False,
            )

        if "resume_cursor" not in columns:
            op.add_column(
                "slack_ingestion_state",
                sa.Column("resume_cursor", sa.String(length=255), nullable=True),
            )
        if "resume_min_ts" not in columns:
            op.add_column(
                "slack_ingestion_state",
                sa.Column("resume_min_ts", sa.String(length=50), nullable=True),
            )

    if _has_table(inspector, "event_candidates"):
        event_columns = _get_columns(inspector, "event_candidates")
        if "source_id" not in event_columns:
            op.add_column(
                "event_candidates",
                sa.Column(
                    "source_id",
                    sa.String(length=20),
                    nullable=True,
                    server_default=sa.text("'slack'"),
                ),
            )
            op.execute(
                sa.text(
                    "UPDATE event_candidates SET source_id = 'slack' "
                    "WHERE source_id IS NULL OR source_id = ''"
                )
            )
            op.alter_column(
                "event_candidates",
                "source_id",
                existing_type=sa.String(length=20),
                nullable=False,
                server_default=None,
            )
        else:
            op.execute(
                sa.text(
                    "UPDATE event_candidates SET source_id = 'slack' "
                    "WHERE source_id IS NULL OR source_id = ''"
                )
            )
            if event_columns["source_id"].get("nullable", True):
                op.alter_column(
                    "event_candidates",
                    "source_id",
                    existing_type=event_columns["source_id"]["type"],
                    nullable=False,
                )

    if not _has_table(inspector, "pipeline_tasks"):
        op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        op.create_table(
            "pipeline_tasks",
            sa.Column(
                "task_id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column("task_type", sa.String(length=100), nullable=False),
            sa.Column(
                "payload",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column(
                "priority",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("50"),
            ),
            sa.Column(
                "run_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "status",
                sa.String(length=20),
                nullable=False,
                server_default=sa.text("'queued'"),
            ),
            sa.Column(
                "attempts",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column(
                "max_attempts",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("5"),
            ),
            sa.Column("idempotency_key", sa.String(length=255), nullable=False),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.CheckConstraint("priority >= 0", name="pipeline_tasks_priority_check"),
            sa.CheckConstraint("attempts >= 0", name="pipeline_tasks_attempts_check"),
            sa.CheckConstraint(
                "max_attempts > 0",
                name="pipeline_tasks_max_attempts_check",
            ),
            sa.CheckConstraint(
                "status IN ('queued', 'in_progress', 'done', 'failed')",
                name="pipeline_tasks_status_check",
            ),
        )
        op.create_index(
            "pipeline_tasks_idempotency_key_idx",
            "pipeline_tasks",
            ["idempotency_key"],
            unique=True,
        )
        op.create_index(
            "pipeline_tasks_status_run_at_idx",
            "pipeline_tasks",
            ["status", "task_type", "run_at"],
        )


def downgrade() -> None:
    """Drop pipeline tasks and revert Slack ingestion schema rename."""

    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "pipeline_tasks"):
        op.drop_index("pipeline_tasks_status_run_at_idx", table_name="pipeline_tasks")
        op.drop_index("pipeline_tasks_idempotency_key_idx", table_name="pipeline_tasks")
        op.drop_table("pipeline_tasks")

    inspector = sa.inspect(bind)

    if _has_table(inspector, "event_candidates"):
        event_columns = _get_columns(inspector, "event_candidates")
        if "source_id" in event_columns:
            op.drop_column("event_candidates", "source_id")

    inspector = sa.inspect(bind)

    if _has_table(inspector, "slack_ingestion_state"):
        columns = _get_columns(inspector, "slack_ingestion_state")
        if "resume_cursor" in columns:
            op.drop_column("slack_ingestion_state", "resume_cursor")
        if "resume_min_ts" in columns:
            op.drop_column("slack_ingestion_state", "resume_min_ts")
        columns = _get_columns(inspector, "slack_ingestion_state")
        if "max_processed_ts" in columns:
            op.alter_column(
                "slack_ingestion_state",
                "max_processed_ts",
                new_column_name="last_processed_ts",
                existing_type=columns["max_processed_ts"]["type"],
                existing_nullable=False,
            )
        if not _has_table(inspector, "ingestion_state"):
            op.rename_table("slack_ingestion_state", "ingestion_state")
