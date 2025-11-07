"""PostgreSQL implementation of the task queue port."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from psycopg2.extras import RealDictCursor

from src.config.logging_config import get_logger
from src.domain.exceptions import RepositoryError
from src.domain.task_queue import Task, TaskCreate, TaskStatus, TaskType
from src.ports.task_queue import TaskQueuePort

logger = get_logger(__name__)


class PostgresTaskQueue(TaskQueuePort):
    """Task queue backed by a PostgreSQL table."""

    def __init__(self, connection_provider: Callable[[], AbstractContextManager[Any]]):
        self._connection_provider = connection_provider

    def enqueue(self, task: TaskCreate) -> Task:
        results = self.enqueue_many([task])
        return results[0]

    def enqueue_many(self, tasks: list[TaskCreate]) -> list[Task]:
        if not tasks:
            return []

        inserted_rows: list[dict[str, Any]] = []
        conflicts: list[str] = []

        with self._connection_provider() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for task in tasks:
                    cur.execute(
                        """
                        INSERT INTO pipeline_tasks (
                            task_id,
                            task_type,
                            payload,
                            priority,
                            run_at,
                            status,
                            attempts,
                            max_attempts,
                            idempotency_key,
                            last_error,
                            locked_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s, NULL, NULL)
                        ON CONFLICT (idempotency_key) DO NOTHING
                        RETURNING *
                        """,
                        (
                            uuid4(),
                            task.task_type.value,
                            json.dumps(task.payload),
                            task.priority,
                            _ensure_utc(task.run_at),
                            TaskStatus.QUEUED.value,
                            task.max_attempts,
                            task.idempotency_key,
                        ),
                    )
                    row = cur.fetchone()
                    if row is None:
                        conflicts.append(task.idempotency_key)
                    else:
                        inserted_rows.append(dict(row))

                if conflicts:
                    cur.execute(
                        """
                        SELECT * FROM pipeline_tasks
                        WHERE idempotency_key = ANY(%s)
                        """,
                        (conflicts,),
                    )
                    existing = cur.fetchall()
                    inserted_rows.extend(dict(item) for item in existing)

                conn.commit()

        return [Task.model_validate(row) for row in inserted_rows]

    def lease(self, task_type: TaskType, limit: int) -> list[Task]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        now = datetime.now(tz=UTC)
        with self._connection_provider() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    WITH candidates AS (
                        SELECT task_id
                        FROM pipeline_tasks
                        WHERE status = %s
                          AND task_type = %s
                          AND run_at <= %s
                        ORDER BY priority ASC, run_at ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT %s
                    )
                    UPDATE pipeline_tasks AS t
                    SET status = %s,
                        attempts = attempts + 1,
                        locked_at = %s,
                        updated_at = %s
                    FROM candidates
                    WHERE t.task_id = candidates.task_id
                    RETURNING t.*
                    """,
                    (
                        TaskStatus.QUEUED.value,
                        task_type.value,
                        now,
                        limit,
                        TaskStatus.IN_PROGRESS.value,
                        now,
                        now,
                    ),
                )
                rows = cur.fetchall()
                conn.commit()

        return [Task.model_validate(dict(row)) for row in rows]

    def complete(self, task_id: UUID) -> None:
        now = datetime.now(tz=UTC)
        with self._connection_provider() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_tasks
                    SET status = %s,
                        last_error = NULL,
                        locked_at = NULL,
                        updated_at = %s
                    WHERE task_id = %s
                    """,
                    (TaskStatus.DONE.value, now, task_id),
                )
                if cur.rowcount == 0:
                    conn.rollback()
                    raise RepositoryError(f"Task not found: {task_id}")
                conn.commit()

    def fail(self, task_id: UUID, *, error: str, retry_at: datetime | None) -> None:
        retry_at_utc = _ensure_optional_utc(retry_at)
        now = datetime.now(tz=UTC)

        with self._connection_provider() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT attempts, max_attempts, run_at
                    FROM pipeline_tasks
                    WHERE task_id = %s
                    FOR UPDATE
                    """,
                    (task_id,),
                )
                row = cur.fetchone()
                if row is None:
                    conn.rollback()
                    raise RepositoryError(f"Task not found: {task_id}")

                attempts = int(row["attempts"])
                max_attempts = int(row["max_attempts"])
                should_retry = retry_at_utc is not None and attempts < max_attempts
                next_status = (
                    TaskStatus.QUEUED.value if should_retry else TaskStatus.FAILED.value
                )
                next_run_at = retry_at_utc if should_retry else row["run_at"]

                cur.execute(
                    """
                    UPDATE pipeline_tasks
                    SET status = %s,
                        run_at = %s,
                        last_error = %s,
                        locked_at = NULL,
                        updated_at = %s
                    WHERE task_id = %s
                    """,
                    (
                        next_status,
                        _ensure_optional_utc(next_run_at),
                        error,
                        now,
                        task_id,
                    ),
                )
                conn.commit()


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _ensure_optional_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return _ensure_utc(value)


__all__ = ["PostgresTaskQueue"]
