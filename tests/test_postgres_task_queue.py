from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, ContextManager
from uuid import UUID, uuid4

from pytest_mock import MockerFixture

from src.adapters.postgres_task_queue import PostgresTaskQueue
from src.domain.task_queue import TaskCreate, TaskStatus, TaskType


def _make_connection(
    mocker: MockerFixture,
) -> tuple[Any, Any, Callable[[], ContextManager[Any]]]:
    """Create mocked psycopg connection and cursor."""

    connection = mocker.MagicMock()
    cursor = mocker.MagicMock()
    cursor_cm = mocker.MagicMock()
    cursor_cm.__enter__.return_value = cursor
    cursor_cm.__exit__.return_value = False
    connection.cursor.return_value = cursor_cm

    @contextmanager
    def _connection_ctx() -> Iterator[Any]:
        yield connection

    def _provider() -> ContextManager[Any]:
        return _connection_ctx()

    return connection, cursor, _provider


def _queue(connection_provider: Callable[[], ContextManager[Any]]) -> PostgresTaskQueue:
    """Create queue instance from connection provider."""

    return PostgresTaskQueue(connection_provider=connection_provider)


def _build_row(task_id: UUID | None = None) -> dict[str, Any]:
    """Return dictionary that mirrors a queue row."""

    now = datetime.now(tz=UTC)
    return {
        "task_id": (task_id or uuid4()),
        "task_type": TaskType.INGEST.value,
        "payload": {"channel": "C123"},
        "priority": 10,
        "run_at": now,
        "status": TaskStatus.QUEUED.value,
        "attempts": 0,
        "max_attempts": 5,
        "idempotency_key": "ingest:C123",
        "last_error": None,
        "locked_at": None,
        "created_at": now,
        "updated_at": now,
    }


def test_enqueue_task_inserts_record(mocker: MockerFixture) -> None:
    connection, cursor, provider = _make_connection(mocker)
    row = _build_row()
    cursor.fetchone.return_value = row

    queue = _queue(provider)

    created = queue.enqueue(
        TaskCreate(
            task_type=TaskType.INGEST,
            payload={"channel": "C123"},
            idempotency_key="ingest:C123",
            priority=10,
            run_at=row["run_at"],
            max_attempts=5,
        )
    )

    cursor.execute.assert_called()
    connection.commit.assert_called_once()
    assert created.task_type is TaskType.INGEST
    assert created.status is TaskStatus.QUEUED
    assert created.idempotency_key == "ingest:C123"


def test_enqueue_task_returns_existing_on_conflict(mocker: MockerFixture) -> None:
    connection, cursor, provider = _make_connection(mocker)
    row = _build_row()
    cursor.fetchone.side_effect = [None]
    cursor.fetchall.return_value = [row]

    queue = _queue(provider)

    created = queue.enqueue(
        TaskCreate(
            task_type=TaskType.INGEST,
            payload={"channel": "C456"},
            idempotency_key="ingest:C456",
        )
    )

    # second select should fetch existing row
    assert cursor.execute.call_count >= 2
    connection.commit.assert_called_once()
    assert created.task_type is TaskType.INGEST


def test_lease_tasks_marks_in_progress(mocker: MockerFixture) -> None:
    connection, cursor, provider = _make_connection(mocker)
    row = _build_row()
    row["status"] = TaskStatus.IN_PROGRESS.value
    row["attempts"] = 1
    cursor.fetchall.return_value = [row]

    queue = _queue(provider)

    leased = queue.lease(TaskType.INGEST, limit=4)

    cursor.execute.assert_called()
    connection.commit.assert_called_once()
    assert len(leased) == 1
    assert leased[0].status is TaskStatus.IN_PROGRESS
    assert leased[0].attempts == 1


def test_fail_task_with_retry_requeues(mocker: MockerFixture) -> None:
    connection, cursor, provider = _make_connection(mocker)
    row = _build_row()
    row["attempts"] = 1
    row["max_attempts"] = 5
    cursor.fetchone.return_value = row

    queue = _queue(provider)

    retry_at = datetime.now(tz=UTC) + timedelta(seconds=30)
    queue.fail(task_id=row["task_id"], error="temporary", retry_at=retry_at)

    cursor.execute.assert_called()
    connection.commit.assert_called_once()


def test_fail_task_without_retry_marks_failed(mocker: MockerFixture) -> None:
    connection, cursor, provider = _make_connection(mocker)
    row = _build_row()
    row["attempts"] = row["max_attempts"]
    cursor.fetchone.return_value = row

    queue = _queue(provider)

    queue.fail(task_id=row["task_id"], error="fatal", retry_at=None)

    cursor.execute.assert_called()
    connection.commit.assert_called_once()

