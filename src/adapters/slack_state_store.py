"""Slack ingestion state persistence adapter."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from typing import Any, Final, Protocol


class CursorProtocol(Protocol):
    """Protocol for database cursors used by the state store."""

    def execute(self, query: str, params: tuple[Any, ...]) -> Any:
        """Execute a SQL statement with positional parameters."""

    def fetchone(self) -> Any:
        """Fetch a single result row."""

    def close(self) -> None:
        """Release cursor resources."""


class ConnectionProtocol(Protocol):
    """Protocol for connections compatible with the state store."""

    def cursor(self) -> CursorProtocol:
        """Create a database cursor."""

    def commit(self) -> None:
        """Commit the active transaction."""


GetConnectionCallable = Callable[[], AbstractContextManager[ConnectionProtocol]]


class SlackStateStore:
    """Persistence layer for Slack ingestion state."""

    def __init__(self, get_conn: GetConnectionCallable) -> None:
        """Initialize the store.

        Args:
            get_conn: Callable returning a context manager that yields a database connection.
        """
        self._get_conn = get_conn

    _TABLE_NAME: Final[str] = "slack_ingestion_state"

    def get(self, channel_id: str) -> dict[str, float | str | None]:
        """Load state for a channel.

        Args:
            channel_id: Slack channel identifier.

        Returns:
            Dictionary with `max_processed_ts`, `resume_cursor`, `resume_min_ts` keys.
        """
        with self._connection_scope() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    self._build_select_sql(conn),
                    (channel_id,),
                )
                row = cursor.fetchone()
            finally:
                cursor.close()

        if not row:
            return {
                "max_processed_ts": 0.0,
                "resume_cursor": None,
                "resume_min_ts": None,
            }

        return {
            "max_processed_ts": float(row[0]) if row[0] is not None else 0.0,
            "resume_cursor": row[1],
            "resume_min_ts": float(row[2]) if row[2] is not None else None,
        }

    def upsert(
        self,
        channel_id: str,
        *,
        max_processed_ts: float,
        resume_cursor: str | None,
        resume_min_ts: float | None,
    ) -> None:
        """Persist ingestion state for a channel."""
        timestamp = datetime.now(UTC).isoformat()

        with self._connection_scope() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    self._build_upsert_sql(conn),
                    (
                        channel_id,
                        max_processed_ts,
                        resume_cursor,
                        resume_min_ts,
                        timestamp,
                    ),
                )
                conn.commit()
            finally:
                cursor.close()

    @contextmanager
    def _connection_scope(self) -> Iterator[ConnectionProtocol]:
        with self._get_conn() as conn:
            yield conn

    @staticmethod
    def _is_sqlite(conn: ConnectionProtocol) -> bool:
        return isinstance(conn, sqlite3.Connection)

    def _build_select_sql(self, conn: ConnectionProtocol) -> str:
        placeholder = "?" if self._is_sqlite(conn) else "%s"
        return (
            "SELECT max_processed_ts, resume_cursor, resume_min_ts "
            f"FROM {self._TABLE_NAME} WHERE channel_id = " + placeholder
        )

    def _build_upsert_sql(self, conn: ConnectionProtocol) -> str:
        if self._is_sqlite(conn):
            return (
                f"INSERT INTO {self._TABLE_NAME} (channel_id, max_processed_ts, resume_cursor, "
                "resume_min_ts, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(channel_id) DO UPDATE SET "
                "max_processed_ts=excluded.max_processed_ts, "
                "resume_cursor=excluded.resume_cursor, "
                "resume_min_ts=excluded.resume_min_ts, "
                "updated_at=excluded.updated_at"
            )
        return (
            f"INSERT INTO {self._TABLE_NAME} (channel_id, max_processed_ts, resume_cursor, "
            "resume_min_ts, updated_at) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (channel_id) DO UPDATE SET "
            "max_processed_ts=EXCLUDED.max_processed_ts, "
            "resume_cursor=EXCLUDED.resume_cursor, "
            "resume_min_ts=EXCLUDED.resume_min_ts, "
            "updated_at=EXCLUDED.updated_at"
        )
