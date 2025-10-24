"""Persistence-backed cache for Slack permalinks."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any, cast

from src.config.logging_config import get_logger

__all__ = ["SlackPermalinkCache"]

logger = get_logger(__name__)

GetConnCallable = Callable[[], AbstractContextManager[Any]]


def _is_sqlite_connection(conn: Any) -> bool:
    """Return True if the connection comes from sqlite3."""

    module_name = cast(str, getattr(conn.__class__, "__module__", ""))
    return module_name.startswith("sqlite3")


class SlackPermalinkCache:
    """Database-backed cache for Slack message permalinks."""

    def __init__(self, get_conn: GetConnCallable) -> None:
        """Initialize cache with a connection provider."""

        self._get_conn = get_conn

    def get(self, channel_id: str, ts: str) -> str | None:
        """Return cached permalink for channel/timestamp pair if present."""

        if not channel_id or not ts:
            return None

        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                if _is_sqlite_connection(conn):
                    cursor.execute(
                        """
                        SELECT url FROM slack_permalink_cache
                        WHERE channel_id = ? AND ts = ?
                        """,
                        (channel_id, ts),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT url FROM slack_permalink_cache
                        WHERE channel_id = %s AND ts = %s
                        """,
                        (channel_id, ts),
                    )
                row = cast(tuple[Any, ...] | None, cursor.fetchone())
            finally:
                cursor.close()

        if not row:
            return None

        return cast(str, row[0])

    def put(self, channel_id: str, ts: str, url: str) -> None:
        """Persist permalink for future lookups."""

        if not channel_id or not ts or not url:
            return

        updated_at = datetime.now(UTC).isoformat()

        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                if _is_sqlite_connection(conn):
                    cursor.execute(
                        """
                        INSERT INTO slack_permalink_cache (channel_id, ts, url, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(channel_id, ts) DO UPDATE SET
                            url = excluded.url,
                            updated_at = excluded.updated_at
                        """,
                        (channel_id, ts, url, updated_at),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO slack_permalink_cache (channel_id, ts, url, updated_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (channel_id, ts) DO UPDATE SET
                            url = EXCLUDED.url,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (channel_id, ts, url, updated_at),
                    )
                conn.commit()
            finally:
                cursor.close()
