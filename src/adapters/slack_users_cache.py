"""Persistence-backed cache for Slack user profiles."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any, cast

from src.config.logging_config import get_logger

__all__ = ["SlackUsersCache"]

logger = get_logger(__name__)

GetConnCallable = Callable[[], AbstractContextManager[Any]]
JSONDict = dict[str, Any]


def _is_sqlite_connection(conn: Any) -> bool:
    """Return True if the connection comes from sqlite3."""

    module_name = cast(str, getattr(conn.__class__, "__module__", ""))
    return module_name.startswith("sqlite3")


class SlackUsersCache:
    """Database-backed cache for Slack user profiles."""

    def __init__(self, get_conn: GetConnCallable) -> None:
        """Initialize cache with a connection provider."""

        self._get_conn = get_conn

    def get(self, user_id: str) -> JSONDict | None:
        """Fetch cached user profile if available."""

        if not user_id:
            return None

        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                if _is_sqlite_connection(conn):
                    cursor.execute(
                        "SELECT profile_json FROM slack_users_cache WHERE user_id = ?",
                        (user_id,),
                    )
                else:
                    cursor.execute(
                        "SELECT profile_json FROM slack_users_cache WHERE user_id = %s",
                        (user_id,),
                    )
                row = cast(tuple[Any, ...] | None, cursor.fetchone())
            finally:
                cursor.close()

        if not row:
            return None

        payload = cast(str, row[0])
        try:
            return cast(dict[str, Any], json.loads(payload))
        except json.JSONDecodeError:
            logger.warning("slack_users_cache_invalid_json", user_id=user_id)
            return None

    def put(self, user_id: str, profile: JSONDict) -> None:
        """Store or update a user profile in the cache."""

        if not user_id:
            return

        serialized = json.dumps(profile, separators=(",", ":"), sort_keys=True)
        updated_at = datetime.now(UTC).isoformat()

        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                if _is_sqlite_connection(conn):
                    cursor.execute(
                        """
                        INSERT INTO slack_users_cache (user_id, profile_json, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET
                            profile_json = excluded.profile_json,
                            updated_at = excluded.updated_at
                        """,
                        (user_id, serialized, updated_at),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO slack_users_cache (user_id, profile_json, updated_at)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id) DO UPDATE SET
                            profile_json = EXCLUDED.profile_json,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (user_id, serialized, updated_at),
                    )
                conn.commit()
            finally:
                cursor.close()
