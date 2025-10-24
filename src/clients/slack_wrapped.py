"""Slack client decorator that adds persistence caching and rate-limit handling."""

from __future__ import annotations

import os
import random
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Final, cast

from slack_sdk.errors import SlackApiError

from src.adapters.slack_client import (
    DEFAULT_SLACK_MAX_RETRIES,
    DEFAULT_SLACK_PAGE_DELAY_SECONDS,
)
from src.adapters.slack_client import (
    SlackClient as BaseSlackClient,
)
from src.adapters.slack_permalink_cache import SlackPermalinkCache
from src.adapters.slack_users_cache import SlackUsersCache
from src.config.logging_config import get_logger
from src.config.settings import Settings, get_settings
from src.domain.exceptions import RateLimitError, SlackAPIError

__all__ = ["SlackClient"]

logger = get_logger(__name__)

GetConnCallable = Callable[[], AbstractContextManager[Any]]
SleepCallable = Callable[[float], None]
MAX_RATE_LIMIT_ATTEMPTS: Final[int] = 5
MAX_JITTER_SECONDS: Final[float] = 0.5
USERS_PREFETCH_PAGE_SIZE: Final[int] = 200
HTTP_STATUS_TOO_MANY_REQUESTS: Final[int] = 429


def _sqlite_connection_factory(db_path: str) -> GetConnCallable:
    @contextmanager
    def _conn() -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(db_path)
        try:
            yield connection
        finally:
            connection.close()

    return _conn


def _postgres_connection_factory(settings: Settings) -> GetConnCallable:
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "psycopg2 must be installed for PostgreSQL caching support"
        ) from exc

    password = settings.postgres_password
    if password is None:
        raise RuntimeError(
            "POSTGRES_PASSWORD must be configured for PostgreSQL caching support"
        )

    @contextmanager
    def _conn() -> Iterator[Any]:
        connection = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_database,
            user=settings.postgres_user,
            password=password.get_secret_value(),
            connect_timeout=settings.postgres_connect_timeout_seconds,
        )
        try:
            yield connection
        finally:
            connection.close()

    return _conn


def _build_default_get_conn() -> GetConnCallable:
    settings = get_settings()
    if settings.database_type == "sqlite":
        return _sqlite_connection_factory(settings.db_path)
    if settings.database_type == "postgres":
        return _postgres_connection_factory(settings)

    raise RuntimeError(f"Unsupported database type: {settings.database_type}")


class SlackClient(BaseSlackClient):
    """Slack client wrapper with persistent caching and enhanced 429 handling."""

    def __init__(
        self,
        bot_token: str,
        *,
        page_size: int | None = None,
        max_total_messages: int | None = None,
        page_delay_seconds: float | None = None,
        max_retries: int | None = None,
        client: Any | None = None,
        users_cache: SlackUsersCache | None = None,
        permalink_cache: SlackPermalinkCache | None = None,
        get_conn: GetConnCallable | None = None,
        sleep: SleepCallable | None = None,
    ) -> None:
        self._sleep = sleep or time.sleep
        self._random = random.Random()
        self._get_conn = get_conn or _build_default_get_conn()
        self._users_cache = users_cache or SlackUsersCache(self._get_conn)
        self._permalink_cache = permalink_cache or SlackPermalinkCache(self._get_conn)

        super().__init__(
            bot_token,
            page_size=page_size,
            max_total_messages=max_total_messages,
            page_delay_seconds=(
                page_delay_seconds
                if page_delay_seconds is not None
                else DEFAULT_SLACK_PAGE_DELAY_SECONDS
            ),
            max_retries=(
                max_retries if max_retries is not None else DEFAULT_SLACK_MAX_RETRIES
            ),
            client=client,
        )

        if self._should_prefetch_users():
            self._prefetch_users()

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        cached = self._users_cache.get(user_id)
        if cached is not None:
            return cached

        response = self._call_with_backoff(
            lambda: self.client.users_info(user=user_id),
            action="users_info",
            context={"user_id": user_id},
        )
        data = self._extract_data(response)
        if not data.get("ok", False):
            logger.warning(
                "slack_user_fetch_failed",
                user_id=user_id,
                error=data.get("error"),
            )
            return {"real_name": "Unknown", "name": "unknown"}

        user = cast(dict[str, Any], data.get("user", {}))
        if user:
            self._users_cache.put(user_id, user)
        return user

    def get_permalink(self, channel_id: str, message_ts: str) -> str | None:
        cached = self._permalink_cache.get(channel_id, message_ts)
        if cached is not None:
            return cached

        response = self._call_with_backoff(
            lambda: self.client.chat_getPermalink(
                channel=channel_id, message_ts=message_ts
            ),
            action="chat_getPermalink",
            context={"channel_id": channel_id, "message_ts": message_ts},
        )
        data = self._extract_data(response)
        if not data.get("ok", False):
            return None

        permalink = cast(str | None, data.get("permalink"))
        if permalink:
            self._permalink_cache.put(channel_id, message_ts, permalink)
        return permalink

    def _prefetch_users(self) -> None:
        logger.info("slack_users_prefetch_started")
        cursor: str | None = None
        total_cached = 0

        while True:
            cursor_value = cursor

            def _fetch_users_list() -> Any:
                return self.client.users_list(
                    cursor=cursor_value, limit=USERS_PREFETCH_PAGE_SIZE
                )

            response = self._call_with_backoff(
                _fetch_users_list,
                action="users_list",
            )
            data = self._extract_data(response)
            members = cast(list[dict[str, Any]], data.get("members", []))
            for member in members:
                user_id = member.get("id")
                if user_id:
                    self._users_cache.put(str(user_id), member)
                    total_cached += 1

            cursor = cast(
                str | None,
                data.get("response_metadata", {}).get("next_cursor")
                if isinstance(data.get("response_metadata"), dict)
                else None,
            )
            if not cursor:
                break

        logger.info("slack_users_prefetch_completed", users_cached=total_cached)

    def _should_prefetch_users(self) -> bool:
        return os.getenv("SLACK_PREFETCH_USERS", "false").lower() == "true"

    def _extract_data(self, response: Any) -> dict[str, Any]:
        if hasattr(response, "data"):
            return cast(dict[str, Any], response.data)
        return cast(dict[str, Any], response)

    def _call_with_backoff(
        self,
        func: Callable[[], Any],
        *,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        attempts = 0
        while True:
            try:
                return func()
            except SlackApiError as exc:
                retry_after = self._retry_after_seconds(exc)
                if retry_after is None:
                    logger.warning(
                        "slack_api_error_no_retry_after",
                        action=action,
                        error=str(exc),
                        context=context or {},
                    )
                    raise SlackAPIError(str(exc)) from exc

                if attempts >= MAX_RATE_LIMIT_ATTEMPTS:
                    raise RateLimitError(retry_after=int(retry_after)) from exc

                attempts += 1
                logger.warning(
                    "slack_rate_limit_backoff",
                    action=action,
                    attempt=attempts,
                    retry_after_seconds=retry_after,
                    context=context or {},
                )
                self._sleep_with_backoff(retry_after, attempts - 1)
            except Exception as exc:  # pragma: no cover - safety net
                raise SlackAPIError(str(exc)) from exc

    def _sleep_with_backoff(self, base_seconds: float, attempt: int) -> None:
        delay = base_seconds * (2**attempt) + self._random.uniform(
            0.0, MAX_JITTER_SECONDS
        )
        self._sleep(delay)

    def _retry_after_seconds(self, error: SlackApiError) -> float | None:
        response = getattr(error, "response", None)
        if response is None:
            return None

        headers = getattr(response, "headers", {}) or {}
        retry_after_str = headers.get("Retry-After") or headers.get("retry-after")
        if (
            retry_after_str is None
            and getattr(response, "status_code", None) != HTTP_STATUS_TOO_MANY_REQUESTS
        ):
            return None

        if retry_after_str is None:
            retry_after_str = "1"

        try:
            return max(float(retry_after_str), 0.0)
        except (TypeError, ValueError):
            return None
