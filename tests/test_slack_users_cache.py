from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import pytest

from src.clients.slack_wrapped import SlackClient


@pytest.fixture
def conn_factory() -> Callable[[], AbstractContextManager[sqlite3.Connection]]:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE slack_users_cache (
            user_id TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL,
            updated_at TEXT
        );
        CREATE TABLE slack_permalink_cache (
            channel_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            url TEXT NOT NULL,
            updated_at TEXT,
            PRIMARY KEY (channel_id, ts)
        );
        """
    )
    conn.commit()

    @contextmanager
    def _ctx() -> Iterator[sqlite3.Connection]:
        yield conn

    try:
        yield _ctx
    finally:
        conn.close()


class StubWebClient:
    def __init__(self) -> None:
        self.users_info_calls = 0

    def users_info(self, *, user: str) -> dict[str, object]:
        self.users_info_calls += 1
        return {
            "ok": True,
            "user": {
                "id": user,
                "name": "Test User",
                "profile": {"real_name": "Test User"},
            },
        }

    def chat_getPermalink(self, *, channel: str, message_ts: str) -> dict[str, object]:  # noqa: N802
        return {"ok": True, "permalink": f"https://slack.test/{channel}/{message_ts}"}


def test_user_cache_prevents_second_api_call(
    conn_factory: Callable[[], AbstractContextManager[sqlite3.Connection]],
) -> None:
    client = SlackClient(
        bot_token="xoxb-test", client=StubWebClient(), get_conn=conn_factory
    )

    profile_first = client.get_user_info("U123")
    profile_second = client.get_user_info("U123")

    assert profile_first == profile_second
    assert client.client.users_info_calls == 1
