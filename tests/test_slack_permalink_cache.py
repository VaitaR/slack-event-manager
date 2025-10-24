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


class StubPermalinkClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_getPermalink(self, *, channel: str, message_ts: str) -> dict[str, object]:  # noqa: N802
        self.calls += 1
        return {
            "ok": True,
            "permalink": f"https://slack.test/{channel}/{message_ts}",
        }

    def users_info(
        self, *, user: str
    ) -> dict[str, object]:  # pragma: no cover - unused fallback
        return {
            "ok": True,
            "user": {
                "id": user,
                "name": "Test",
                "profile": {"real_name": "Test"},
            },
        }


def test_permalink_cache_hits_database_after_first_call(
    conn_factory: Callable[[], AbstractContextManager[sqlite3.Connection]],
) -> None:
    stub_client = StubPermalinkClient()
    client = SlackClient(
        bot_token="xoxb-test", client=stub_client, get_conn=conn_factory
    )

    permalink_first = client.get_permalink("C123", "111.222")
    permalink_second = client.get_permalink("C123", "111.222")

    assert permalink_first == permalink_second
    assert stub_client.calls == 1
