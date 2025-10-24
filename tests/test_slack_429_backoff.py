from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import pytest
from slack_sdk.errors import SlackApiError
from slack_sdk.web.client import WebClient
from slack_sdk.web.slack_response import SlackResponse

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


class RateLimitOnceClient:
    def __init__(self) -> None:
        self.calls = 0
        self._web_client = WebClient(token="xoxb-test")

    def chat_getPermalink(self, *, channel: str, message_ts: str) -> dict[str, object]:  # noqa: N802
        if self.calls == 0:
            self.calls += 1
            response = SlackResponse(
                client=self._web_client,
                http_verb="GET",
                api_url="https://slack.com/api/chat.getPermalink",
                req_args={},
                data={"ok": False, "error": "ratelimited"},
                headers={"Retry-After": "1"},
                status_code=429,
            )
            raise SlackApiError(message="ratelimited", response=response)

        self.calls += 1
        return {"ok": True, "permalink": f"https://slack.test/{channel}/{message_ts}"}

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


def test_rate_limit_backoff(
    monkeypatch: pytest.MonkeyPatch,
    conn_factory: Callable[[], AbstractContextManager[sqlite3.Connection]],
) -> None:
    stub_client = RateLimitOnceClient()
    sleep_calls: list[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    client = SlackClient(
        bot_token="xoxb-test",
        client=stub_client,
        get_conn=conn_factory,
        sleep=fake_sleep,
    )
    monkeypatch.setattr(client._random, "uniform", lambda _a, _b: 0.0)

    permalink = client.get_permalink("C123", "111.222")

    assert permalink == "https://slack.test/C123/111.222"
    assert stub_client.calls == 2
    assert sum(sleep_calls) >= 1.0
