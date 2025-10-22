"""Tests for SlackClient pagination behavior."""

from typing import Any

import pytest

from src.adapters.slack_client import SlackClient


@pytest.fixture(autouse=True)
def patch_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real sleep during tests."""

    monkeypatch.setattr("src.adapters.slack_client.time.sleep", lambda *_: None)


def test_fetch_messages_paginates_until_cursor_exhausted() -> None:
    """SlackClient should fetch all pages when limit is not provided."""

    class StubWebClient:
        def __init__(self, queued_responses: list[dict[str, Any]]) -> None:
            self._queued_responses = queued_responses
            self.calls: list[dict[str, Any]] = []

        def conversations_history(self, **params: Any) -> dict[str, Any]:
            self.calls.append(params)
            if not self._queued_responses:
                raise AssertionError(
                    "No response available for conversations_history call"
                )
            return self._queued_responses.pop(0)

        def users_info(self, user: str) -> dict[str, Any]:
            return {"ok": True, "user": {"id": user}}

    responses: list[dict[str, Any]] = [
        {
            "ok": True,
            "messages": [
                {"ts": "1", "thread_ts": None, "text": "first"},
                {"ts": "2", "thread_ts": None, "text": "second"},
            ],
            "response_metadata": {"next_cursor": "cursor-1"},
        },
        {
            "ok": True,
            "messages": [
                {"ts": "3", "thread_ts": None, "text": "third"},
            ],
            "response_metadata": {},
        },
    ]

    stub_web_client = StubWebClient(responses)
    client = SlackClient(bot_token="test-token", page_size=2, client=stub_web_client)

    messages = client.fetch_messages(channel_id="C123")

    assert len(messages) == 3
    assert stub_web_client.calls[0]["limit"] == 2
    assert stub_web_client.calls[1]["cursor"] == "cursor-1"
