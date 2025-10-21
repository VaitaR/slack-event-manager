"""Tests for SlackClient pagination behavior."""

from typing import Any

import pytest

from src.adapters.slack_client import SlackClient


@pytest.fixture(autouse=True)
def patch_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real sleep during tests."""

    monkeypatch.setattr("src.adapters.slack_client.time.sleep", lambda *_: None)


def test_fetch_messages_paginates_until_cursor_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SlackClient should fetch all pages when limit is not provided."""

    client = SlackClient(bot_token="test-token", page_size=2)

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

    call_history: list[dict[str, Any]] = []

    def fake_history(**kwargs: Any) -> dict[str, Any]:
        call_history.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(client.client, "conversations_history", fake_history)

    messages = client.fetch_messages(channel_id="C123")

    assert len(messages) == 3
    assert call_history[0]["limit"] == 2
    assert call_history[1]["cursor"] == "cursor-1"
