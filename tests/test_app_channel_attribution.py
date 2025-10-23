"""Tests for presentation-layer channel attribution helpers."""

from __future__ import annotations

from typing import Any

import pytest

from app import fetch_channel_messages


class StubSlackClient:
    """Test double capturing channel fetch requests."""

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, int | None]] = []

    def fetch_messages(
        self, *, channel_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        self.calls.append((channel_id, limit))
        return list(self._responses.get(channel_id, []))


@pytest.mark.parametrize(
    "channels,limit",
    [
        (
            [
                "C123",
                "C999",
            ],
            2,
        )
    ],
)
def test_fetch_channel_messages_preserves_channel_identity(
    channels: list[str], limit: int | None
) -> None:
    """Messages must remain associated with their origin channel."""

    responses: dict[str, list[dict[str, Any]]] = {
        "C123": [
            {"ts": "1", "text": "hello"},
            {"ts": "2", "text": "world"},
        ],
        "C999": [
            {"ts": "9", "text": "other"},
        ],
    }
    client = StubSlackClient(responses)

    collected = fetch_channel_messages(client, channels=channels, limit=limit)

    assert collected == [
        ("C123", {"ts": "1", "text": "hello"}),
        ("C123", {"ts": "2", "text": "world"}),
        ("C999", {"ts": "9", "text": "other"}),
    ]
    assert client.calls == [("C123", limit), ("C999", limit)]
