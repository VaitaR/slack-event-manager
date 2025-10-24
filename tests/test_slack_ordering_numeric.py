from __future__ import annotations

from typing import Any

import pytest

from src.domain.models import ChannelConfig
from src.use_cases.ingest_messages import ingest_messages_use_case
from tests.test_slack_ingest_watermark_resume import DummySettings, InMemoryRepository


class OrderingSlackClient:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages

    def _fetch_page_with_retries(
        self, channel_id: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        limit = params.get("limit")
        if limit is not None:
            payload = self._messages[:limit]
        else:
            payload = self._messages
        return {
            "ok": True,
            "messages": payload,
            "response_metadata": {"next_cursor": ""},
        }

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        return {"id": user_id, "profile": {}}

    def get_permalink(self, channel_id: str, ts: str) -> str:
        return f"https://slack.test/{channel_id}/{ts}"


def test_slack_ordering_numeric() -> None:
    repository = InMemoryRepository()
    try:
        messages = [
            {"ts": "1720000000.200", "user": "U1", "text": "later"},
            {"ts": "1720000000.010", "user": "U2", "text": "earliest"},
            {"ts": "1720000000.090", "user": "U3", "text": "middle"},
        ]
        slack_client = OrderingSlackClient(messages)
        settings = DummySettings(
            lookback_hours_default=24,
            slack_channels=[ChannelConfig(channel_id="C123", channel_name="Test")],
            slack_max_messages_per_run=None,
            slack_page_size=5,
        )

        result = ingest_messages_use_case(slack_client, repository, settings)
        assert result.messages_saved == 3
        assert result.errors == []

        saved_ts = [float(message.ts) for message in repository.saved]
        assert saved_ts == pytest.approx(sorted(saved_ts))
    finally:
        repository.close()
