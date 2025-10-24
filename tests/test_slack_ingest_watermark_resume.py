from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from src.domain.models import ChannelConfig
from src.use_cases.ingest_messages import ingest_messages_use_case

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS slack_ingestion_state (
  channel_id TEXT PRIMARY KEY,
  max_processed_ts REAL NOT NULL DEFAULT 0,
  resume_cursor TEXT,
  resume_min_ts REAL,
  updated_at TEXT
);
"""


class InMemoryRepository:
    def __init__(self) -> None:
        self._connection = sqlite3.connect(":memory:")
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(CREATE_SQL)
        self._connection.commit()
        self.saved: list[Any] = []

    @contextmanager
    def _get_connection(self) -> sqlite3.Connection:
        yield self._connection

    def save_messages(self, messages: list[Any]) -> int:
        self.saved.extend(messages)
        return len(messages)

    def get_last_processed_ts(self, channel_id: str) -> float | None:
        state = self.get_state(channel_id)
        if state is None:
            return None
        value = state["max_processed_ts"]
        return float(value) if value is not None else None

    def get_state(self, channel_id: str) -> sqlite3.Row | None:
        cursor = self._connection.execute(
            "SELECT max_processed_ts, resume_cursor, resume_min_ts FROM slack_ingestion_state WHERE channel_id = ?",
            (channel_id,),
        )
        try:
            return cursor.fetchone()
        finally:
            cursor.close()

    def close(self) -> None:
        self._connection.close()


class FakeSlackClient:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = sorted(messages, key=lambda msg: float(msg["ts"]))

    def _fetch_page_with_retries(
        self, channel_id: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        limit = params["limit"]
        cursor_value = params.get("cursor")
        oldest_param = params.get("oldest")
        oldest_float = float(oldest_param) if oldest_param is not None else None

        filtered = [
            msg
            for msg in self.messages
            if oldest_float is None or float(msg["ts"]) >= oldest_float
        ]

        start_index = 0
        if cursor_value:
            start_index = int(cursor_value.split("-", maxsplit=1)[1])

        page_messages = filtered[start_index : start_index + limit]
        next_index = start_index + len(page_messages)
        next_cursor = f"cursor-{next_index}" if next_index < len(filtered) else ""

        return {
            "ok": True,
            "messages": page_messages,
            "response_metadata": {"next_cursor": next_cursor},
        }

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        return {"id": user_id, "profile": {}}

    def get_permalink(self, channel_id: str, ts: str) -> str:
        return f"https://slack.test/{channel_id}/{ts}"


@dataclass
class DummySettings:
    lookback_hours_default: int
    slack_channels: list[ChannelConfig]
    slack_max_messages_per_run: int | None
    slack_page_size: int


def _build_messages(base_ts: float, count: int) -> list[dict[str, Any]]:
    return [
        {
            "ts": f"{base_ts + float(index) + 1.0:.6f}",
            "user": f"U{index}",
            "text": f"message-{index}",
        }
        for index in range(count)
    ]


def test_slack_ingest_watermark_resume() -> None:
    repository = InMemoryRepository()
    try:
        base_ts = datetime.utcnow().timestamp() - 100.0
        messages = _build_messages(base_ts, 8)
        slack_client = FakeSlackClient(messages)
        settings = DummySettings(
            lookback_hours_default=24,
            slack_channels=[ChannelConfig(channel_id="C123", channel_name="Test")],
            slack_max_messages_per_run=5,
            slack_page_size=3,
        )

        first_run = ingest_messages_use_case(slack_client, repository, settings)
        assert first_run.messages_saved == 5
        assert first_run.errors == []

        state_after_first = repository.get_state("C123")
        assert state_after_first is not None
        assert state_after_first["resume_cursor"] == "cursor-5"
        assert state_after_first["max_processed_ts"] == pytest.approx(0.0)
        assert state_after_first["resume_min_ts"] is not None

        second_run = ingest_messages_use_case(slack_client, repository, settings)
        assert second_run.messages_saved == 3
        assert second_run.errors == []

        state_after_second = repository.get_state("C123")
        assert state_after_second is not None
        assert state_after_second["resume_cursor"] is None
        assert state_after_second["resume_min_ts"] is None
        assert state_after_second["max_processed_ts"] == pytest.approx(
            float(messages[-1]["ts"])
        )
        assert len(repository.saved) == 8
    finally:
        repository.close()
