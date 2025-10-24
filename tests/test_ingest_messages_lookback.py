"""Tests for Slack ingestion lookback and backfill behavior."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta
from unittest.mock import create_autospec

import pytest
import pytz

from src.adapters.slack_client import SlackClient
from src.config.settings import Settings
from src.domain.models import ChannelConfig
from src.domain.protocols import RepositoryProtocol
from src.use_cases import ingest_messages as ingest_messages_module


class _FixedDateTime(datetime):
    """Deterministic datetime replacement for testing."""

    _NOW = datetime(2025, 1, 2, 12, 0, 0)

    @classmethod
    def utcnow(cls) -> datetime:
        return cls._NOW


@pytest.fixture
def fixed_now() -> datetime:
    """Provide the frozen current time with UTC tzinfo."""

    return pytz.UTC.localize(_FixedDateTime._NOW)


@pytest.fixture
def base_settings(settings: Settings) -> Settings:
    """Return settings with a single enabled Slack channel."""

    channel = ChannelConfig(channel_id="C123", channel_name="general")
    return settings.model_copy(update={"slack_channels": [channel]})


def _make_clients() -> tuple[SlackClient, RepositoryProtocol]:
    slack_client = create_autospec(SlackClient, instance=True)
    slack_client.fetch_messages.return_value = []
    slack_client._fetch_page_with_retries.return_value = {
        "ok": True,
        "messages": [],
        "response_metadata": {},
    }
    repository = create_autospec(RepositoryProtocol, instance=True)
    repository.save_messages.return_value = 0
    return slack_client, repository


CREATE_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS slack_ingestion_state (
  channel_id TEXT PRIMARY KEY,
  max_processed_ts REAL NOT NULL DEFAULT 0,
  resume_cursor TEXT,
  resume_min_ts REAL,
  updated_at TEXT
);
"""


def _patch_state_store(
    monkeypatch: pytest.MonkeyPatch,
) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(CREATE_STATE_TABLE_SQL)
    connection.commit()

    def _factory(
        _: RepositoryProtocol,
    ) -> Callable[[], AbstractContextManager[sqlite3.Connection]]:
        @contextmanager
        def _ctx() -> sqlite3.Connection:
            yield connection

        return _ctx

    monkeypatch.setattr(
        ingest_messages_module,
        "_resolve_state_connection_factory",
        _factory,
    )
    return connection


def test_ingest_uses_explicit_lookback_when_no_state(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch, fixed_now: datetime
) -> None:
    """Ingestion should honor explicit lookback when no prior state exists."""

    slack_client, repository = _make_clients()
    repository.get_last_processed_ts.return_value = None

    monkeypatch.setattr(ingest_messages_module, "datetime", _FixedDateTime)
    captured: dict[str, object] = {}
    state_conn = _patch_state_store(monkeypatch)

    def _capture_fetch(
        slack_client: SlackClient,
        channel_id: str,
        *,
        oldest_ts: str | None,
        cursor: str | None,
        limit: int | None,
        page_size: int,
    ) -> tuple[list[dict[str, object]], str | None, bool]:
        captured["oldest_ts"] = oldest_ts
        captured["cursor"] = cursor
        captured["limit"] = limit
        captured["page_size"] = page_size
        return [], None, False

    monkeypatch.setattr(ingest_messages_module, "_fetch_slack_messages", _capture_fetch)

    try:
        result = ingest_messages_module.ingest_messages_use_case(
            slack_client=slack_client,
            repository=repository,
            settings=base_settings,
            lookback_hours=6,
        )

        assert result.messages_fetched == 0
        oldest_value = captured["oldest_ts"]
        assert isinstance(oldest_value, str)
        oldest_ts = float(oldest_value)
        expected_oldest = (fixed_now - timedelta(hours=6)).timestamp()
        assert oldest_ts == pytest.approx(expected_oldest)
    finally:
        state_conn.close()


def test_ingest_prefers_backfill_date_over_lookback(
    base_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    """Explicit backfill date should take precedence over lookback window."""

    slack_client, repository = _make_clients()
    repository.get_last_processed_ts.return_value = None

    monkeypatch.setattr(ingest_messages_module, "datetime", _FixedDateTime)
    captured: dict[str, object] = {}

    def _capture_fetch(
        slack_client: SlackClient,
        channel_id: str,
        *,
        oldest_ts: str | None,
        cursor: str | None,
        limit: int | None,
        page_size: int,
    ) -> tuple[list[dict[str, object]], str | None, bool]:
        captured["oldest_ts"] = oldest_ts
        return [], None, False

    monkeypatch.setattr(ingest_messages_module, "_fetch_slack_messages", _capture_fetch)

    backfill_from_date = fixed_now - timedelta(days=3)
    state_conn = _patch_state_store(monkeypatch)

    try:
        result = ingest_messages_module.ingest_messages_use_case(
            slack_client=slack_client,
            repository=repository,
            settings=base_settings,
            lookback_hours=6,
            backfill_from_date=backfill_from_date,
        )

        assert result.messages_fetched == 0
        oldest_value = captured["oldest_ts"]
        assert isinstance(oldest_value, str)
        oldest_ts = float(oldest_value)
        expected_oldest = (fixed_now - timedelta(days=3)).timestamp()
        assert oldest_ts == pytest.approx(expected_oldest)
    finally:
        state_conn.close()


def test_ingest_uses_state_timestamp_when_available(
    base_settings: Settings, monkeypatch: pytest.MonkeyPatch, fixed_now: datetime
) -> None:
    """Existing ingestion state should override lookback window."""

    slack_client, repository = _make_clients()
    last_ts_dt = fixed_now - timedelta(hours=2)
    repository.get_last_processed_ts.return_value = last_ts_dt.timestamp()

    monkeypatch.setattr(ingest_messages_module, "datetime", _FixedDateTime)
    captured: dict[str, object] = {}
    state_conn = _patch_state_store(monkeypatch)

    def _capture_fetch(
        slack_client: SlackClient,
        channel_id: str,
        *,
        oldest_ts: str | None,
        cursor: str | None,
        limit: int | None,
        page_size: int,
    ) -> tuple[list[dict[str, object]], str | None, bool]:
        captured["oldest_ts"] = oldest_ts
        return [], None, False

    monkeypatch.setattr(ingest_messages_module, "_fetch_slack_messages", _capture_fetch)

    try:
        result = ingest_messages_module.ingest_messages_use_case(
            slack_client=slack_client,
            repository=repository,
            settings=base_settings,
            lookback_hours=6,
        )

        assert result.messages_fetched == 0
        oldest_value = captured["oldest_ts"]
        assert isinstance(oldest_value, str)
        oldest_ts = float(oldest_value)
        expected_oldest = last_ts_dt.timestamp()
        assert oldest_ts == pytest.approx(expected_oldest)
    finally:
        state_conn.close()
