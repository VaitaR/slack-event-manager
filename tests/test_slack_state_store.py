import sqlite3
from contextlib import contextmanager

import pytest

from src.adapters.slack_state_store import SlackStateStore

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS slack_ingestion_state (
  channel_id TEXT PRIMARY KEY,
  max_processed_ts REAL NOT NULL DEFAULT 0,
  resume_cursor TEXT,
  resume_min_ts REAL,
  updated_at TEXT
);
"""


def _create_store() -> tuple[SlackStateStore, sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(CREATE_SQL)
    connection.commit()

    @contextmanager
    def _get_conn() -> sqlite3.Connection:
        yield connection

    return SlackStateStore(_get_conn), connection


def test_slack_state_store_roundtrip() -> None:
    store, connection = _create_store()

    try:
        state = store.get("C123")
        assert state["max_processed_ts"] == pytest.approx(0.0)
        assert state["resume_cursor"] is None
        assert state["resume_min_ts"] is None

        store.upsert(
            "C123",
            max_processed_ts=123.45,
            resume_cursor="cursor-1",
            resume_min_ts=100.0,
        )

        state_after_first = store.get("C123")
        assert state_after_first["max_processed_ts"] == pytest.approx(123.45)
        assert state_after_first["resume_cursor"] == "cursor-1"
        assert state_after_first["resume_min_ts"] == pytest.approx(100.0)

        store.upsert(
            "C123", max_processed_ts=555.0, resume_cursor=None, resume_min_ts=None
        )

        state_after_second = store.get("C123")
        assert state_after_second["max_processed_ts"] == pytest.approx(555.0)
        assert state_after_second["resume_cursor"] is None
        assert state_after_second["resume_min_ts"] is None
    finally:
        connection.close()
