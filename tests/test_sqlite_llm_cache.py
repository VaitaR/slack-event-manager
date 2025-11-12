"""Tests for SQLite LLM cache persistence."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from src.adapters.sqlite_repository import SQLiteRepository
from src.domain.models import LLMCallMetadata


def test_save_llm_response_updates_latest_call(tmp_path) -> None:
    """Ensure the cached response is stored on the latest call row."""

    db_path = tmp_path / "llm_cache.db"
    repository = SQLiteRepository(str(db_path))

    prompt_hash = "prompt-hash"
    older_metadata = LLMCallMetadata(
        message_id="message-1",
        prompt_hash=prompt_hash,
        model="gpt-5-nano",
        tokens_in=10,
        tokens_out=15,
        cost_usd=0.001,
        latency_ms=1200,
        cached=False,
        ts=datetime.now(tz=UTC) - timedelta(minutes=5),
    )
    newer_metadata = LLMCallMetadata(
        message_id="message-2",
        prompt_hash=prompt_hash,
        model="gpt-5-nano",
        tokens_in=12,
        tokens_out=18,
        cost_usd=0.0015,
        latency_ms=800,
        cached=False,
        ts=datetime.now(tz=UTC),
    )

    repository.save_llm_call(older_metadata)
    repository.save_llm_call(newer_metadata)

    cached_payload = '{"events": []}'
    repository.save_llm_response(prompt_hash, cached_payload)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, response_json, ts
            FROM llm_calls
            WHERE prompt_hash = ?
            ORDER BY ts ASC
            """,
            (prompt_hash,),
        ).fetchall()

    assert len(rows) == 2
    assert rows[0]["response_json"] is None
    assert rows[1]["response_json"] == cached_payload
