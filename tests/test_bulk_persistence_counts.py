"""Tests for bulk persistence helpers."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.adapters.bulk_persistence import (
    DatabaseBackend,
    EventDTO,
    RelationDTO,
    upsert_event_relations_bulk,
    upsert_events_bulk,
)


class CountingCursor(sqlite3.Cursor):
    """Cursor that tracks batch executions for insert statements."""

    def executemany(self, sql: str, seq_of_parameters: list[tuple[object, ...]]):  # type: ignore[override]
        normalized = sql.strip().upper()
        if normalized.startswith("INSERT OR REPLACE INTO EVENTS"):
            self.connection.event_batches += 1  # type: ignore[attr-defined]
        elif normalized.startswith("INSERT OR REPLACE INTO EVENT_RELATIONS"):
            self.connection.relation_batches += 1  # type: ignore[attr-defined]
        return super().executemany(sql, seq_of_parameters)


class CountingConnection(sqlite3.Connection):
    """SQLite connection that exposes batch execution counters."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.event_batches = 0
        self.relation_batches = 0

    def cursor(self, factory: type[sqlite3.Cursor] = CountingCursor):  # type: ignore[override]
        return super().cursor(factory)


def _create_event_values(index: int) -> tuple[object, ...]:
    """Build tuple matching EventDTO value order."""

    now = datetime.now(tz=UTC).isoformat()
    event_id = str(uuid4())
    message_id = f"msg-{index}"
    source_channels = "[]"
    source_id = "slack"
    action = "update"
    object_id = f"OBJ-{index}"
    object_name_raw = f"Object {index}"
    qualifiers = "[]"
    stroke = None
    anchor = None
    category = "product"
    status = "new"
    change_type = "feature"
    environment = "production"
    severity = None
    summary = f"Summary {index}"
    why_it_matters = None
    links = "[]"
    anchors_json = "[]"
    impact_area = "[]"
    impact_type = "[]"
    confidence = 0.8
    importance = 50
    cluster_key = f"cluster-{index % 10}"
    dedup_key = f"dedup-{index}"

    return (
        event_id,
        message_id,
        source_channels,
        now,
        source_id,
        action,
        object_id,
        object_name_raw,
        qualifiers,
        stroke,
        anchor,
        category,
        status,
        change_type,
        environment,
        severity,
        now,
        now,
        now,
        now,
        "detected",
        0.7,
        summary,
        why_it_matters,
        links,
        anchors_json,
        impact_area,
        impact_type,
        confidence,
        importance,
        cluster_key,
        dedup_key,
    )


def _create_relation_values(source_id: str, index: int) -> tuple[object, ...]:
    """Build tuple matching RelationDTO value order."""

    created_at = datetime.now(tz=UTC).isoformat()
    return (
        source_id,
        "related",
        f"target-{index}",
        created_at,
    )


@pytest.mark.parametrize("record_count", [1200])
def test_bulk_upsert_statement_counts(record_count: int) -> None:
    """Bulk upserts should use limited statements and persist data."""

    conn = sqlite3.connect(":memory:", factory=CountingConnection)
    conn.execute(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            message_id TEXT,
            source_channels TEXT,
            extracted_at TEXT,
            source_id TEXT,
            action TEXT,
            object_id TEXT,
            object_name_raw TEXT,
            qualifiers TEXT,
            stroke TEXT,
            anchor TEXT,
            category TEXT,
            status TEXT,
            change_type TEXT,
            environment TEXT,
            severity TEXT,
            planned_start TEXT,
            planned_end TEXT,
            actual_start TEXT,
            actual_end TEXT,
            time_source TEXT,
            time_confidence REAL,
            summary TEXT,
            why_it_matters TEXT,
            links TEXT,
            anchors TEXT,
            impact_area TEXT,
            impact_type TEXT,
            confidence REAL,
            importance INTEGER,
            cluster_key TEXT,
            dedup_key TEXT UNIQUE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE event_relations (
            source_event_id TEXT,
            relation_type TEXT,
            target_event_id TEXT,
            created_at TEXT
        )
        """
    )

    event_records = [
        EventDTO(
            backend=DatabaseBackend.SQLITE,
            connection=conn,
            values=_create_event_values(i),
        )
        for i in range(record_count)
    ]
    upsert_events_bulk(event_records, chunk=500)

    relation_records = [
        RelationDTO(
            backend=DatabaseBackend.SQLITE,
            connection=conn,
            values=_create_relation_values(event_records[i].values[0], i),
        )
        for i in range(record_count)
    ]
    upsert_event_relations_bulk(relation_records, chunk=500)

    expected_max_batches = math.ceil(record_count / 500)
    assert conn.event_batches <= expected_max_batches
    assert conn.relation_batches <= expected_max_batches

    event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    relation_count = conn.execute("SELECT COUNT(*) FROM event_relations").fetchone()[0]
    assert event_count == record_count
    assert relation_count == record_count

    conn.close()
