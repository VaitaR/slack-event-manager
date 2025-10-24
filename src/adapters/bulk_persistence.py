"""Bulk persistence helpers for batched event and relation upserts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import islice
from typing import Any

from psycopg2 import Error as PsycopgError
from psycopg2 import extensions
from psycopg2.extras import execute_batch

from src.domain.models import Event, EventRelation


class DatabaseBackend(StrEnum):
    """Supported database backends for bulk persistence."""

    POSTGRES = "postgres"
    SQLITE = "sqlite"


@dataclass(slots=True)
class EventDTO:
    """Serializable event payload for bulk upserts."""

    backend: DatabaseBackend
    connection: extensions.connection | sqlite3.Connection
    values: tuple[Any, ...]

    @classmethod
    def from_event(
        cls,
        event: Event,
        *,
        backend: DatabaseBackend,
        connection: extensions.connection | sqlite3.Connection,
    ) -> EventDTO:
        """Build DTO from domain event."""

        def _serialize_datetime(dt: datetime | None) -> datetime | str | None:
            if dt is None:
                return None
            if backend is DatabaseBackend.SQLITE:
                return dt.isoformat()
            return dt

        values = (
            str(event.event_id),
            event.message_id,
            json.dumps(event.source_channels),
            _serialize_datetime(event.extracted_at),
            event.source_id.value,
            event.action.value,
            event.object_id,
            event.object_name_raw,
            json.dumps(event.qualifiers),
            event.stroke,
            event.anchor,
            event.category.value,
            event.status.value,
            event.change_type.value,
            event.environment.value,
            event.severity.value if event.severity else None,
            _serialize_datetime(event.planned_start),
            _serialize_datetime(event.planned_end),
            _serialize_datetime(event.actual_start),
            _serialize_datetime(event.actual_end),
            event.time_source.value,
            event.time_confidence,
            event.summary,
            event.why_it_matters,
            json.dumps(event.links),
            json.dumps(event.anchors),
            json.dumps(event.impact_area),
            json.dumps(event.impact_type),
            event.confidence,
            event.importance,
            event.cluster_key,
            event.dedup_key,
        )
        return cls(backend=backend, connection=connection, values=values)


@dataclass(slots=True)
class RelationDTO:
    """Serializable relation payload for bulk upserts."""

    backend: DatabaseBackend
    connection: extensions.connection | sqlite3.Connection
    values: tuple[Any, ...]

    @classmethod
    def from_relation(
        cls,
        source_event_id: str,
        relation: EventRelation,
        *,
        backend: DatabaseBackend,
        connection: extensions.connection | sqlite3.Connection,
    ) -> RelationDTO:
        """Build DTO from event relation."""

        created_at = datetime.now(tz=UTC)
        if backend is DatabaseBackend.SQLITE:
            created_at_value: datetime | str = created_at.isoformat()
        else:
            created_at_value = created_at
        values = (
            source_event_id,
            relation.relation_type.value,
            str(relation.target_event_id),
            created_at_value,
        )
        return cls(backend=backend, connection=connection, values=values)


def upsert_events_bulk(records: list[EventDTO], *, chunk: int = 500) -> None:
    """Persist events using batched upserts."""

    if not records:
        return
    if chunk <= 0:
        raise ValueError("chunk must be positive")

    backend = records[0].backend
    connection = records[0].connection
    _validate_records(records, backend, connection)

    serialized = [record.values for record in records]

    if backend is DatabaseBackend.POSTGRES:
        _postgres_events_batch(connection, serialized, chunk)
    elif backend is DatabaseBackend.SQLITE:
        _sqlite_events_batch(connection, serialized, chunk)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported backend: {backend}")


def upsert_event_relations_bulk(
    records: list[RelationDTO], *, chunk: int = 500
) -> None:
    """Persist event relations using batched upserts."""

    if not records:
        return
    if chunk <= 0:
        raise ValueError("chunk must be positive")

    backend = records[0].backend
    connection = records[0].connection
    _validate_records(records, backend, connection)

    serialized = [record.values for record in records]

    if backend is DatabaseBackend.POSTGRES:
        _postgres_relations_batch(connection, serialized, chunk)
    elif backend is DatabaseBackend.SQLITE:
        _sqlite_relations_batch(connection, serialized, chunk)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported backend: {backend}")


def _validate_records(
    records: Sequence[EventDTO | RelationDTO],
    backend: DatabaseBackend,
    connection: extensions.connection | sqlite3.Connection,
) -> None:
    """Ensure all DTOs share the same backend and connection."""

    for record in records:
        if record.backend is not backend:
            raise ValueError("All records must share the same backend")
        if record.connection is not connection:
            raise ValueError("All records must share the same connection instance")


def _postgres_events_batch(
    connection: extensions.connection,
    values: list[tuple[Any, ...]],
    chunk: int,
) -> None:
    """Execute batched event upserts for PostgreSQL."""

    insert_sql = """
        INSERT INTO events (
            event_id, message_id, source_channels, extracted_at, source_id,
            action, object_id, object_name_raw, qualifiers, stroke, anchor,
            category, status, change_type, environment, severity,
            planned_start, planned_end, actual_start, actual_end,
            time_source, time_confidence,
            summary, why_it_matters, links, anchors, impact_area, impact_type,
            confidence, importance, cluster_key, dedup_key
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s
        )
        ON CONFLICT (dedup_key) DO UPDATE SET
            message_id = EXCLUDED.message_id,
            source_channels = EXCLUDED.source_channels,
            extracted_at = EXCLUDED.extracted_at,
            source_id = EXCLUDED.source_id,
            action = EXCLUDED.action,
            object_id = EXCLUDED.object_id,
            object_name_raw = EXCLUDED.object_name_raw,
            qualifiers = EXCLUDED.qualifiers,
            stroke = EXCLUDED.stroke,
            anchor = EXCLUDED.anchor,
            category = EXCLUDED.category,
            status = EXCLUDED.status,
            change_type = EXCLUDED.change_type,
            environment = EXCLUDED.environment,
            severity = EXCLUDED.severity,
            planned_start = EXCLUDED.planned_start,
            planned_end = EXCLUDED.planned_end,
            actual_start = EXCLUDED.actual_start,
            actual_end = EXCLUDED.actual_end,
            time_source = EXCLUDED.time_source,
            time_confidence = EXCLUDED.time_confidence,
            summary = EXCLUDED.summary,
            why_it_matters = EXCLUDED.why_it_matters,
            links = EXCLUDED.links,
            anchors = EXCLUDED.anchors,
            impact_area = EXCLUDED.impact_area,
            impact_type = EXCLUDED.impact_type,
            confidence = EXCLUDED.confidence,
            importance = EXCLUDED.importance,
            cluster_key = EXCLUDED.cluster_key
    """

    try:
        for batch in _batched(values, chunk):
            with connection.cursor() as cursor:
                execute_batch(cursor, insert_sql, batch)
        connection.commit()
    except PsycopgError:
        connection.rollback()
        raise


def _sqlite_events_batch(
    connection: sqlite3.Connection,
    values: list[tuple[Any, ...]],
    chunk: int,
) -> None:
    """Execute batched event upserts for SQLite."""

    insert_sql = """
        INSERT OR REPLACE INTO events (
            event_id, message_id, source_channels, extracted_at, source_id,
            action, object_id, object_name_raw, qualifiers, stroke, anchor,
            category, status, change_type, environment, severity,
            planned_start, planned_end, actual_start, actual_end,
            time_source, time_confidence,
            summary, why_it_matters, links, anchors, impact_area, impact_type,
            confidence, importance, cluster_key, dedup_key
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?
        )
    """

    cursor = connection.cursor()
    try:
        for batch in _batched(values, chunk):
            cursor.executemany(insert_sql, batch)
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise
    finally:
        cursor.close()


def _postgres_relations_batch(
    connection: extensions.connection,
    values: list[tuple[Any, ...]],
    chunk: int,
) -> None:
    """Execute batched relation upserts for PostgreSQL."""

    insert_sql = """
        INSERT INTO event_relations (
            source_event_id, relation_type, target_event_id, created_at
        ) VALUES (%s, %s, %s, %s)
        ON CONFLICT (source_event_id, relation_type, target_event_id) DO UPDATE SET
            created_at = EXCLUDED.created_at
    """

    try:
        for batch in _batched(values, chunk):
            with connection.cursor() as cursor:
                execute_batch(cursor, insert_sql, batch)
        connection.commit()
    except PsycopgError:
        connection.rollback()
        raise


def _sqlite_relations_batch(
    connection: sqlite3.Connection,
    values: list[tuple[Any, ...]],
    chunk: int,
) -> None:
    """Execute batched relation upserts for SQLite."""

    insert_sql = """
        INSERT OR REPLACE INTO event_relations (
            source_event_id, relation_type, target_event_id, created_at
        ) VALUES (?, ?, ?, ?)
    """

    cursor = connection.cursor()
    try:
        for batch in _batched(values, chunk):
            cursor.executemany(insert_sql, batch)
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise
    finally:
        cursor.close()


def _batched(
    values: Sequence[tuple[Any, ...]], chunk: int
) -> Iterator[Sequence[tuple[Any, ...]]]:
    """Yield fixed-size batches from sequence."""

    iterator = iter(values)
    while True:
        batch = list(islice(iterator, chunk))
        if not batch:
            return
        yield batch


__all__ = [
    "DatabaseBackend",
    "EventDTO",
    "RelationDTO",
    "upsert_events_bulk",
    "upsert_event_relations_bulk",
]
