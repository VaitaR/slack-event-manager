"""PostgreSQL repository adapter for production deployment.

Implements RepositoryProtocol with PostgreSQL backend using psycopg2.
Uses connection pooling for efficient resource management.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg2
import psycopg2.extras
import psycopg2.pool
import pytz
from psycopg2.extensions import connection as Connection

from src.adapters.query_builders import (
    CandidateQueryCriteria,
    EventQueryCriteria,
)
from src.domain.exceptions import RepositoryError
from src.domain.models import (
    CandidateStatus,
    Event,
    EventCandidate,
    EventCategory,
    LLMCallMetadata,
    ScoringFeatures,
    SlackMessage,
)


class PostgresRepository:
    """PostgreSQL-based repository for production deployment."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        min_connections: int = 1,
        max_connections: int = 10,
    ) -> None:
        """Initialize repository with connection pooling.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

        # Create connection pool
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                min_connections,
                max_connections,
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
            )
            print(
                f"🔧 PostgreSQL connection pool created: {user}@{host}:{port}/{database}"
            )
        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to create connection pool: {e}")

    @contextmanager
    def _get_connection(self) -> Iterator[Connection]:
        """Get database connection from pool.

        Yields:
            PostgreSQL connection

        Raises:
            RepositoryError: On connection errors
        """
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise RepositoryError(f"Database connection error: {e}")
        finally:
            if conn:
                self.pool.putconn(conn)

    def save_messages(self, messages: list[SlackMessage]) -> int:
        """Save messages to storage (idempotent upsert).

        Args:
            messages: List of Slack messages

        Returns:
            Number of messages saved

        Raises:
            RepositoryError: On storage errors
        """
        if not messages:
            return 0

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    for msg in messages:
                        cursor.execute(
                            """
                            INSERT INTO raw_slack_messages (
                                message_id, channel, ts, ts_dt, "user", user_real_name, user_display_name,
                                user_email, user_profile_image, is_bot, subtype,
                                text, blocks_text, text_norm, links_raw, links_norm,
                                anchors, attachments_count, files_count, reactions, total_reactions,
                                reply_count, permalink, edited_ts, edited_user, ingested_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (message_id) DO UPDATE SET
                                channel = EXCLUDED.channel,
                                ts = EXCLUDED.ts,
                                ts_dt = EXCLUDED.ts_dt,
                                "user" = EXCLUDED."user",
                                user_real_name = EXCLUDED.user_real_name,
                                user_display_name = EXCLUDED.user_display_name,
                                user_email = EXCLUDED.user_email,
                                user_profile_image = EXCLUDED.user_profile_image,
                                is_bot = EXCLUDED.is_bot,
                                subtype = EXCLUDED.subtype,
                                text = EXCLUDED.text,
                                blocks_text = EXCLUDED.blocks_text,
                                text_norm = EXCLUDED.text_norm,
                                links_raw = EXCLUDED.links_raw,
                                links_norm = EXCLUDED.links_norm,
                                anchors = EXCLUDED.anchors,
                                attachments_count = EXCLUDED.attachments_count,
                                files_count = EXCLUDED.files_count,
                                reactions = EXCLUDED.reactions,
                                total_reactions = EXCLUDED.total_reactions,
                                reply_count = EXCLUDED.reply_count,
                                permalink = EXCLUDED.permalink,
                                edited_ts = EXCLUDED.edited_ts,
                                edited_user = EXCLUDED.edited_user,
                                ingested_at = EXCLUDED.ingested_at
                            """,
                            (
                                msg.message_id,
                                msg.channel,
                                msg.ts,
                                msg.ts_dt,
                                msg.user,
                                msg.user_real_name,
                                msg.user_display_name,
                                msg.user_email,
                                msg.user_profile_image,
                                msg.is_bot,
                                msg.subtype,
                                msg.text,
                                msg.blocks_text,
                                msg.text_norm,
                                json.dumps(msg.links_raw),
                                json.dumps(msg.links_norm),
                                json.dumps(msg.anchors),
                                msg.attachments_count,
                                msg.files_count,
                                json.dumps(msg.reactions),
                                msg.total_reactions,
                                msg.reply_count,
                                msg.permalink,
                                msg.edited_ts,
                                msg.edited_user,
                                msg.ingested_at,
                            ),
                        )

                    conn.commit()
                    return len(messages)

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to save messages: {e}")

    def get_watermark(self, channel: str) -> str | None:
        """Get committed watermark timestamp for channel.

        Args:
            channel: Channel ID

        Returns:
            Last committed timestamp or None
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    cursor.execute(
                        "SELECT committed_ts FROM channel_watermarks WHERE channel = %s",
                        (channel,),
                    )
                    row = cursor.fetchone()
                    return row["committed_ts"] if row else None

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to get watermark: {e}")

    def update_watermark(self, channel: str, ts: str) -> None:
        """Update committed watermark for channel.

        Args:
            channel: Channel ID
            ts: New watermark timestamp
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO channel_watermarks (channel, committed_ts, processing_ts)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (channel) DO UPDATE SET
                            committed_ts = EXCLUDED.committed_ts,
                            processing_ts = EXCLUDED.processing_ts
                        """,
                        (channel, ts, datetime.utcnow()),
                    )
                    conn.commit()

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to update watermark: {e}")

    def get_new_messages_for_candidates(self) -> list[SlackMessage]:
        """Get messages not yet in candidates table.

        Returns:
            List of messages to process
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    cursor.execute(
                        """
                        SELECT m.* FROM raw_slack_messages m
                        LEFT JOIN event_candidates c ON m.message_id = c.message_id
                        WHERE c.message_id IS NULL
                        ORDER BY m.ts_dt DESC
                    """
                    )
                    rows = cursor.fetchall()
                    return [self._row_to_message(row) for row in rows]

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to get new messages: {e}")

    def save_candidates(self, candidates: list[EventCandidate]) -> int:
        """Save event candidates (idempotent).

        Args:
            candidates: List of candidates

        Returns:
            Number saved
        """
        if not candidates:
            return 0

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    for candidate in candidates:
                        cursor.execute(
                            """
                            INSERT INTO event_candidates (
                                message_id, channel, ts_dt, text_norm, links_norm,
                                anchors, score, status, features_json
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (message_id) DO UPDATE SET
                                channel = EXCLUDED.channel,
                                ts_dt = EXCLUDED.ts_dt,
                                text_norm = EXCLUDED.text_norm,
                                links_norm = EXCLUDED.links_norm,
                                anchors = EXCLUDED.anchors,
                                score = EXCLUDED.score,
                                status = EXCLUDED.status,
                                features_json = EXCLUDED.features_json
                            """,
                            (
                                candidate.message_id,
                                candidate.channel,
                                candidate.ts_dt,
                                candidate.text_norm,
                                json.dumps(candidate.links_norm),
                                json.dumps(candidate.anchors),
                                candidate.score,
                                candidate.status.value,
                                candidate.features.model_dump_json(),
                            ),
                        )

                    conn.commit()
                    return len(candidates)

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to save candidates: {e}")

    def get_candidates_for_extraction(
        self, batch_size: int | None = 50, min_score: float | None = None
    ) -> list[EventCandidate]:
        """Get candidates ready for LLM extraction.

        Args:
            batch_size: Maximum candidates to return (None = no limit)
            min_score: Minimum score filter

        Returns:
            List of candidates ordered by score DESC
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    query = """
                        SELECT * FROM event_candidates
                        WHERE status = 'new'
                    """
                    params: list[Any] = []

                    if min_score is not None:
                        query += " AND score >= %s"
                        params.append(min_score)

                    query += " ORDER BY score DESC"

                    if batch_size is not None:
                        query += " LIMIT %s"
                        params.append(batch_size)

                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    return [self._row_to_candidate(row) for row in rows]

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to get candidates: {e}")

    def update_candidate_status(self, message_id: str, status: str) -> None:
        """Update candidate processing status.

        Args:
            message_id: Message ID
            status: New status
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE event_candidates SET status = %s WHERE message_id = %s",
                        (status, message_id),
                    )
                    conn.commit()

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to update candidate status: {e}")

    def save_events(self, events: list[Event]) -> int:
        """Save events with versioning (upsert by dedup_key).

        Args:
            events: List of events

        Returns:
            Number saved
        """
        if not events:
            return 0

        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    for event in events:
                        # Check if dedup_key exists
                        cursor.execute(
                            "SELECT version FROM events WHERE dedup_key = %s",
                            (event.dedup_key,),
                        )
                        existing = cursor.fetchone()

                        if existing:
                            # Update with incremented version
                            cursor.execute(
                                """
                                UPDATE events SET
                                    version = %s,
                                    title = %s,
                                    summary = %s,
                                    category = %s,
                                    event_date = %s,
                                    event_end = %s,
                                    impact_area = %s,
                                    tags = %s,
                                    links = %s,
                                    anchors = %s,
                                    confidence = %s,
                                    source_channels = %s
                                WHERE dedup_key = %s
                                """,
                                (
                                    event.version,
                                    event.title,
                                    event.summary,
                                    event.category.value,
                                    event.event_date,
                                    event.event_end,
                                    json.dumps(event.impact_area),
                                    json.dumps(event.tags),
                                    json.dumps(event.links),
                                    json.dumps(event.anchors),
                                    event.confidence,
                                    json.dumps(event.source_channels),
                                    event.dedup_key,
                                ),
                            )
                        else:
                            # Insert new
                            cursor.execute(
                                """
                                INSERT INTO events (
                                    event_id, version, message_id, source_msg_event_idx,
                                    dedup_key, event_date, event_end, category, title,
                                    summary, impact_area, tags, links, anchors,
                                    confidence, source_channels, ingested_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    str(event.event_id),
                                    event.version,
                                    event.message_id,
                                    event.source_msg_event_idx,
                                    event.dedup_key,
                                    event.event_date,
                                    event.event_end,
                                    event.category.value,
                                    event.title,
                                    event.summary,
                                    json.dumps(event.impact_area),
                                    json.dumps(event.tags),
                                    json.dumps(event.links),
                                    json.dumps(event.anchors),
                                    event.confidence,
                                    json.dumps(event.source_channels),
                                    event.ingested_at,
                                ),
                            )

                    conn.commit()
                    return len(events)

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to save events: {e}")

    def get_events_in_window(self, start_dt: datetime, end_dt: datetime) -> list[Event]:
        """Get events within date window.

        Args:
            start_dt: Start datetime (UTC)
            end_dt: End datetime (UTC)

        Returns:
            List of events
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    cursor.execute(
                        """
                        SELECT * FROM events
                        WHERE event_date >= %s AND event_date <= %s
                        ORDER BY event_date ASC
                        """,
                        (start_dt, end_dt),
                    )
                    rows = cursor.fetchall()
                    return [self._row_to_event(row) for row in rows]

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to get events: {e}")

    def get_events_in_window_filtered(
        self,
        start_dt: datetime,
        end_dt: datetime,
        min_confidence: float = 0.0,
        max_events: int | None = None,
    ) -> list[Event]:
        """Get events within date window with filtering.

        Args:
            start_dt: Start datetime (UTC)
            end_dt: End datetime (UTC)
            min_confidence: Minimum confidence score (0.0-1.0)
            max_events: Maximum number of events to return (None = unlimited)

        Returns:
            List of filtered events
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    query = """
                        SELECT * FROM events
                        WHERE event_date >= %s AND event_date <= %s
                        AND confidence >= %s
                        ORDER BY event_date ASC
                    """
                    params: list[Any] = [start_dt, end_dt, min_confidence]

                    if max_events is not None:
                        query += " LIMIT %s"
                        params.append(max_events)

                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    return [self._row_to_event(row) for row in rows]

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to get filtered events: {e}")

    def query_events(self, criteria: EventQueryCriteria) -> list[Event]:
        """Query events using criteria builder.

        Args:
            criteria: Query criteria

        Returns:
            List of matching events
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    # Build query parts
                    where_clause, where_params = criteria.to_where_clause()
                    order_clause = criteria.to_order_clause()
                    limit_clause, limit_params = criteria.to_limit_clause()

                    # Combine into full query
                    query = f"""
                        SELECT * FROM events
                        WHERE {where_clause}
                        ORDER BY {order_clause}
                        {limit_clause}
                    """

                    # Execute with all parameters
                    all_params = where_params + limit_params
                    cursor.execute(query, all_params)
                    rows = cursor.fetchall()
                    return [self._row_to_event(row) for row in rows]

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to query events: {e}")

    def query_candidates(
        self, criteria: CandidateQueryCriteria
    ) -> list[EventCandidate]:
        """Query event candidates using criteria builder.

        Args:
            criteria: Query criteria

        Returns:
            List of matching candidates
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    # Build query parts
                    where_clause, where_params = criteria.to_where_clause()
                    order_clause = criteria.to_order_clause()
                    limit_clause, limit_params = criteria.to_limit_clause()

                    # Combine into full query
                    query = f"""
                        SELECT * FROM event_candidates
                        WHERE {where_clause}
                        ORDER BY {order_clause}
                        {limit_clause}
                    """

                    # Execute with all parameters
                    all_params = where_params + limit_params
                    cursor.execute(query, all_params)
                    rows = cursor.fetchall()
                    return [self._row_to_candidate(row) for row in rows]

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to query candidates: {e}")

    def save_llm_call(self, metadata: LLMCallMetadata) -> None:
        """Save LLM call metadata.

        Args:
            metadata: Call metadata
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO llm_calls (
                            message_id, prompt_hash, model, tokens_in, tokens_out,
                            cost_usd, latency_ms, cached, ts
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            metadata.message_id,
                            metadata.prompt_hash,
                            metadata.model,
                            metadata.tokens_in,
                            metadata.tokens_out,
                            metadata.cost_usd,
                            metadata.latency_ms,
                            metadata.cached,
                            metadata.ts,
                        ),
                    )
                    conn.commit()

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to save LLM call: {e}")

    def get_daily_llm_cost(self, date: datetime) -> float:
        """Get total LLM cost for a day.

        Args:
            date: Date to check

        Returns:
            Total cost in USD
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    date_str = date.strftime("%Y-%m-%d")
                    cursor.execute(
                        """
                        SELECT SUM(cost_usd) as total FROM llm_calls
                        WHERE DATE(ts) = %s
                        """,
                        (date_str,),
                    )
                    row = cursor.fetchone()
                    return float(row["total"]) if row and row["total"] else 0.0

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to get daily cost: {e}")

    def get_cached_llm_response(self, prompt_hash: str) -> str | None:
        """Get cached LLM response by prompt hash.

        Args:
            prompt_hash: SHA256 hash of prompt

        Returns:
            Cached JSON response or None
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    cursor.execute(
                        """
                        SELECT response_json FROM llm_calls
                        WHERE prompt_hash = %s AND response_json IS NOT NULL
                        ORDER BY ts DESC
                        LIMIT 1
                        """,
                        (prompt_hash,),
                    )
                    row = cursor.fetchone()
                    return row["response_json"] if row else None

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to get cached response: {e}")

    def save_llm_response(self, prompt_hash: str, response_json: str) -> None:
        """Save LLM response for caching.

        Args:
            prompt_hash: Prompt hash
            response_json: JSON response string
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE llm_calls
                        SET response_json = %s
                        WHERE prompt_hash = %s
                        AND ts = (
                            SELECT MAX(ts) FROM llm_calls WHERE prompt_hash = %s
                        )
                        """,
                        (response_json, prompt_hash, prompt_hash),
                    )
                    conn.commit()

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to save LLM response: {e}")

    def get_last_processed_ts(self, channel_id: str) -> float | None:
        """Get last processed timestamp for a channel.

        Args:
            channel_id: Slack channel ID

        Returns:
            Last processed timestamp (epoch seconds) or None if first run
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    cursor.execute(
                        """
                        SELECT last_ts FROM ingestion_state
                        WHERE channel_id = %s
                        """,
                        (channel_id,),
                    )
                    row = cursor.fetchone()
                    return float(row["last_ts"]) if row else None

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to get last processed ts: {e}")

    def update_last_processed_ts(self, channel_id: str, last_ts: float) -> None:
        """Update last processed timestamp for a channel.

        Args:
            channel_id: Slack channel ID
            last_ts: Last processed timestamp (epoch seconds)
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO ingestion_state (channel_id, last_ts)
                        VALUES (%s, %s)
                        ON CONFLICT (channel_id) DO UPDATE SET
                            last_ts = EXCLUDED.last_ts
                        """,
                        (channel_id, last_ts),
                    )
                    conn.commit()

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to update last processed ts: {e}")

    def _row_to_message(self, row: dict[str, Any]) -> SlackMessage:
        """Convert database row to SlackMessage.

        Args:
            row: Database row as dictionary

        Returns:
            SlackMessage instance
        """
        return SlackMessage(
            message_id=row["message_id"],
            channel=row["channel"],
            ts=row["ts"],
            ts_dt=row["ts_dt"].replace(tzinfo=pytz.UTC)
            if row["ts_dt"].tzinfo is None
            else row["ts_dt"],
            user=row["user"],
            user_real_name=row.get("user_real_name"),
            user_display_name=row.get("user_display_name"),
            user_email=row.get("user_email"),
            user_profile_image=row.get("user_profile_image"),
            is_bot=bool(row["is_bot"]),
            subtype=row["subtype"],
            text=row["text"] or "",
            blocks_text=row["blocks_text"] or "",
            text_norm=row["text_norm"] or "",
            links_raw=row["links_raw"] or [],
            links_norm=row["links_norm"] or [],
            anchors=row["anchors"] or [],
            attachments_count=row.get("attachments_count", 0),
            files_count=row.get("files_count", 0),
            reactions=row["reactions"] or {},
            total_reactions=row.get("total_reactions", 0),
            reply_count=row["reply_count"] or 0,
            permalink=row.get("permalink"),
            edited_ts=row.get("edited_ts"),
            edited_user=row.get("edited_user"),
            ingested_at=row["ingested_at"].replace(tzinfo=pytz.UTC)
            if row["ingested_at"].tzinfo is None
            else row["ingested_at"],
        )

    def _row_to_candidate(self, row: dict[str, Any]) -> EventCandidate:
        """Convert database row to EventCandidate.

        Args:
            row: Database row as dictionary

        Returns:
            EventCandidate instance
        """
        return EventCandidate(
            message_id=row["message_id"],
            channel=row["channel"],
            ts_dt=row["ts_dt"].replace(tzinfo=pytz.UTC)
            if row["ts_dt"].tzinfo is None
            else row["ts_dt"],
            text_norm=row["text_norm"] or "",
            links_norm=row["links_norm"] or [],
            anchors=row["anchors"] or [],
            score=float(row["score"]),
            status=CandidateStatus(row["status"]),
            features=ScoringFeatures.model_validate(row["features_json"]),
        )

    def _row_to_event(self, row: dict[str, Any]) -> Event:
        """Convert database row to Event.

        Args:
            row: Database row as dictionary

        Returns:
            Event instance
        """
        return Event(
            event_id=UUID(row["event_id"]),
            version=int(row["version"]),
            message_id=row["message_id"],
            source_msg_event_idx=int(row["source_msg_event_idx"]),
            dedup_key=row["dedup_key"],
            event_date=row["event_date"].replace(tzinfo=pytz.UTC)
            if row["event_date"].tzinfo is None
            else row["event_date"],
            event_end=(
                row["event_end"].replace(tzinfo=pytz.UTC)
                if row["event_end"] and row["event_end"].tzinfo is None
                else row["event_end"]
            )
            if row["event_end"]
            else None,
            category=EventCategory(row["category"]),
            title=row["title"],
            summary=row["summary"],
            impact_area=row["impact_area"] or [],
            tags=row["tags"] or [],
            links=row["links"] or [],
            anchors=row["anchors"] or [],
            confidence=float(row["confidence"]),
            source_channels=row["source_channels"] or [],
            ingested_at=row["ingested_at"].replace(tzinfo=pytz.UTC)
            if row["ingested_at"].tzinfo is None
            else row["ingested_at"],
        )

    def close(self) -> None:
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            print("🔒 PostgreSQL connection pool closed")
