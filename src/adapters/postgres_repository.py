"""PostgreSQL repository implementation using psycopg2.

Connection pooling for production use with configurable min/max connections.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2 import pool
from psycopg2.extensions import connection as Connection

from src.domain.exceptions import RepositoryError
from src.domain.models import Event, EventCandidate, LLMCallMetadata, SlackMessage


class PostgresRepository:
    """PostgreSQL repository with connection pooling."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        minconn: int = 1,
        maxconn: int = 10,
    ):
        """Initialize PostgreSQL connection pool.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            minconn: Minimum connections in pool
            maxconn: Maximum connections in pool

        Raises:
            RepositoryError: On connection pool creation failure
        """
        try:
            self._pool = pool.SimpleConnectionPool(
                minconn,
                maxconn,
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
            raise RepositoryError(f"Failed to create connection pool: {e}") from e

    @contextmanager
    def _get_connection(self) -> Iterator[Connection]:
        """Get connection from pool (context manager).

        Yields:
            Database connection

        Raises:
            RepositoryError: On connection errors
        """
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise RepositoryError(f"Database error: {e}") from e
        finally:
            if conn:
                self._pool.putconn(conn)

    def _row_to_message(self, row: dict[str, Any]) -> SlackMessage:
        """Convert database row to SlackMessage.

        Args:
            row: Database row as dict

        Returns:
            SlackMessage instance
        """
        return SlackMessage(
            message_id=row["message_id"],
            channel=row["channel_id"],
            user=row["user"],
            user_real_name=row.get("user_real_name"),
            user_display_name=row.get("user_display_name"),
            user_email=row.get("user_email"),
            user_profile_image=row.get("user_profile_image"),
            ts=row["ts"],
            ts_dt=row["ts_dt"],
            text=row["text_raw"],
            blocks_text=row.get("blocks_text", ""),
            text_norm=row["text_norm"],
            is_bot=row["is_bot"],
            subtype=row.get("subtype"),
            reply_count=row["reply_count"],
            reactions=row.get("reactions") or {},  # Already parsed dict from JSONB
            links_raw=row.get("links_raw") or [],  # Already parsed list from JSONB
            links_norm=row.get("links_norm") or [],  # Already parsed list from JSONB
            anchors=row.get("anchors") or [],  # Already parsed list from JSONB
            attachments_count=row.get("attachments_count", 0),
            files_count=row.get("files_count", 0),
            total_reactions=row.get("total_reactions", 0),
            permalink=row.get("permalink"),
            edited_ts=row.get("edited_ts"),
            edited_user=row.get("edited_user"),
        )

    def _row_to_candidate(self, row: dict[str, Any]) -> EventCandidate:
        """Convert database row to EventCandidate.

        Args:
            row: Database row as dict

        Returns:
            EventCandidate instance
        """
        from src.domain.models import CandidateStatus, ScoringFeatures

        return EventCandidate(
            message_id=row["message_id"],
            channel=row["channel"],
            ts_dt=row["ts_dt"],  # Already datetime from PostgreSQL
            text_norm=row["text_norm"] or "",
            links_norm=row.get("links_norm") or [],  # Already parsed list from JSONB
            anchors=row.get("anchors") or [],  # Already parsed list from JSONB
            score=row["score"] or 0.0,
            status=CandidateStatus(row["status"])
            if row["status"]
            else CandidateStatus.NEW,
            features=ScoringFeatures.model_validate(
                row["features_json"] or {}
            ),  # Already parsed dict
        )

    def _row_to_event(self, row: dict[str, Any]) -> Event:
        """Convert database row to Event.

        Args:
            row: Database row as dict

        Returns:
            Event instance
        """
        from uuid import UUID

        from src.domain.models import EventCategory

        return Event(
            event_id=UUID(row["event_id"]),
            version=row["version"],
            message_id=row["message_id"],
            source_msg_event_idx=row["source_msg_event_idx"],
            dedup_key=row["dedup_key"],
            event_date=row["event_date"],  # Already datetime from PostgreSQL
            event_end=row.get("event_end"),  # Already datetime or None
            category=EventCategory(row["category"])
            if row["category"]
            else EventCategory.UNKNOWN,
            title=row["title"] or "",
            summary=row["summary"] or "",
            impact_area=row.get("impact_area") or [],  # Already parsed list from JSONB
            tags=row.get("tags") or [],  # Already parsed list from JSONB
            links=row.get("links") or [],  # Already parsed list from JSONB
            anchors=row.get("anchors") or [],  # Already parsed list from JSONB
            confidence=row["confidence"] or 0.0,
            source_channels=row.get("source_channels")
            or [],  # Already parsed list from JSONB
            ingested_at=row["ingested_at"]
            or row["event_date"],  # Fallback to event_date
        )

    def save_messages(self, messages: list[SlackMessage]) -> int:
        """Save messages to PostgreSQL (upsert).

        Args:
            messages: List of slack messages

        Returns:
            Number of messages saved

        Raises:
            RepositoryError: On storage errors
        """
        if not messages:
            return 0

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for msg in messages:
                    cur.execute(
                        """
                        INSERT INTO raw_slack_messages (
                            message_id, channel_id, "user", user_real_name,
                            user_display_name, user_email, user_profile_image,
                            ts, ts_dt, text_raw, blocks_text, text_norm, is_bot, subtype,
                            reply_count, reactions, links_raw, links_norm, anchors,
                            attachments_count, files_count, total_reactions,
                            permalink, edited_ts, edited_user
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (message_id) DO UPDATE SET
                            text_raw = EXCLUDED.text_raw,
                            text_norm = EXCLUDED.text_norm,
                            reply_count = EXCLUDED.reply_count,
                            reactions = EXCLUDED.reactions,
                            edited_ts = EXCLUDED.edited_ts,
                            edited_user = EXCLUDED.edited_user
                        """,
                        (
                            msg.message_id,
                            msg.channel,
                            msg.user,
                            msg.user_real_name,
                            msg.user_display_name,
                            msg.user_email,
                            msg.user_profile_image,
                            msg.ts,
                            msg.ts_dt,
                            msg.text,
                            msg.blocks_text,
                            msg.text_norm,
                            msg.is_bot,
                            msg.subtype,
                            msg.reply_count,
                            json.dumps(msg.reactions),  # Convert dict to JSON
                            json.dumps(msg.links_raw),  # Convert list to JSON
                            json.dumps(msg.links_norm),  # Convert list to JSON
                            json.dumps(msg.anchors),  # Convert list to JSON
                            msg.attachments_count,
                            msg.files_count,
                            msg.total_reactions,
                            msg.permalink,
                            msg.edited_ts,
                            msg.edited_user,
                        ),
                    )
                conn.commit()
        return len(messages)

    def get_watermark(self, channel: str) -> str | None:
        """Get watermark timestamp for channel.

        Args:
            channel: Channel ID

        Returns:
            Last committed timestamp or None

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT ts FROM channel_watermarks WHERE channel_id = %s",
                    (channel,),
                )
                row = cur.fetchone()
                return row["ts"] if row else None

    def update_watermark(self, channel: str, ts: str) -> None:
        """Update watermark for channel.

        Args:
            channel: Channel ID
            ts: New watermark timestamp

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO channel_watermarks (channel_id, ts, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (channel_id) DO UPDATE SET
                        ts = EXCLUDED.ts,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (channel, ts),
                )
                conn.commit()

    def get_new_messages_for_candidates(self) -> list[SlackMessage]:
        """Get messages not yet in candidates table.

        Returns:
            List of messages to process

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT m.*
                    FROM raw_slack_messages m
                    LEFT JOIN event_candidates c ON m.message_id = c.message_id
                    WHERE c.message_id IS NULL
                    ORDER BY m.ts_dt DESC
                    """
                )
                rows = cur.fetchall()
                return [self._row_to_message(dict(row)) for row in rows]

    def save_candidates(self, candidates: list[EventCandidate]) -> int:
        """Save event candidates (upsert).

        Args:
            candidates: List of candidates

        Returns:
            Number of candidates saved

        Raises:
            RepositoryError: On storage errors
        """
        if not candidates:
            return 0

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for cand in candidates:
                    cur.execute(
                        """
                        INSERT INTO event_candidates (
                            message_id, channel, ts_dt, text_norm, links_norm,
                            anchors, score, status, features_json
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (message_id) DO UPDATE SET
                            score = EXCLUDED.score,
                            features_json = EXCLUDED.features_json,
                            status = EXCLUDED.status
                        """,
                        (
                            cand.message_id,
                            cand.channel,
                            cand.ts_dt,
                            cand.text_norm,
                            json.dumps(cand.links_norm),
                            json.dumps(cand.anchors),
                            cand.score,
                            cand.status.value,
                            json.dumps(cand.features.model_dump()),
                        ),
                    )
                conn.commit()
        return len(candidates)

    def get_candidates_for_extraction(
        self, batch_size: int | None = 50, min_score: float | None = None
    ) -> list[EventCandidate]:
        """Get candidates ready for LLM extraction.

        Args:
            batch_size: Maximum candidates to return (None = all)
            min_score: Minimum score filter

        Returns:
            List of candidates ordered by score DESC

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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

                cur.execute(query, params)
                rows = cur.fetchall()
                return [self._row_to_candidate(dict(row)) for row in rows]

    def update_candidate_status(self, message_id: str, status: str) -> None:
        """Update candidate processing status.

        Args:
            message_id: Message ID
            status: New status (llm_ok, llm_fail)

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE event_candidates SET status = %s WHERE message_id = %s",
                    (status, message_id),
                )
                conn.commit()

    def save_events(self, events: list[Event]) -> int:
        """Save events with versioning (upsert by dedup_key).

        Args:
            events: List of events

        Returns:
            Number of events saved

        Raises:
            RepositoryError: On storage errors
        """
        if not events:
            return 0

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for event in events:
                    # Check if dedup_key exists
                    cur.execute(
                        "SELECT version FROM events WHERE dedup_key = %s",
                        (event.dedup_key,),
                    )
                    existing = cur.fetchone()

                    if existing:
                        # Update with incremented version
                        cur.execute(
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
                        # Insert new event
                        cur.execute(
                            """
                            INSERT INTO events (
                                event_id, version, message_id, source_msg_event_idx,
                                dedup_key, event_date, event_end, category, title,
                                summary, impact_area, tags, links, anchors,
                                confidence, source_channels, ingested_at
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            )
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

    def get_events_in_window(self, start_dt: datetime, end_dt: datetime) -> list[Event]:
        """Get events within date window.

        Args:
            start_dt: Start datetime (UTC)
            end_dt: End datetime (UTC)

        Returns:
            List of events

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM events
                    WHERE event_date >= %s AND event_date <= %s
                    ORDER BY event_date DESC
                    """,
                    (start_dt, end_dt),
                )
                rows = cur.fetchall()
                return [self._row_to_event(dict(row)) for row in rows]

    def save_llm_call(self, metadata: LLMCallMetadata) -> None:
        """Save LLM call metadata.

        Args:
            metadata: Call metadata

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Note: LLMCallMetadata uses tokens_in/out, not prompt_tokens/completion_tokens
                # We map to database schema for consistency with SQLite
                cur.execute(
                    """
                    INSERT INTO llm_calls (
                        call_id, message_id, model, temperature, prompt_hash,
                        prompt_tokens, completion_tokens, total_tokens,
                        cost_usd, latency_ms, success, error_msg,
                        response_cached, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        f"{metadata.message_id}_{metadata.ts.isoformat()}",  # call_id
                        metadata.message_id,
                        metadata.model,
                        1.0,  # temperature (not in metadata)
                        metadata.prompt_hash,
                        metadata.tokens_in,  # Map to prompt_tokens
                        metadata.tokens_out,  # Map to completion_tokens
                        metadata.tokens_in + metadata.tokens_out,  # total_tokens
                        metadata.cost_usd,
                        metadata.latency_ms,
                        True,  # success (LLMCallMetadata doesn't track failures)
                        None,  # error_msg
                        metadata.cached,  # Map to response_cached
                        metadata.ts,  # Map to created_at
                    ),
                )
                conn.commit()

    def get_daily_llm_cost(self, date: datetime) -> float:
        """Get total LLM cost for a day.

        Args:
            date: Date to check

        Returns:
            Total cost in USD

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(cost_usd), 0.0) as total
                    FROM llm_calls
                    WHERE DATE(created_at) = DATE(%s)
                    """,
                    (date,),
                )
                row = cur.fetchone()
                return float(row[0]) if row else 0.0

    def get_cached_llm_response(self, prompt_hash: str) -> str | None:
        """Get cached LLM response by prompt hash.

        Args:
            prompt_hash: SHA256 hash of prompt

        Returns:
            Cached JSON response or None

        Raises:
            RepositoryError: On storage errors
        """
        # Not implemented yet - would need separate cache table
        return None

    def get_last_processed_ts(self, channel_id: str) -> str | None:
        """Get last processed timestamp for a channel from ingestion_state.

        Args:
            channel_id: Channel ID

        Returns:
            Last processed timestamp or None

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT last_processed_ts FROM ingestion_state WHERE channel_id = %s",
                    (channel_id,),
                )
                row = cur.fetchone()
                return row["last_processed_ts"] if row else None

    def update_last_processed_ts(self, channel_id: str, ts: str) -> None:
        """Update last processed timestamp for a channel in ingestion_state.

        Args:
            channel_id: Channel ID
            ts: New timestamp

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingestion_state (channel_id, last_processed_ts, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (channel_id) DO UPDATE SET
                        last_processed_ts = EXCLUDED.last_processed_ts,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (channel_id, ts),
                )
                conn.commit()

    def close(self) -> None:
        """Close all connections in pool."""
        if self._pool:
            self._pool.closeall()
            print("🔌 PostgreSQL connection pool closed")
