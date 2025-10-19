"""PostgreSQL repository implementation using psycopg2.

Connection pooling for production use with configurable min/max connections.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

import psycopg2

from src.domain.exceptions import RepositoryError
from src.domain.models import (
    Event,
    EventCandidate,
    LLMCallMetadata,
    MessageRecord,
    MessageSource,
    SlackMessage,
    TelegramMessage,
)

if TYPE_CHECKING:
    from src.adapters.query_builders import (
        CandidateQueryCriteria,
        EventQueryCriteria,
        PostgresCandidateQueryCriteria,
        PostgresEventQueryCriteria,
    )
    from src.config.settings import Settings


class PostgresRepository:
    """PostgreSQL repository with direct connections."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        settings: Settings | None = None,
    ):
        """Initialize PostgreSQL repository.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            settings: Optional settings for source configuration (for backward compatibility)

        Raises:
            RepositoryError: On connection errors
        """
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._settings = settings

        # Test connection on initialization
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
            )
            conn.close()
            print(f"🔧 PostgreSQL connected: {user}@{host}:{port}/{database}")
        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to connect to PostgreSQL: {e}") from e

    def _get_state_table_name(self, source_id: MessageSource | None) -> str:
        """Get state table name for the given source.

        Args:
            source_id: Message source identifier

        Returns:
            State table name from source configuration or fallback to defaults
        """
        # If no settings available, use backward compatible defaults
        if not self._settings or not source_id:
            return "ingestion_state_telegram" if source_id else "ingestion_state"

        # Try to get source configuration
        source_config = self._settings.get_source_config(source_id)
        if source_config and source_config.state_table:
            return source_config.state_table

        # Fallback to backward compatible defaults
        return "ingestion_state_telegram" if source_id else "ingestion_state"

    @contextmanager
    def _get_connection(self) -> Iterator[Any]:
        """Get database connection (context manager).

        Yields:
            Database connection

        Raises:
            RepositoryError: On connection errors
        """
        conn = None
        try:
            conn = psycopg2.connect(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password,
            )
            yield conn
        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to get database connection: {e}") from e
        finally:
            if conn:
                conn.close()

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
        """Convert database row to Event with new comprehensive structure.

        Args:
            row: Database row as dict

        Returns:
            Event instance
        """
        from uuid import UUID

        from src.domain.models import (
            ActionType,
            ChangeType,
            Environment,
            EventCategory,
            EventStatus,
            Severity,
            TimeSource,
        )

        return Event(
            # Identification
            event_id=UUID(row["event_id"]),
            message_id=row["message_id"],
            source_channels=row.get("source_channels")
            or [],  # Already parsed from JSONB
            extracted_at=row["extracted_at"],  # Already datetime from PostgreSQL
            # Title slots
            action=ActionType(row["action"]),
            object_id=row.get("object_id"),
            object_name_raw=row["object_name_raw"],
            qualifiers=row.get("qualifiers") or [],  # Already parsed from JSONB
            stroke=row.get("stroke"),
            anchor=row.get("anchor"),
            # Classification
            category=EventCategory(row["category"]),
            status=EventStatus(row["status"]),
            change_type=ChangeType(row["change_type"]),
            environment=Environment(row["environment"]),
            severity=Severity(row["severity"]) if row.get("severity") else None,
            # Time fields
            planned_start=row.get("planned_start"),  # Already datetime or None
            planned_end=row.get("planned_end"),
            actual_start=row.get("actual_start"),
            actual_end=row.get("actual_end"),
            time_source=TimeSource(row["time_source"]),
            time_confidence=row.get("time_confidence", 0.0),
            # Content
            summary=row.get("summary") or "",
            why_it_matters=row.get("why_it_matters"),
            impact_area=row.get("impact_area") or [],  # Already parsed from JSONB
            impact_type=row.get("impact_type") or [],  # Already parsed from JSONB
            links=row.get("links") or [],  # Already parsed from JSONB
            anchors=row.get("anchors") or [],  # Already parsed from JSONB
            # Metadata
            confidence=row.get("confidence", 0.0),
            importance=row.get("importance", 0),
            cluster_key=row.get("cluster_key") or "",
            dedup_key=row.get("dedup_key") or "",
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
                            user_real_name = EXCLUDED.user_real_name,
                            user_display_name = EXCLUDED.user_display_name,
                            user_email = EXCLUDED.user_email,
                            user_profile_image = EXCLUDED.user_profile_image,
                            text_raw = EXCLUDED.text_raw,
                            blocks_text = EXCLUDED.blocks_text,
                            text_norm = EXCLUDED.text_norm,
                            reply_count = EXCLUDED.reply_count,
                            reactions = EXCLUDED.reactions,
                            links_raw = EXCLUDED.links_raw,
                            links_norm = EXCLUDED.links_norm,
                            anchors = EXCLUDED.anchors,
                            attachments_count = EXCLUDED.attachments_count,
                            files_count = EXCLUDED.files_count,
                            total_reactions = EXCLUDED.total_reactions,
                            permalink = EXCLUDED.permalink,
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
            with conn.cursor(cursor_factory=psycopg2.extensions.cursor) as cur:
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
            with conn.cursor(cursor_factory=psycopg2.extensions.cursor) as cur:
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
                # Get column names from cursor description
                columns = [desc[0] for desc in cur.description]
                return [self._row_to_message(dict(zip(columns, row))) for row in rows]

    def get_new_messages_for_candidates_by_source(
        self, source_id: MessageSource
    ) -> list[MessageRecord]:
        """Get messages not yet in candidates table for a specific source.

        This method is source-agnostic and returns messages that implement
        the MessageRecord protocol, allowing scoring logic to work with any source.

        Args:
            source_id: Message source (SLACK, TELEGRAM, etc.)

        Returns:
            List of messages implementing MessageRecord protocol

        Raises:
            RepositoryError: On storage errors
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extensions.cursor) as cur:
                if source_id == MessageSource.SLACK:
                    # Query Slack messages
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
                    columns = [desc[0] for desc in cur.description]
                    return [
                        self._row_to_message(dict(zip(columns, row))) for row in rows
                    ]

                elif source_id == MessageSource.TELEGRAM:
                    # Query Telegram messages
                    cur.execute(
                        """
                        SELECT m.*
                        FROM raw_telegram_messages m
                        LEFT JOIN event_candidates c ON m.message_id = c.message_id
                        WHERE c.message_id IS NULL
                        ORDER BY m.message_date DESC
                        """
                    )
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    return [
                        self._row_to_telegram_message(dict(zip(columns, row)))
                        for row in rows
                    ]

                else:
                    raise RepositoryError(
                        f"Unsupported message source: {source_id}. "
                        f"Supported sources: {[s.value for s in MessageSource]}"
                    )

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
            with conn.cursor(cursor_factory=psycopg2.extensions.cursor) as cur:
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
                # Get column names from cursor description
                columns = [desc[0] for desc in cur.description]
                return [self._row_to_candidate(dict(zip(columns, row))) for row in rows]

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
                        "SELECT event_id FROM events WHERE dedup_key = %s",
                        (event.dedup_key,),
                    )
                    existing = cur.fetchone()

                    if existing:
                        # Update existing event
                        cur.execute(
                            """
                            UPDATE events SET
                                message_id = %s,
                                source_channels = %s,
                                extracted_at = %s,
                                action = %s,
                                object_id = %s,
                                object_name_raw = %s,
                                qualifiers = %s,
                                stroke = %s,
                                anchor = %s,
                                category = %s,
                                status = %s,
                                change_type = %s,
                                environment = %s,
                                severity = %s,
                                planned_start = %s,
                                planned_end = %s,
                                actual_start = %s,
                                actual_end = %s,
                                time_source = %s,
                                time_confidence = %s,
                                summary = %s,
                                why_it_matters = %s,
                                impact_area = %s,
                                impact_type = %s,
                                links = %s,
                                anchors = %s,
                                confidence = %s,
                                importance = %s,
                                cluster_key = %s
                            WHERE dedup_key = %s
                            """,
                            (
                                event.message_id,
                                json.dumps(event.source_channels),
                                event.extracted_at,
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
                                event.planned_start,
                                event.planned_end,
                                event.actual_start,
                                event.actual_end,
                                event.time_source.value,
                                event.time_confidence,
                                event.summary,
                                event.why_it_matters,
                                json.dumps(event.impact_area),
                                json.dumps(event.impact_type),
                                json.dumps(event.links),
                                json.dumps(event.anchors),
                                event.confidence,
                                event.importance,
                                event.cluster_key,
                                event.dedup_key,
                            ),
                        )
                    else:
                        # Insert new event
                        cur.execute(
                            """
                            INSERT INTO events (
                                event_id, message_id, source_channels, extracted_at,
                                action, object_id, object_name_raw, qualifiers, stroke, anchor,
                                category, status, change_type, environment, severity,
                                planned_start, planned_end, actual_start, actual_end,
                                time_source, time_confidence,
                                summary, why_it_matters, impact_area, impact_type, links, anchors,
                                confidence, importance, cluster_key, dedup_key
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            )
                            """,
                            (
                                str(event.event_id),
                                event.message_id,
                                json.dumps(event.source_channels),
                                event.extracted_at,
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
                                event.planned_start,
                                event.planned_end,
                                event.actual_start,
                                event.actual_end,
                                event.time_source.value,
                                event.time_confidence,
                                event.summary,
                                event.why_it_matters,
                                json.dumps(event.impact_area),
                                json.dumps(event.impact_type),
                                json.dumps(event.links),
                                json.dumps(event.anchors),
                                event.confidence,
                                event.importance,
                                event.cluster_key,
                                event.dedup_key,
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
            with conn.cursor(cursor_factory=psycopg2.extensions.cursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM events
                    WHERE COALESCE(actual_start, actual_end, planned_start, planned_end) >= %s
                      AND COALESCE(actual_start, actual_end, planned_start, planned_end) <= %s
                    ORDER BY COALESCE(actual_start, actual_end, planned_start, planned_end) DESC
                    """,
                    (start_dt, end_dt),
                )
                rows = cur.fetchall()
                # Get column names from cursor description
                columns = [desc[0] for desc in cur.description]
                return [self._row_to_event(dict(zip(columns, row))) for row in rows]

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

    def get_last_processed_ts(
        self, channel: str, source_id: MessageSource | None = None
    ) -> float | None:
        """Get last processed timestamp for a channel from ingestion_state.

        Args:
            channel: Channel ID
            source_id: Message source (optional for backward compatibility)

        Returns:
            Last processed timestamp or None

        Raises:
            RepositoryError: On storage errors
        """
        # Use source-specific state table name from configuration
        table_name = self._get_state_table_name(source_id)

        with self._get_connection() as conn:
            # Use transaction for atomic read
            conn.autocommit = False
            try:
                with conn.cursor(cursor_factory=psycopg2.extensions.cursor) as cur:
                    cur.execute(
                        f"SELECT last_processed_ts FROM {table_name} WHERE channel_id = %s",
                        (channel,),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return float(row["last_processed_ts"]) if row else None
            except Exception:
                conn.rollback()
                raise

    def update_last_processed_ts(
        self, channel: str, ts: float, source_id: MessageSource | None = None
    ) -> None:
        """Update last processed timestamp for a channel in ingestion_state.

        Args:
            channel: Channel ID
            ts: New timestamp
            source_id: Message source (optional for backward compatibility)

        Raises:
            RepositoryError: On storage errors
        """
        # Use source-specific state table name from configuration
        table_name = self._get_state_table_name(source_id)

        with self._get_connection() as conn:
            # Use transaction for atomic write
            conn.autocommit = False
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {table_name} (channel_id, last_processed_ts, updated_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (channel_id) DO UPDATE SET
                            last_processed_ts = EXCLUDED.last_processed_ts,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (channel, ts),
                    )
                    conn.commit()
            except Exception:
                conn.rollback()
                raise

    def query_events(self, criteria: "EventQueryCriteria") -> list[Event]:
        """Query events using structured criteria.

        Args:
            criteria: Query builder criteria object

        Returns:
            List of events matching criteria

        Raises:
            RepositoryError: On storage errors
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extensions.cursor) as cur:
                    # Build query parts
                    where_clause, where_params = criteria.to_where_clause()
                    order_clause = criteria.to_order_clause()
                    limit_clause, limit_params = criteria.to_limit_clause()

                    # Combine into full query (PostgreSQL uses %s placeholders)
                    query = f"""
                        SELECT * FROM events
                        WHERE {where_clause}
                        ORDER BY {order_clause}
                        {limit_clause}
                    """

                    # Execute with all parameters
                    all_params = where_params + limit_params
                    cur.execute(query, all_params)

                    rows = cur.fetchall()
                    return [self._row_to_event(dict(row)) for row in rows]

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to query events: {e}") from e

    def query_candidates(
        self, criteria: "CandidateQueryCriteria"
    ) -> list[EventCandidate]:
        """Query event candidates using structured criteria.

        Args:
            criteria: Query builder criteria object

        Returns:
            List of event candidates matching criteria

        Raises:
            RepositoryError: On storage errors
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extensions.cursor) as cur:
                    # Build query parts
                    where_clause, where_params = criteria.to_where_clause()
                    order_clause = criteria.to_order_clause()
                    limit_clause, limit_params = criteria.to_limit_clause()

                    # Combine into full query (PostgreSQL uses %s placeholders)
                    query = f"""
                        SELECT * FROM event_candidates
                        WHERE {where_clause}
                        ORDER BY {order_clause}
                        {limit_clause}
                    """

                    # Execute with all parameters
                    all_params = where_params + limit_params
                    cur.execute(query, all_params)

                    rows = cur.fetchall()
                    return [self._row_to_candidate(dict(row)) for row in rows]

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to query candidates: {e}") from e

    def save_telegram_messages(self, messages: list[TelegramMessage]) -> int:
        """Save Telegram messages to storage (idempotent upsert).

        Args:
            messages: List of Telegram messages

        Returns:
            Number of messages saved

        Raises:
            RepositoryError: On storage errors
        """
        if not messages:
            return 0

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    for msg in messages:
                        cur.execute(
                            """
                            INSERT INTO raw_telegram_messages (
                                message_id, channel, message_date, sender_id, sender_name,
                                text, text_norm, forward_from_channel, forward_from_message_id,
                                media_type, links_raw, links_norm, anchors, views, reply_count,
                                reactions, post_url, ingested_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (message_id) DO UPDATE SET
                                channel = EXCLUDED.channel,
                                message_date = EXCLUDED.message_date,
                                sender_id = EXCLUDED.sender_id,
                                sender_name = EXCLUDED.sender_name,
                                text = EXCLUDED.text,
                                text_norm = EXCLUDED.text_norm,
                                forward_from_channel = EXCLUDED.forward_from_channel,
                                forward_from_message_id = EXCLUDED.forward_from_message_id,
                                media_type = EXCLUDED.media_type,
                                links_raw = EXCLUDED.links_raw,
                                links_norm = EXCLUDED.links_norm,
                                anchors = EXCLUDED.anchors,
                                views = EXCLUDED.views,
                                reply_count = EXCLUDED.reply_count,
                                reactions = EXCLUDED.reactions,
                                post_url = EXCLUDED.post_url,
                                ingested_at = EXCLUDED.ingested_at
                            """,
                            (
                                msg.message_id,
                                msg.channel,
                                msg.message_date,
                                msg.sender_id,
                                msg.sender_name,
                                msg.text,
                                msg.text_norm,
                                msg.forward_from_channel,
                                msg.forward_from_message_id,
                                msg.media_type,
                                json.dumps(msg.links_raw),
                                json.dumps(msg.links_norm),
                                json.dumps(msg.anchors),
                                msg.views,
                                msg.reply_count,
                                json.dumps(msg.reactions),
                                msg.post_url,
                                msg.ingested_at,
                            ),
                        )

                    conn.commit()
                    count = len(messages)

            return count

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to save Telegram messages: {e}") from e

    def get_telegram_messages(
        self, channel: str, limit: int = 100
    ) -> list[TelegramMessage]:
        """Get Telegram messages from storage.

        Args:
            channel: Channel username or ID (optional filter)
            limit: Maximum messages to return

        Returns:
            List of Telegram messages ordered by date DESC
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extensions.cursor) as cur:
                    if channel:
                        cur.execute(
                            """
                            SELECT * FROM raw_telegram_messages
                            WHERE channel = %s
                            ORDER BY message_date DESC
                            LIMIT %s
                            """,
                            (channel, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT * FROM raw_telegram_messages
                            ORDER BY message_date DESC
                            LIMIT %s
                            """,
                            (limit,),
                        )

                    rows = cur.fetchall()
                    # Get column names from cursor description
                    columns = [desc[0] for desc in cur.description]
                    return [
                        self._row_to_telegram_message(dict(zip(columns, row)))
                        for row in rows
                    ]

        except psycopg2.Error as e:
            raise RepositoryError(f"Failed to get Telegram messages: {e}") from e

    def _row_to_telegram_message(self, row: dict[str, Any]) -> TelegramMessage:
        """Convert database row to TelegramMessage.

        Args:
            row: Database row

        Returns:
            TelegramMessage instance
        """
        return TelegramMessage(
            message_id=row["message_id"],
            channel=row["channel"],
            message_date=row["message_date"],
            sender_id=row["sender_id"],
            sender_name=row["sender_name"],
            text=row["text"] or "",
            text_norm=row["text_norm"] or "",
            forward_from_channel=row["forward_from_channel"],
            forward_from_message_id=row["forward_from_message_id"],
            media_type=row["media_type"],
            links_raw=row["links_raw"] if row["links_raw"] else [],
            links_norm=row["links_norm"] if row["links_norm"] else [],
            anchors=row["anchors"] if row["anchors"] else [],
            views=row["views"] or 0,
            reply_count=row["reply_count"] or 0,
            reactions=row["reactions"] if row["reactions"] else {},
            post_url=row["post_url"],
            ingested_at=row["ingested_at"],
        )

    def close(self) -> None:
        """Close repository (no-op for direct connections)."""
        print("🔌 PostgreSQL repository closed")
