"""SQLite repository adapter for local storage.

Implements RepositoryProtocol with SQLite backend.
Schema compatible with future ClickHouse migration.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytz

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


class SQLiteRepository:
    """SQLite-based repository for MVP."""

    def __init__(self, db_path: str) -> None:
        """Initialize repository and ensure schema.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._create_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection.

        Returns:
            SQLite connection
        """
        conn = sqlite3.Connection(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_schema(self) -> None:
        """Create database schema if not exists."""
        print(f"🔧 Creating schema for database: {self.db_path}")
        conn = self._get_connection()
        cursor = conn.cursor()

        # Raw messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_slack_messages (
                message_id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                ts TEXT NOT NULL,
                ts_dt TEXT NOT NULL,
                user TEXT,
                user_real_name TEXT,
                user_display_name TEXT,
                user_email TEXT,
                user_profile_image TEXT,
                is_bot INTEGER,
                subtype TEXT,
                text TEXT,
                blocks_text TEXT,
                text_norm TEXT,
                links_raw TEXT,
                links_norm TEXT,
                anchors TEXT,
                attachments_count INTEGER DEFAULT 0,
                files_count INTEGER DEFAULT 0,
                reactions TEXT,
                total_reactions INTEGER DEFAULT 0,
                reply_count INTEGER DEFAULT 0,
                permalink TEXT,
                edited_ts TEXT,
                edited_user TEXT,
                ingested_at TEXT
            )
        """)

        # Candidates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_candidates (
                message_id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                ts_dt TEXT NOT NULL,
                text_norm TEXT,
                links_norm TEXT,
                anchors TEXT,
                score REAL,
                status TEXT CHECK(status IN ('new', 'llm_ok', 'llm_fail')),
                features_json TEXT
            )
        """)

        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                version INTEGER DEFAULT 1,
                message_id TEXT NOT NULL,
                source_msg_event_idx INTEGER,
                dedup_key TEXT UNIQUE,
                event_date TEXT NOT NULL,
                event_end TEXT,
                category TEXT,
                title TEXT,
                summary TEXT,
                impact_area TEXT,
                tags TEXT,
                links TEXT,
                anchors TEXT,
                confidence REAL,
                source_channels TEXT,
                ingested_at TEXT
            )
        """)

        # Create index on dedup_key
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_dedup_key
            ON events(dedup_key)
        """)

        # Create index on event_date
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_date
            ON events(event_date)
        """)

        # LLM calls table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT,
                prompt_hash TEXT,
                model TEXT,
                tokens_in INTEGER,
                tokens_out INTEGER,
                cost_usd REAL,
                latency_ms INTEGER,
                cached INTEGER,
                response_json TEXT,
                ts TEXT
            )
        """)

        # Watermarks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channel_watermarks (
                channel TEXT PRIMARY KEY,
                processing_ts TEXT,
                committed_ts TEXT
            )
        """)

        # Ingestion state table (tracks last processed timestamp per channel)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_state (
                channel_id TEXT PRIMARY KEY,
                last_ts REAL NOT NULL
            )
        """)

        conn.commit()
        print("✅ Schema creation completed")
        conn.close()

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
            conn = self._get_connection()
            cursor = conn.cursor()

            for msg in messages:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO raw_slack_messages (
                        message_id, channel, ts, ts_dt, user, user_real_name, user_display_name,
                        user_email, user_profile_image, is_bot, subtype,
                        text, blocks_text, text_norm, links_raw, links_norm,
                        anchors, attachments_count, files_count, reactions, total_reactions,
                        reply_count, permalink, edited_ts, edited_user, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        msg.message_id,
                        msg.channel,
                        msg.ts,
                        msg.ts_dt.isoformat(),
                        msg.user,
                        msg.user_real_name,
                        msg.user_display_name,
                        msg.user_email,
                        msg.user_profile_image,
                        1 if msg.is_bot else 0,
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
                        msg.ingested_at.isoformat(),
                    ),
                )

            conn.commit()
            count = len(messages)
            conn.close()
            return count

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to save messages: {e}")

    def get_watermark(self, channel: str) -> str | None:
        """Get committed watermark timestamp for channel.

        Args:
            channel: Channel ID

        Returns:
            Last committed timestamp or None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT committed_ts FROM channel_watermarks WHERE channel = ?",
                (channel,),
            )
            row = cursor.fetchone()
            conn.close()

            return row["committed_ts"] if row else None

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get watermark: {e}")

    def update_watermark(self, channel: str, ts: str) -> None:
        """Update committed watermark for channel.

        Args:
            channel: Channel ID
            ts: New watermark timestamp
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO channel_watermarks (channel, committed_ts, processing_ts)
                VALUES (?, ?, ?)
                """,
                (channel, ts, datetime.utcnow().isoformat()),
            )

            conn.commit()
            conn.close()

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to update watermark: {e}")

    def get_new_messages_for_candidates(self) -> list[SlackMessage]:
        """Get messages not yet in candidates table.

        Returns:
            List of messages to process
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT m.* FROM raw_slack_messages m
                LEFT JOIN event_candidates c ON m.message_id = c.message_id
                WHERE c.message_id IS NULL
                ORDER BY m.ts_dt DESC
            """)

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_message(row) for row in rows]

        except sqlite3.Error as e:
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
            conn = self._get_connection()
            cursor = conn.cursor()

            for candidate in candidates:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO event_candidates (
                        message_id, channel, ts_dt, text_norm, links_norm,
                        anchors, score, status, features_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.message_id,
                        candidate.channel,
                        candidate.ts_dt.isoformat(),
                        candidate.text_norm,
                        json.dumps(candidate.links_norm),
                        json.dumps(candidate.anchors),
                        candidate.score,
                        candidate.status.value,
                        candidate.features.model_dump_json(),
                    ),
                )

            conn.commit()
            count = len(candidates)
            conn.close()
            return count

        except sqlite3.Error as e:
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
            conn = self._get_connection()
            cursor = conn.cursor()

            if min_score is not None and batch_size is not None:
                cursor.execute(
                    """
                    SELECT * FROM event_candidates
                    WHERE status = 'new' AND score >= ?
                    ORDER BY score DESC
                    LIMIT ?
                    """,
                    (min_score, batch_size),
                )
            elif min_score is not None:
                cursor.execute(
                    """
                    SELECT * FROM event_candidates
                    WHERE status = 'new' AND score >= ?
                    ORDER BY score DESC
                    """,
                    (min_score,),
                )
            elif batch_size is not None:
                cursor.execute(
                    """
                    SELECT * FROM event_candidates
                    WHERE status = 'new'
                    ORDER BY score DESC
                    LIMIT ?
                    """,
                    (batch_size,),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM event_candidates
                    WHERE status = 'new'
                    ORDER BY score DESC
                    """
                )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_candidate(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get candidates: {e}")

    def update_candidate_status(self, message_id: str, status: str) -> None:
        """Update candidate processing status.

        Args:
            message_id: Message ID
            status: New status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE event_candidates SET status = ? WHERE message_id = ?",
                (status, message_id),
            )

            conn.commit()
            conn.close()

        except sqlite3.Error as e:
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
            conn = self._get_connection()
            cursor = conn.cursor()

            for event in events:
                # Check if dedup_key exists
                cursor.execute(
                    "SELECT version FROM events WHERE dedup_key = ?", (event.dedup_key,)
                )
                existing = cursor.fetchone()

                if existing:
                    # Update with incremented version
                    cursor.execute(
                        """
                        UPDATE events SET
                            version = ?,
                            title = ?,
                            summary = ?,
                            category = ?,
                            event_date = ?,
                            event_end = ?,
                            impact_area = ?,
                            tags = ?,
                            links = ?,
                            anchors = ?,
                            confidence = ?,
                            source_channels = ?
                        WHERE dedup_key = ?
                        """,
                        (
                            event.version,
                            event.title,
                            event.summary,
                            event.category.value,
                            event.event_date.isoformat(),
                            event.event_end.isoformat() if event.event_end else None,
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
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(event.event_id),
                            event.version,
                            event.message_id,
                            event.source_msg_event_idx,
                            event.dedup_key,
                            event.event_date.isoformat(),
                            event.event_end.isoformat() if event.event_end else None,
                            event.category.value,
                            event.title,
                            event.summary,
                            json.dumps(event.impact_area),
                            json.dumps(event.tags),
                            json.dumps(event.links),
                            json.dumps(event.anchors),
                            event.confidence,
                            json.dumps(event.source_channels),
                            event.ingested_at.isoformat(),
                        ),
                    )

            conn.commit()
            count = len(events)
            conn.close()
            return count

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to save events: {e}")

    def get_events_in_window(
        self, start_dt: datetime, end_dt: datetime
    ) -> list[Event]:
        """Get events within date window.

        Args:
            start_dt: Start datetime (UTC)
            end_dt: End datetime (UTC)

        Returns:
            List of events
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM events
                WHERE event_date >= ? AND event_date <= ?
                ORDER BY event_date ASC
                """,
                (start_dt.isoformat(), end_dt.isoformat()),
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_event(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get events: {e}")

    def query_events(self, criteria: EventQueryCriteria) -> list[Event]:
        """Query events using criteria builder.

        This method provides a type-safe alternative to building SQL strings.

        Args:
            criteria: Query criteria

        Returns:
            List of matching events

        Example:
            >>> from src.adapters.query_builders import recent_events_criteria
            >>> criteria = recent_events_criteria(days=7)
            >>> recent_events = repo.query_events(criteria)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

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
            conn.close()

            return [self._row_to_event(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to query events: {e}")

    def query_candidates(self, criteria: CandidateQueryCriteria) -> list[EventCandidate]:
        """Query event candidates using criteria builder.

        Args:
            criteria: Query criteria

        Returns:
            List of matching candidates

        Example:
            >>> from src.adapters.query_builders import high_priority_candidates_criteria
            >>> criteria = high_priority_candidates_criteria(threshold=15.0)
            >>> candidates = repo.query_candidates(criteria)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

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
            conn.close()

            return [self._row_to_candidate(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to query candidates: {e}")

    def save_llm_call(self, metadata: LLMCallMetadata) -> None:
        """Save LLM call metadata.

        Args:
            metadata: Call metadata
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO llm_calls (
                    message_id, prompt_hash, model, tokens_in, tokens_out,
                    cost_usd, latency_ms, cached, ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.message_id,
                    metadata.prompt_hash,
                    metadata.model,
                    metadata.tokens_in,
                    metadata.tokens_out,
                    metadata.cost_usd,
                    metadata.latency_ms,
                    1 if metadata.cached else 0,
                    metadata.ts.isoformat(),
                ),
            )

            conn.commit()
            conn.close()

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to save LLM call: {e}")

    def get_daily_llm_cost(self, date: datetime) -> float:
        """Get total LLM cost for a day.

        Args:
            date: Date to check

        Returns:
            Total cost in USD
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            date_str = date.strftime("%Y-%m-%d")
            cursor.execute(
                """
                SELECT SUM(cost_usd) as total FROM llm_calls
                WHERE DATE(ts) = ?
                """,
                (date_str,),
            )

            row = cursor.fetchone()
            conn.close()

            return float(row["total"]) if row and row["total"] else 0.0

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get daily cost: {e}")

    def get_cached_llm_response(self, prompt_hash: str) -> str | None:
        """Get cached LLM response by prompt hash.

        Args:
            prompt_hash: SHA256 hash of prompt

        Returns:
            Cached JSON response or None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT response_json FROM llm_calls
                WHERE prompt_hash = ? AND response_json IS NOT NULL
                ORDER BY ts DESC
                LIMIT 1
                """,
                (prompt_hash,),
            )

            row = cursor.fetchone()
            conn.close()

            return row["response_json"] if row else None

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get cached response: {e}")

    def save_llm_response(self, prompt_hash: str, response_json: str) -> None:
        """Save LLM response for caching.

        Args:
            prompt_hash: Prompt hash
            response_json: JSON response string
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE llm_calls
                SET response_json = ?
                WHERE prompt_hash = ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (response_json, prompt_hash),
            )

            conn.commit()
            conn.close()

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to save LLM response: {e}")

    def _row_to_message(self, row: sqlite3.Row) -> SlackMessage:
        """Convert database row to SlackMessage."""
        # Helper to safely get optional fields from sqlite3.Row
        def safe_get(key: str) -> str | None:
            try:
                value = row[key]
                return value if value else None
            except (KeyError, IndexError):
                return None

        def safe_get_int(key: str, default: int = 0) -> int:
            try:
                value = row[key]
                return int(value) if value is not None else default
            except (KeyError, IndexError, TypeError, ValueError):
                return default

        return SlackMessage(
            message_id=row["message_id"],
            channel=row["channel"],
            ts=row["ts"],
            ts_dt=datetime.fromisoformat(row["ts_dt"]).replace(tzinfo=pytz.UTC),
            user=row["user"],
            user_real_name=safe_get("user_real_name"),
            user_display_name=safe_get("user_display_name"),
            user_email=safe_get("user_email"),
            user_profile_image=safe_get("user_profile_image"),
            is_bot=bool(row["is_bot"]),
            subtype=row["subtype"],
            text=row["text"] or "",
            blocks_text=row["blocks_text"] or "",
            text_norm=row["text_norm"] or "",
            links_raw=json.loads(row["links_raw"] or "[]"),
            links_norm=json.loads(row["links_norm"] or "[]"),
            anchors=json.loads(row["anchors"] or "[]"),
            attachments_count=safe_get_int("attachments_count", 0),
            files_count=safe_get_int("files_count", 0),
            reactions=json.loads(row["reactions"] or "{}"),
            total_reactions=safe_get_int("total_reactions", 0),
            reply_count=row["reply_count"] or 0,
            permalink=safe_get("permalink"),
            edited_ts=safe_get("edited_ts"),
            edited_user=safe_get("edited_user"),
            ingested_at=datetime.fromisoformat(row["ingested_at"]).replace(
                tzinfo=pytz.UTC
            ),
        )

    def _row_to_candidate(self, row: sqlite3.Row) -> EventCandidate:
        """Convert database row to EventCandidate."""
        return EventCandidate(
            message_id=row["message_id"],
            channel=row["channel"],
            ts_dt=datetime.fromisoformat(row["ts_dt"]).replace(tzinfo=pytz.UTC),
            text_norm=row["text_norm"] or "",
            links_norm=json.loads(row["links_norm"] or "[]"),
            anchors=json.loads(row["anchors"] or "[]"),
            score=float(row["score"]),
            status=CandidateStatus(row["status"]),
            features=ScoringFeatures.model_validate_json(row["features_json"]),
        )

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        """Convert database row to Event."""
        return Event(
            event_id=UUID(row["event_id"]),
            version=int(row["version"]),
            message_id=row["message_id"],
            source_msg_event_idx=int(row["source_msg_event_idx"]),
            dedup_key=row["dedup_key"],
            event_date=datetime.fromisoformat(row["event_date"]).replace(
                tzinfo=pytz.UTC
            ),
            event_end=(
                datetime.fromisoformat(row["event_end"]).replace(tzinfo=pytz.UTC)
                if row["event_end"]
                else None
            ),
            category=EventCategory(row["category"]),
            title=row["title"],
            summary=row["summary"],
            impact_area=json.loads(row["impact_area"] or "[]"),
            tags=json.loads(row["tags"] or "[]"),
            links=json.loads(row["links"] or "[]"),
            anchors=json.loads(row["anchors"] or "[]"),
            confidence=float(row["confidence"]),
            source_channels=json.loads(row["source_channels"] or "[]"),
            ingested_at=datetime.fromisoformat(row["ingested_at"]).replace(
                tzinfo=pytz.UTC
            ),
        )

    def get_last_processed_ts(self, channel_id: str) -> float | None:
        """Get last processed timestamp for a channel.

        Args:
            channel_id: Slack channel ID

        Returns:
            Last processed timestamp (epoch seconds) or None if first run

        Raises:
            RepositoryError: On database errors
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT last_ts FROM ingestion_state
                WHERE channel_id = ?
                """,
                (channel_id,),
            )

            row = cursor.fetchone()
            conn.close()

            return float(row["last_ts"]) if row else None

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get last processed ts: {e}")

    def update_last_processed_ts(self, channel_id: str, last_ts: float) -> None:
        """Update last processed timestamp for a channel.

        Args:
            channel_id: Slack channel ID
            last_ts: Last processed timestamp (epoch seconds)

        Raises:
            RepositoryError: On database errors
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO ingestion_state (channel_id, last_ts)
                VALUES (?, ?)
                """,
                (channel_id, last_ts),
            )

            conn.commit()
            conn.close()

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to update last processed ts: {e}")

