"""SQLite repository adapter for local storage.

Implements RepositoryProtocol with SQLite backend.
"""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast
from uuid import UUID

import pytz

from src.adapters.bulk_persistence import (
    DatabaseBackend,
    EventDTO,
    RelationDTO,
    upsert_event_relations_bulk,
    upsert_events_bulk,
)
from src.adapters.query_builders import (
    CandidateQueryCriteria,
    EventQueryCriteria,
)
from src.config.logging_config import get_logger
from src.domain.candidate_constants import CANDIDATE_LEASE_TIMEOUT
from src.domain.exceptions import RepositoryError
from src.domain.models import (
    CandidateStatus,
    Event,
    EventCandidate,
    EventCategory,
    LLMCallMetadata,
    MessageSource,
    ScoringFeatures,
    SlackMessage,
    TelegramMessage,
)
from src.ports.task_queue import TaskQueuePort

logger = get_logger(__name__)

SLACK_STATE_TABLE: Final[str] = "slack_ingestion_state"
LEGACY_SLACK_STATE_TABLE: Final[str] = "ingestion_state_slack"
LEGACY_SLACK_STATE_BACKUP_TABLE: Final[str] = "ingestion_state_slack_legacy"


class SQLiteRepository:
    """SQLite-based repository for MVP."""

    def __init__(self, db_path: str, *, chunk_size: int = 500) -> None:
        """Initialize repository and ensure schema.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        if chunk_size <= 0:
            raise RepositoryError("chunk_size must be positive")
        self._bulk_chunk_size = chunk_size

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
        logger.info("sqlite_schema_creation_started", db_path=str(self.db_path))
        conn = self._get_connection()
        cursor = conn.cursor()

        # Raw Slack messages table
        cursor.execute(
            """
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
        """
        )

        # Raw Telegram messages table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_telegram_messages (
                message_id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                message_date TEXT NOT NULL,
                sender_id TEXT,
                sender_name TEXT,
                text TEXT,
                text_norm TEXT,
                forward_from_channel TEXT,
                forward_from_message_id TEXT,
                media_type TEXT,
                links_raw TEXT,
                links_norm TEXT,
                anchors TEXT,
                views INTEGER DEFAULT 0,
                reply_count INTEGER DEFAULT 0,
                reactions TEXT,
                reactions_count INTEGER DEFAULT 0,
                post_url TEXT,
                attachments_count INTEGER DEFAULT 0,
                files_count INTEGER DEFAULT 0,
                bot_id TEXT,
                is_bot INTEGER DEFAULT 0,
                reply_to_id TEXT,
                thread_id TEXT,
                has_file INTEGER DEFAULT 0,
                file_mime TEXT,
                ingested_at TEXT
            )
        """
        )

        telegram_alter_statements = [
            "ALTER TABLE raw_telegram_messages ADD COLUMN reactions_count INTEGER DEFAULT 0",
            "ALTER TABLE raw_telegram_messages ADD COLUMN attachments_count INTEGER DEFAULT 0",
            "ALTER TABLE raw_telegram_messages ADD COLUMN files_count INTEGER DEFAULT 0",
            "ALTER TABLE raw_telegram_messages ADD COLUMN bot_id TEXT",
            "ALTER TABLE raw_telegram_messages ADD COLUMN is_bot INTEGER DEFAULT 0",
            "ALTER TABLE raw_telegram_messages ADD COLUMN reply_to_id TEXT",
            "ALTER TABLE raw_telegram_messages ADD COLUMN thread_id TEXT",
            "ALTER TABLE raw_telegram_messages ADD COLUMN has_file INTEGER DEFAULT 0",
            "ALTER TABLE raw_telegram_messages ADD COLUMN file_mime TEXT",
        ]

        for statement in telegram_alter_statements:
            try:
                cursor.execute(statement)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise

        # Candidates table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS event_candidates (
                message_id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                ts_dt TEXT NOT NULL,
                text_norm TEXT,
                links_norm TEXT,
                anchors TEXT,
                score REAL,
                status TEXT CHECK(status IN ('new', 'processing', 'llm_ok', 'llm_fail')),
                features_json TEXT,
                source_id TEXT DEFAULT 'slack',
                processing_started_at TEXT,
                lease_attempts INTEGER DEFAULT 0
            )
        """
        )

        # Backfill new lease tracking columns for existing databases
        try:
            cursor.execute(
                "ALTER TABLE event_candidates ADD COLUMN processing_started_at TEXT"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise

        try:
            cursor.execute(
                "ALTER TABLE event_candidates ADD COLUMN lease_attempts INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise

        cursor.execute(
            "UPDATE event_candidates SET lease_attempts = 0 WHERE lease_attempts IS NULL"
        )

        # Events table (new structure with title slots)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                -- Identification
                event_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                source_channels TEXT NOT NULL,
                extracted_at TEXT NOT NULL,

                -- Title Slots (source of truth)
                action TEXT NOT NULL,
                object_id TEXT,
                object_name_raw TEXT NOT NULL,
                qualifiers TEXT,
                stroke TEXT,
                anchor TEXT,

                -- Classification & Lifecycle
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                change_type TEXT NOT NULL,
                environment TEXT NOT NULL,
                severity TEXT,

                -- Time Fields
                planned_start TEXT,
                planned_end TEXT,
                actual_start TEXT,
                actual_end TEXT,
                time_source TEXT NOT NULL,
                time_confidence REAL NOT NULL,

                -- Content & Links
                summary TEXT NOT NULL,
                why_it_matters TEXT,
                links TEXT,
                anchors TEXT,
                impact_area TEXT,
                impact_type TEXT,

                -- Quality & Importance
                confidence REAL NOT NULL,
                importance INTEGER NOT NULL,

                -- Clustering
                cluster_key TEXT NOT NULL,
                dedup_key TEXT UNIQUE NOT NULL,

                -- Source Tracking
                source_id TEXT DEFAULT 'slack'
            )
        """
        )

        # Event relations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS event_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_event_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_event_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_event_id) REFERENCES events(event_id),
                FOREIGN KEY (target_event_id) REFERENCES events(event_id),
                UNIQUE(source_event_id, target_event_id, relation_type)
            )
        """
        )

        # Create indexes on events
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_dedup_key
            ON events(dedup_key)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_cluster_key
            ON events(cluster_key)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_status
            ON events(status)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_importance
            ON events(importance)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_object_id
            ON events(object_id)
        """
        )

        # Create index on event_relations
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_relations_source
            ON event_relations(source_event_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_relations_target
            ON event_relations(target_event_id)
        """
        )

        # LLM calls table
        cursor.execute(
            """
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
        """
        )

        # Source-specific ingestion state tables
        self._ensure_slack_state_schema(cursor)

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_state_telegram (
                channel_id TEXT PRIMARY KEY,
                last_processed_ts REAL NOT NULL,
                last_processed_message_id TEXT,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS channel_watermarks (
                channel TEXT PRIMARY KEY,
                committed_ts TEXT,
                processing_ts TEXT
            )
            """
        )

        conn.commit()
        conn.close()
        logger.info("sqlite_schema_creation_completed", db_path=str(self.db_path))

    def _ensure_slack_state_schema(self, cursor: sqlite3.Cursor) -> None:
        """Ensure Slack ingestion state table uses canonical schema and migrate data."""

        if self._sqlite_object_exists(
            cursor, LEGACY_SLACK_STATE_TABLE, "table"
        ) and not self._sqlite_object_exists(cursor, SLACK_STATE_TABLE, "table"):
            cursor.execute(
                f"ALTER TABLE {LEGACY_SLACK_STATE_TABLE} RENAME TO {SLACK_STATE_TABLE}"
            )

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SLACK_STATE_TABLE} (
                channel_id TEXT PRIMARY KEY,
                max_processed_ts REAL NOT NULL DEFAULT 0,
                resume_cursor TEXT,
                resume_min_ts REAL,
                updated_at TEXT
            )
            """
        )

        self._ensure_slack_state_columns(cursor)

        if self._sqlite_object_exists(cursor, LEGACY_SLACK_STATE_TABLE, "table"):
            self._backfill_slack_state_data(cursor, LEGACY_SLACK_STATE_TABLE)
            self._archive_legacy_slack_state_table(cursor)

        cursor.execute(f"DROP VIEW IF EXISTS {LEGACY_SLACK_STATE_TABLE}")
        cursor.execute(
            f"""
            CREATE VIEW IF NOT EXISTS {LEGACY_SLACK_STATE_TABLE} AS
            SELECT channel_id,
                   max_processed_ts AS last_processed_ts,
                   updated_at
            FROM {SLACK_STATE_TABLE}
            """
        )

    @staticmethod
    def _sqlite_object_exists(cursor: sqlite3.Cursor, name: str, obj_type: str) -> bool:
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
            (obj_type, name),
        )
        return cursor.fetchone() is not None

    def _backfill_slack_state_data(
        self, cursor: sqlite3.Cursor, source_table: str
    ) -> None:
        cursor.execute(f"PRAGMA table_info({source_table})")
        columns = {row[1] for row in cursor.fetchall()}
        if "last_processed_ts" not in columns:
            return

        cursor.execute(
            f"""
            INSERT INTO {SLACK_STATE_TABLE} (channel_id, max_processed_ts, updated_at)
            SELECT channel_id, last_processed_ts, updated_at FROM {source_table}
            ON CONFLICT(channel_id) DO UPDATE SET
                max_processed_ts=excluded.max_processed_ts,
                updated_at=excluded.updated_at
            """
        )

    def _archive_legacy_slack_state_table(self, cursor: sqlite3.Cursor) -> None:
        backup_name = LEGACY_SLACK_STATE_BACKUP_TABLE
        if self._sqlite_object_exists(cursor, backup_name, "table"):
            backup_name = f"{backup_name}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        cursor.execute(
            f"ALTER TABLE {LEGACY_SLACK_STATE_TABLE} RENAME TO {backup_name}"
        )

    def _ensure_slack_state_columns(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(f"PRAGMA table_info({SLACK_STATE_TABLE})")
        columns_info = {row[1]: row for row in cursor.fetchall()}

        if "max_processed_ts" not in columns_info:
            if "last_processed_ts" in columns_info:
                cursor.execute(
                    f"ALTER TABLE {SLACK_STATE_TABLE} RENAME COLUMN last_processed_ts TO max_processed_ts"
                )
            else:
                cursor.execute(
                    f"ALTER TABLE {SLACK_STATE_TABLE} ADD COLUMN max_processed_ts REAL NOT NULL DEFAULT 0"
                )

        cursor.execute(f"PRAGMA table_info({SLACK_STATE_TABLE})")
        column_names = {row[1] for row in cursor.fetchall()}

        if "resume_cursor" not in column_names:
            cursor.execute(
                f"ALTER TABLE {SLACK_STATE_TABLE} ADD COLUMN resume_cursor TEXT"
            )

        if "resume_min_ts" not in column_names:
            cursor.execute(
                f"ALTER TABLE {SLACK_STATE_TABLE} ADD COLUMN resume_min_ts REAL"
            )

        if "updated_at" not in column_names:
            cursor.execute(
                f"ALTER TABLE {SLACK_STATE_TABLE} ADD COLUMN updated_at TEXT"
            )

        cursor.execute(
            f"""
            UPDATE {SLACK_STATE_TABLE}
            SET updated_at = ?
            WHERE updated_at IS NULL
            """,
            (datetime.now(UTC).isoformat(),),
        )

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
            conn = self._get_connection()
            cursor = conn.cursor()

            for msg in messages:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO raw_telegram_messages (
                        message_id, channel, message_date, sender_id, sender_name,
                        text, text_norm, forward_from_channel, forward_from_message_id,
                        media_type, links_raw, links_norm, anchors, views, reply_count,
                        reactions, reactions_count, post_url, attachments_count,
                        files_count, bot_id, is_bot, reply_to_id, thread_id, has_file,
                        file_mime, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        msg.message_id,
                        msg.channel,
                        msg.message_date.isoformat(),
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
                        msg.reactions_count,
                        msg.post_url,
                        msg.attachments_count,
                        msg.files_count,
                        msg.bot_id,
                        1 if msg.is_bot else 0,
                        msg.reply_to_id,
                        msg.thread_id,
                        1 if msg.has_file else 0,
                        msg.file_mime,
                        msg.ingested_at.isoformat(),
                    ),
                )

            conn.commit()
            count = len(messages)
            conn.close()
            return count

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to save Telegram messages: {e}")

    def get_telegram_messages(
        self, channel: str, limit: int = 100
    ) -> list[TelegramMessage]:
        """Get Telegram messages from storage.

        Args:
            channel: Channel username or ID
            limit: Maximum messages to return

        Returns:
            List of Telegram messages ordered by date DESC
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM raw_telegram_messages
                WHERE channel = ?
                ORDER BY message_date DESC
                LIMIT ?
                """,
                (channel, limit),
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_telegram_message(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get Telegram messages: {e}")

    def _row_to_telegram_message(self, row: sqlite3.Row) -> TelegramMessage:
        """Convert database row to TelegramMessage.

        Args:
            row: Database row

        Returns:
            TelegramMessage instance
        """
        reactions_data = json.loads(row["reactions"]) if row["reactions"] else {}
        reactions_count = row["reactions_count"] or sum(reactions_data.values())

        return TelegramMessage(
            message_id=row["message_id"],
            channel=row["channel"],
            message_date=datetime.fromisoformat(row["message_date"]),
            sender_id=row["sender_id"],
            sender_name=row["sender_name"],
            text=row["text"] or "",
            text_norm=row["text_norm"] or "",
            forward_from_channel=row["forward_from_channel"],
            forward_from_message_id=row["forward_from_message_id"],
            media_type=row["media_type"],
            links_raw=json.loads(row["links_raw"]) if row["links_raw"] else [],
            links_norm=json.loads(row["links_norm"]) if row["links_norm"] else [],
            anchors=json.loads(row["anchors"]) if row["anchors"] else [],
            views=row["views"] or 0,
            reply_count=row["reply_count"] or 0,
            reactions=reactions_data,
            reactions_count=reactions_count,
            post_url=row["post_url"],
            attachments_count=row["attachments_count"] or 0,
            files_count=row["files_count"] or 0,
            bot_id=row["bot_id"],
            is_bot=bool(row["is_bot"]),
            reply_to_id=row["reply_to_id"],
            thread_id=row["thread_id"],
            is_reply=bool(row["reply_to_id"]),
            has_file=bool(row["has_file"]),
            file_mime=row["file_mime"],
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
        )

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
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO channel_watermarks (channel, committed_ts, processing_ts)
                VALUES (?, ?, ?)
                """,
                (channel, ts, datetime.now(tz=pytz.UTC).isoformat()),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to update watermark: {e}") from e
        finally:
            conn.close()

    def task_queue(self) -> TaskQueuePort:
        """SQLite backend does not support the task queue."""

        msg = "Task queue is only supported when using PostgreSQL backend"
        raise NotImplementedError(msg)

    def get_new_messages_for_candidates(self) -> list[SlackMessage]:
        """Get messages not yet in candidates table.

        Returns:
            List of messages to process
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT m.* FROM raw_slack_messages m
                LEFT JOIN event_candidates c ON m.message_id = c.message_id
                WHERE c.message_id IS NULL
                ORDER BY m.ts_dt DESC
            """
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_message(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get new messages: {e}")

    def get_new_messages_for_candidates_by_source(
        self, source_id: MessageSource
    ) -> list[SlackMessage | TelegramMessage]:
        """Get messages not yet in candidates table for a specific source.

        This method is source-agnostic and returns messages that implement
        the MessageRecord protocol (both SlackMessage and TelegramMessage do).

        Args:
            source_id: Message source (SLACK, TELEGRAM, etc.)

        Returns:
            List of messages (SlackMessage or TelegramMessage)

        Raises:
            RepositoryError: On storage errors or unsupported source
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if source_id == MessageSource.SLACK:
                # Query Slack messages
                cursor.execute(
                    """
                    SELECT m.* FROM raw_slack_messages m
                    LEFT JOIN event_candidates c ON m.message_id = c.message_id
                    WHERE c.message_id IS NULL
                    ORDER BY m.ts_dt DESC
                """
                )
                rows = cursor.fetchall()
                conn.close()
                return [self._row_to_message(row) for row in rows]

            elif source_id == MessageSource.TELEGRAM:
                # Query Telegram messages
                cursor.execute(
                    """
                    SELECT m.* FROM raw_telegram_messages m
                    LEFT JOIN event_candidates c ON m.message_id = c.message_id
                    WHERE c.message_id IS NULL
                    ORDER BY m.message_date DESC
                """
                )
                rows = cursor.fetchall()
                conn.close()
                return [self._row_to_telegram_message(row) for row in rows]

            else:
                conn.close()
                raise RepositoryError(
                    f"Unsupported message source: {source_id}. "
                    f"Supported sources: {[s.value for s in MessageSource]}"
                )

        except sqlite3.Error as e:
            raise RepositoryError(
                f"Failed to get new messages for source {source_id}: {e}"
            )

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
                        anchors, score, status, features_json, source_id,
                        processing_started_at, lease_attempts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        candidate.source_id.value,
                        candidate.processing_started_at.isoformat()
                        if candidate.processing_started_at
                        else None,
                        candidate.lease_attempts,
                    ),
                )

            conn.commit()
            count = len(candidates)
            conn.close()
            return count

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to save candidates: {e}")

    def get_candidates_for_extraction(
        self,
        batch_size: int | None = 50,
        min_score: float | None = None,
        source_id: MessageSource | None = None,
    ) -> list[EventCandidate]:
        """Get candidates ready for LLM extraction.

        Args:
            batch_size: Maximum candidates to return (None = no limit)
            min_score: Minimum score filter
            source_id: Filter by message source (None = all sources)

        Returns:
            List of candidates ordered by score DESC
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("BEGIN IMMEDIATE")

            now = datetime.now(tz=pytz.UTC)
            stale_before = (now - CANDIDATE_LEASE_TIMEOUT).isoformat()

            cursor.execute(
                """
                UPDATE event_candidates
                SET status = ?, processing_started_at = NULL
                WHERE status = ? AND (
                    processing_started_at IS NULL OR processing_started_at < ?
                )
                """,
                (
                    CandidateStatus.NEW.value,
                    CandidateStatus.PROCESSING.value,
                    stale_before,
                ),
            )

            where_conditions = ["status = 'new'"]
            params: list[Any] = []

            if source_id is not None:
                where_conditions.append("source_id = ?")
                params.append(source_id.value)

            if min_score is not None:
                where_conditions.append("score >= ?")
                params.append(min_score)

            query = f"""
                SELECT * FROM event_candidates
                WHERE {" AND ".join(where_conditions)}
                ORDER BY score DESC
            """

            if batch_size is not None:
                query += " LIMIT ?"
                params.append(batch_size)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                conn.commit()
                conn.close()
                return []

            message_ids = [row["message_id"] for row in rows]
            placeholders = ",".join(["?"] * len(message_ids))
            cursor.execute(
                f"""
                UPDATE event_candidates
                SET status = ?, processing_started_at = ?,
                    lease_attempts = COALESCE(lease_attempts, 0) + 1
                WHERE message_id IN ({placeholders})
                """,
                [CandidateStatus.PROCESSING.value, now.isoformat(), *message_ids],
            )

            conn.commit()
            conn.close()

            return [
                self._row_to_candidate(row).model_copy(
                    update={
                        "status": CandidateStatus.PROCESSING,
                        "processing_started_at": now,
                        "lease_attempts": (row["lease_attempts"] or 0) + 1,
                    }
                )
                for row in rows
            ]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get candidates: {e}")

    def get_candidate_by_message_id(self, message_id: str) -> EventCandidate | None:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT *
                FROM event_candidates
                WHERE message_id = ?
                """,
                (message_id,),
            )
            row = cursor.fetchone()
            conn.close()
        except sqlite3.Error as exc:  # pragma: no cover - defensive path
            raise RepositoryError(f"Failed to load candidate: {exc}")

        if row is None:
            return None

        return self._row_to_candidate(row)

    def get_recent_slack_messages(self, limit: int = 100) -> list[SlackMessage]:
        """Get most recent Slack messages for presentation use."""

        if limit <= 0:
            return []

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM raw_slack_messages
                ORDER BY ts_dt DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            conn.close()
            return [self._row_to_message(row) for row in rows]
        except sqlite3.Error as exc:  # pragma: no cover - defensive path
            raise RepositoryError(f"Failed to load recent Slack messages: {exc}")

    def get_recent_candidates(self, limit: int = 100) -> list[EventCandidate]:
        """Get most recent event candidates for presentation use."""

        if limit <= 0:
            return []

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM event_candidates
                ORDER BY score DESC, ts_dt DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            conn.close()
            return [self._row_to_candidate(row) for row in rows]
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load recent candidates: {exc}")

    def get_recent_events(self, limit: int = 100) -> list[Event]:
        """Get most recently extracted events for presentation use."""

        if limit <= 0:
            return []

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM events
                ORDER BY extracted_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            conn.close()
            return [self._row_to_event(row) for row in rows]
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load recent events: {exc}")

    def update_candidate_status(self, message_id: str, status: str) -> None:
        """Update candidate processing status.

        Args:
            message_id: Message ID
            status: New status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if status == CandidateStatus.PROCESSING.value:
                cursor.execute(
                    """
                    UPDATE event_candidates
                    SET status = ?, processing_started_at = ?,
                        lease_attempts = COALESCE(lease_attempts, 0) + 1
                    WHERE message_id = ?
                    """,
                    (status, datetime.now(tz=pytz.UTC).isoformat(), message_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE event_candidates
                    SET status = ?, processing_started_at = NULL
                    WHERE message_id = ?
                    """,
                    (status, message_id),
                )

            conn.commit()
            conn.close()

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to update candidate status: {e}")

    def save_events(self, events: list[Event]) -> int:
        """Save events with new comprehensive structure (upsert by dedup_key)."""

        if not events:
            return 0

        conn = self._get_connection()
        try:
            event_dtos = [
                EventDTO.from_event(
                    event,
                    backend=DatabaseBackend.SQLITE,
                    connection=conn,
                )
                for event in events
            ]
            upsert_events_bulk(event_dtos, chunk=self._bulk_chunk_size)

            relation_dtos: list[RelationDTO] = []
            for event in events:
                if not event.relations:
                    continue
                for relation in event.relations:
                    relation_dtos.append(
                        RelationDTO.from_relation(
                            str(event.event_id),
                            relation,
                            backend=DatabaseBackend.SQLITE,
                            connection=conn,
                        )
                    )

            if relation_dtos:
                upsert_event_relations_bulk(relation_dtos, chunk=self._bulk_chunk_size)

            return len(events)
        except sqlite3.Error as exc:
            conn.rollback()
            raise RepositoryError(f"Failed to save events: {exc}") from exc
        finally:
            conn.close()

    def get_events_in_window(self, start_dt: datetime, end_dt: datetime) -> list[Event]:
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
                WHERE COALESCE(actual_start, actual_end, planned_start, planned_end) >= ?
                  AND COALESCE(actual_start, actual_end, planned_start, planned_end) <= ?
                ORDER BY COALESCE(actual_start, actual_end, planned_start, planned_end) ASC
                """,
                (start_dt.isoformat(), end_dt.isoformat()),
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_event(row) for row in rows]

        except sqlite3.Error as e:
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

        Example:
            >>> repo = SQLiteRepository("data/events.db")
            >>> events = repo.get_events_in_window_filtered(
            ...     start_dt=datetime(2025, 10, 1),
            ...     end_dt=datetime(2025, 10, 13),
            ...     min_confidence=0.7,
            ...     max_events=10
            ... )
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build query with confidence filter
            query = """
                SELECT * FROM events
                WHERE COALESCE(actual_start, actual_end, planned_start, planned_end) >= ?
                  AND COALESCE(actual_start, actual_end, planned_start, planned_end) <= ?
                  AND confidence >= ?
                ORDER BY COALESCE(actual_start, actual_end, planned_start, planned_end) ASC
            """

            params: list[Any] = [
                start_dt.isoformat(),
                end_dt.isoformat(),
                min_confidence,
            ]

            # Add limit if specified
            if max_events is not None:
                query += " LIMIT ?"
                params.append(max_events)

            cursor.execute(query, params)

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_event(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get filtered events: {e}")

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

    def query_candidates(
        self, criteria: CandidateQueryCriteria
    ) -> list[EventCandidate]:
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

    def get_cached_llm_response(
        self, prompt_hash: str, *, max_age: timedelta | None = None
    ) -> str | None:
        """Get cached LLM response by prompt hash.

        Args:
            prompt_hash: SHA256 hash of prompt
            max_age: Optional TTL duration

        Returns:
            Cached JSON response or None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT response_json, ts FROM llm_calls
                WHERE prompt_hash = ? AND response_json IS NOT NULL
                ORDER BY ts DESC
                LIMIT 1
                """,
                (prompt_hash,),
            )

            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            response_json = cast(str, row["response_json"])
            cached_ts_raw = cast(str | None, row["ts"])

            if max_age is not None:
                if not cached_ts_raw:
                    logger.warning(
                        "llm_cache_missing_timestamp",
                        prompt_hash=prompt_hash[:12],
                    )
                    self.invalidate_llm_cache_entry(prompt_hash)
                    return None
                try:
                    cached_ts = datetime.fromisoformat(cached_ts_raw)
                except ValueError:
                    logger.warning(
                        "llm_cache_invalid_timestamp",
                        prompt_hash=prompt_hash[:12],
                        ts=cached_ts_raw,
                    )
                    self.invalidate_llm_cache_entry(prompt_hash)
                    return None

                if cached_ts.tzinfo is None:
                    cached_ts = cached_ts.replace(tzinfo=UTC)

                if cached_ts < datetime.now(tz=UTC) - max_age:
                    logger.info(
                        "llm_cache_entry_expired",
                        prompt_hash=prompt_hash[:12],
                        cached_at=cached_ts.isoformat(),
                    )
                    self.invalidate_llm_cache_entry(prompt_hash)
                    return None

            return response_json

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

    def invalidate_llm_cache_entry(self, prompt_hash: str) -> None:
        """Clear cached payload for a prompt hash."""

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE llm_calls
                SET response_json = NULL
                WHERE prompt_hash = ?
            """,
                (prompt_hash,),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to invalidate cached response: {e}")

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
        # Get source_id with backward compatibility
        try:
            source_id_str = row["source_id"]
        except (KeyError, IndexError):
            source_id_str = "slack"
        source_id = (
            MessageSource(source_id_str) if source_id_str else MessageSource.SLACK
        )

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
            source_id=source_id,
            lease_attempts=int(row["lease_attempts"] or 0)
            if "lease_attempts" in row.keys()
            else 0,
            processing_started_at=(
                datetime.fromisoformat(row["processing_started_at"]).replace(
                    tzinfo=pytz.UTC
                )
                if row["processing_started_at"]
                else None
            )
            if "processing_started_at" in row.keys()
            else None,
        )

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        """Convert database row to Event with new comprehensive structure."""
        from src.domain.models import (
            ActionType,
            ChangeType,
            Environment,
            EventStatus,
            Severity,
            TimeSource,
        )

        def _parse_dt(value: str | None) -> datetime | None:
            if not value:
                return None
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=pytz.UTC)
            return parsed.astimezone(pytz.UTC)

        # Get source_id with backward compatibility
        try:
            source_id_str = row["source_id"]
        except (KeyError, IndexError):
            source_id_str = "slack"
        source_id = (
            MessageSource(source_id_str) if source_id_str else MessageSource.SLACK
        )

        extracted_at = _parse_dt(row["extracted_at"])
        if extracted_at is None:
            raise RepositoryError("Event row missing extracted_at timestamp")

        return Event(
            # Identification
            event_id=UUID(row["event_id"]),
            message_id=row["message_id"],
            source_channels=json.loads(row["source_channels"] or "[]"),
            extracted_at=extracted_at,
            # Title slots
            action=ActionType(row["action"]),
            object_id=row["object_id"],
            object_name_raw=row["object_name_raw"],
            qualifiers=json.loads(row["qualifiers"] or "[]"),
            stroke=row["stroke"],
            anchor=row["anchor"],
            # Classification
            category=EventCategory(row["category"]),
            status=EventStatus(row["status"]),
            change_type=ChangeType(row["change_type"]),
            environment=Environment(row["environment"]),
            severity=Severity(row["severity"]) if row["severity"] else None,
            # Time fields
            planned_start=_parse_dt(row["planned_start"]),
            planned_end=_parse_dt(row["planned_end"]),
            actual_start=_parse_dt(row["actual_start"]),
            actual_end=_parse_dt(row["actual_end"]),
            time_source=TimeSource(row["time_source"]),
            time_confidence=float(row["time_confidence"]),
            # Content
            summary=row["summary"],
            why_it_matters=row["why_it_matters"],
            links=json.loads(row["links"] or "[]"),
            anchors=json.loads(row["anchors"] or "[]"),
            impact_area=json.loads(row["impact_area"] or "[]"),
            impact_type=json.loads(row["impact_type"] or "[]"),
            # Quality
            confidence=float(row["confidence"]),
            importance=int(row["importance"]),
            # Clustering
            cluster_key=row["cluster_key"],
            dedup_key=row["dedup_key"],
            relations=[],  # Relations loaded separately if needed
            # Source tracking
            source_id=source_id,
        )

    def get_last_processed_ts(
        self, channel: str, source_id: MessageSource | None = None
    ) -> float | None:
        """Get last processed timestamp for a channel (source-specific or legacy).

        Args:
            channel: Channel ID
            source_id: Message source (None = legacy table for backward compatibility)

        Returns:
            Last processed timestamp (epoch seconds) or None if first run

        Raises:
            RepositoryError: On database errors
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if source_id == MessageSource.TELEGRAM:
                cursor.execute(
                    """
                    SELECT last_processed_ts FROM ingestion_state_telegram
                    WHERE channel_id = ?
                    """,
                    (channel,),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT max_processed_ts FROM {SLACK_STATE_TABLE}
                    WHERE channel_id = ?
                    """,
                    (channel,),
                )

            row = cursor.fetchone()
            conn.close()

            return float(row[0]) if row else None

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get last processed ts: {e}")

    def update_last_processed_ts(
        self, channel: str, ts: float, source_id: MessageSource | None = None
    ) -> None:
        """Update last processed timestamp for a channel (source-specific or legacy).

        Args:
            channel: Channel ID
            ts: Last processed timestamp (epoch seconds)
            source_id: Message source (None = legacy table for backward compatibility)

        Raises:
            RepositoryError: On database errors
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if source_id == MessageSource.TELEGRAM:
                cursor.execute(
                    """
                    INSERT INTO ingestion_state_telegram (
                        channel_id,
                        last_processed_ts,
                        last_processed_message_id,
                        updated_at
                    ) VALUES (?, ?, NULL, datetime('now'))
                    ON CONFLICT(channel_id) DO UPDATE SET
                        last_processed_ts=excluded.last_processed_ts,
                        updated_at=excluded.updated_at,
                        last_processed_message_id=COALESCE(
                            excluded.last_processed_message_id,
                            ingestion_state_telegram.last_processed_message_id
                        )
                    """,
                    (channel, ts),
                )
            else:
                cursor.execute(
                    f"""
                    INSERT INTO {SLACK_STATE_TABLE} (
                        channel_id,
                        max_processed_ts,
                        updated_at
                    ) VALUES (?, ?, datetime('now'))
                    ON CONFLICT(channel_id) DO UPDATE SET
                        max_processed_ts=excluded.max_processed_ts,
                        updated_at=excluded.updated_at
                    """,
                    (channel, ts),
                )

            conn.commit()
            conn.close()

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to update last processed ts: {e}")

    def get_last_processed_message_id(
        self, channel: str, source_id: MessageSource | None = None
    ) -> str | None:
        """Get last processed message ID for a Telegram channel.

        Args:
            channel: Channel ID (Telegram username)
            source_id: Message source (must be TELEGRAM for this method)

        Returns:
            Last processed message ID or None if first run

        Raises:
            RepositoryError: On database errors
        """
        if source_id != MessageSource.TELEGRAM:
            raise RepositoryError(
                f"get_last_processed_message_id only supports TELEGRAM source, got {source_id}"
            )

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT last_processed_message_id FROM ingestion_state_telegram
                WHERE channel_id = ?
                """,
                (channel,),
            )

            row = cursor.fetchone()
            conn.close()

            return row[0] if row else None

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get last processed message ID: {e}")

    def update_last_processed_message_id(
        self, channel: str, message_id: str, source_id: MessageSource | None = None
    ) -> None:
        """Update last processed message ID for a Telegram channel.

        Args:
            channel: Channel ID (Telegram username)
            message_id: Message ID to set
            source_id: Message source (must be TELEGRAM for this method)

        Raises:
            RepositoryError: On database errors
        """
        if source_id != MessageSource.TELEGRAM:
            raise RepositoryError(
                f"update_last_processed_message_id only supports TELEGRAM source, got {source_id}"
            )

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO ingestion_state_telegram
                (channel_id, last_processed_ts, last_processed_message_id, updated_at)
                VALUES (?, 0, ?, datetime('now'))
                """,
                (channel, message_id),
            )

            conn.commit()
            conn.close()

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to update last processed message ID: {e}")

    def get_candidates_by_source(
        self, source_id: MessageSource, limit: int = 100
    ) -> list[EventCandidate]:
        """Get candidates filtered by source_id.

        Args:
            source_id: Message source to filter by
            limit: Maximum candidates to return

        Returns:
            List of candidates from specified source

        Raises:
            RepositoryError: On database errors
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM event_candidates
                WHERE source_id = ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (source_id.value, limit),
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_candidate(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get candidates by source: {e}")

    def get_events_by_source(
        self, source_id: MessageSource, limit: int = 100
    ) -> list[Event]:
        """Get events filtered by source_id.

        Args:
            source_id: Message source to filter by
            limit: Maximum events to return

        Returns:
            List of events from specified source

        Raises:
            RepositoryError: On database errors
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM events
                WHERE source_id = ?
                ORDER BY extracted_at DESC
                LIMIT ?
                """,
                (source_id.value, limit),
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_event(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get events by source: {e}")

    def get_related_events(self, event_id: UUID) -> list[Event]:
        """Get events related to the given event.

        Args:
            event_id: Event ID to find relations for

        Returns:
            List of related events

        Raises:
            RepositoryError: On database errors
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT e.* FROM events e
                JOIN event_relations r ON e.event_id = r.target_event_id
                WHERE r.source_event_id = ?
                """,
                (str(event_id),),
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_event(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get related events: {e}")

    def get_events_by_cluster_key(self, cluster_key: str) -> list[Event]:
        """Get all events in the same cluster (same initiative).

        Args:
            cluster_key: Cluster key

        Returns:
            List of events in cluster

        Raises:
            RepositoryError: On database errors
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM events
                WHERE cluster_key = ?
                ORDER BY extracted_at ASC
                """,
                (cluster_key,),
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_event(row) for row in rows]

        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get events by cluster: {e}")
