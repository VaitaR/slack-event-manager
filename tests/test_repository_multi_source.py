"""Tests for repository multi-source support."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from src.adapters.repository_factory import create_repository
from src.config.settings import Settings
from src.domain.models import (
    ActionType,
    CandidateStatus,
    ChangeType,
    Environment,
    Event,
    EventCandidate,
    EventCategory,
    EventStatus,
    MessageSource,
    ScoringFeatures,
    SlackMessage,
    TelegramMessage,
    TimeSource,
)
from src.domain.protocols import RepositoryProtocol


def make_repository(db_path: Path) -> RepositoryProtocol:
    """Create a repository for the provided database path."""

    settings = Settings().model_copy(
        update={"database_type": "sqlite", "db_path": str(db_path)}
    )
    return create_repository(settings)


def create_test_candidate(
    message_id: str,
    channel: str,
    source_id: MessageSource = MessageSource.SLACK,
    **kwargs: Any,
) -> EventCandidate:
    """Helper to create test EventCandidate with minimal required fields."""
    return EventCandidate(
        message_id=message_id,
        channel=channel,
        ts_dt=kwargs.get("message_date", datetime.now(tz=UTC)),
        text_norm=kwargs.get("text_normalized", "test text"),
        links_norm=kwargs.get("links_norm", []),
        anchors=kwargs.get("anchors", []),
        score=kwargs.get("score", 0.8),
        status=kwargs.get("status", CandidateStatus.NEW),
        features=kwargs.get("features", ScoringFeatures()),
        source_id=source_id,
        lease_attempts=kwargs.get("lease_attempts", 0),
        processing_started_at=kwargs.get("processing_started_at"),
    )


def create_test_event(
    message_id: str,
    channel: str,
    source_id: MessageSource = MessageSource.SLACK,
    **kwargs: Any,
) -> Event:
    """Helper to create test Event with minimal required fields."""
    return Event(
        event_id=kwargs.get("event_id", uuid4()),
        message_id=message_id,
        source_channels=[channel],
        extracted_at=kwargs.get("extracted_at", datetime.now(tz=UTC)),
        action=kwargs.get("action", ActionType.LAUNCH),
        object_id=kwargs.get("object_id", None),
        object_name_raw=kwargs.get("title", "Test Event"),
        qualifiers=kwargs.get("qualifiers", []),
        stroke=kwargs.get("stroke", "Test Event"),
        anchor=kwargs.get("anchor", "test"),
        category=EventCategory(kwargs.get("category", "product")),
        status=kwargs.get("status", EventStatus.PLANNED),
        change_type=kwargs.get("change_type", ChangeType.LAUNCH),
        environment=kwargs.get("environment", Environment.PROD),
        severity=kwargs.get("severity", None),
        planned_start=kwargs.get("event_date", None),
        planned_end=kwargs.get("planned_end", None),
        actual_start=kwargs.get("actual_start", None),
        actual_end=kwargs.get("actual_end", None),
        time_source=kwargs.get("time_source", TimeSource.EXPLICIT),
        time_confidence=kwargs.get("time_confidence", 0.9),
        summary=kwargs.get("summary", "Test event summary"),
        why_it_matters=kwargs.get("why_it_matters", None),
        links=kwargs.get("links", []),
        anchors=kwargs.get("anchors", []),
        impact_area=kwargs.get("impact_area", []),
        impact_type=kwargs.get("impact_type", []),
        confidence=kwargs.get("confidence", 0.9),
        importance=kwargs.get("importance", 5),
        cluster_key=kwargs.get("cluster_key", f"cluster_{message_id}"),
        dedup_key=kwargs.get("dedup_key", f"dedup_{message_id}"),
        source_id=source_id,
    )


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary test database."""
    db_path = tmp_path / "test_multi_source.db"
    return db_path


class TestTelegramRawMessagesTable:
    """Test raw_telegram_messages table creation and operations."""

    def test_raw_telegram_messages_table_created(self, temp_db: Path) -> None:
        """Test that raw_telegram_messages table is created."""

        _ = make_repository(temp_db)

        # Check table exists
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_telegram_messages'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None, "raw_telegram_messages table should exist"

    def test_raw_telegram_messages_has_correct_schema(self, temp_db: Path) -> None:
        """Test that raw_telegram_messages has expected columns."""

        _ = make_repository(temp_db)

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(raw_telegram_messages)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected_columns = {
            "message_id",
            "channel",
            "message_date",
            "sender_id",
            "sender_name",
            "text",
            "text_norm",
            "forward_from_channel",
            "forward_from_message_id",
            "media_type",
            "links_raw",
            "links_norm",
            "anchors",
            "views",
            "ingested_at",
        }

        assert expected_columns.issubset(columns), (
            f"Missing columns: {expected_columns - columns}"
        )


class TestSaveTelegramMessages:
    """Test saving Telegram messages."""

    def test_save_telegram_messages_basic(self, temp_db: Path) -> None:
        """Test saving basic Telegram messages."""

        repo = make_repository(temp_db)

        messages = [
            TelegramMessage(
                message_id="123",
                channel="test_channel",
                message_date=datetime.now(tz=UTC),
                sender_id="user123",
                sender_name="Test User",
                text="Test message",
                text_norm="test message",
            )
        ]

        repo.save_telegram_messages(messages)

        # Verify saved
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw_telegram_messages")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_save_telegram_messages_with_optional_fields(self, temp_db: Path) -> None:
        """Test saving Telegram messages with all optional fields."""

        repo = make_repository(temp_db)

        messages = [
            TelegramMessage(
                message_id="456",
                channel="test_channel",
                message_date=datetime.now(tz=UTC),
                sender_id="user456",
                sender_name="User 456",
                text="Forwarded message",
                text_norm="forwarded message",
                forward_from_channel="original_channel",
                forward_from_message_id="789",
                media_type="photo",
                links_raw=["https://example.com"],
                links_norm=["https://example.com"],
                anchors=["example.com"],
                views=100,
            )
        ]

        repo.save_telegram_messages(messages)

        # Verify all fields saved
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT forward_from_channel, media_type, views FROM raw_telegram_messages WHERE message_id='456'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row == ("original_channel", "photo", 100)

    def test_save_telegram_messages_multiple(self, temp_db: Path) -> None:
        """Test saving multiple Telegram messages."""

        repo = make_repository(temp_db)

        messages = [
            TelegramMessage(
                message_id=str(i),
                channel="test_channel",
                message_date=datetime.now(tz=UTC),
                text=f"Message {i}",
                text_norm=f"message {i}",
            )
            for i in range(5)
        ]

        repo.save_telegram_messages(messages)

        # Verify count
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw_telegram_messages")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 5


class TestGetTelegramMessages:
    """Test retrieving Telegram messages."""

    def test_get_telegram_messages_basic(self, temp_db: Path) -> None:
        """Test retrieving Telegram messages."""

        repo = make_repository(temp_db)

        # Save messages
        messages = [
            TelegramMessage(
                message_id=str(i),
                channel="test_channel",
                message_date=datetime.now(tz=UTC) + timedelta(minutes=i),
                text=f"Message {i}",
                text_norm=f"message {i}",
            )
            for i in range(3)
        ]
        repo.save_telegram_messages(messages)

        # Retrieve
        retrieved = repo.get_telegram_messages(channel="test_channel", limit=10)

        assert len(retrieved) == 3
        assert all(isinstance(msg, TelegramMessage) for msg in retrieved)

    def test_get_telegram_messages_respects_limit(self, temp_db: Path) -> None:
        """Test limit parameter for Telegram messages."""

        repo = make_repository(temp_db)

        # Save 10 messages
        messages = [
            TelegramMessage(
                message_id=str(i),
                channel="test_channel",
                message_date=datetime.now(tz=UTC) + timedelta(minutes=i),
                text=f"Message {i}",
            )
            for i in range(10)
        ]
        repo.save_telegram_messages(messages)

        # Retrieve with limit
        retrieved = repo.get_telegram_messages(channel="test_channel", limit=5)

        assert len(retrieved) == 5

    def test_get_telegram_messages_filters_by_channel(self, temp_db: Path) -> None:
        """Test channel filtering for Telegram messages."""

        repo = make_repository(temp_db)

        # Save messages to different channels
        messages_a = [
            TelegramMessage(
                message_id=f"a{i}",
                channel="channel_a",
                message_date=datetime.now(tz=UTC),
                text=f"Message A{i}",
            )
            for i in range(3)
        ]
        messages_b = [
            TelegramMessage(
                message_id=f"b{i}",
                channel="channel_b",
                message_date=datetime.now(tz=UTC),
                text=f"Message B{i}",
            )
            for i in range(2)
        ]
        repo.save_telegram_messages(messages_a + messages_b)

        # Retrieve from channel_a only
        retrieved = repo.get_telegram_messages(channel="channel_a", limit=10)

        assert len(retrieved) == 3
        assert all(msg.channel == "channel_a" for msg in retrieved)


class TestSourceSpecificIngestionState:
    """Test source-specific ingestion state tracking."""

    def test_ingestion_state_slack_table_created(self, temp_db: Path) -> None:
        """Test that ingestion_state_slack table is created."""

        _ = make_repository(temp_db)

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ingestion_state_slack'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None, "ingestion_state_slack table should exist"

    def test_ingestion_state_telegram_table_created(self, temp_db: Path) -> None:
        """Test that ingestion_state_telegram table is created."""

        _ = make_repository(temp_db)

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ingestion_state_telegram'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None, "ingestion_state_telegram table should exist"

    def test_get_last_processed_ts_slack(self, temp_db: Path) -> None:
        """Test get_last_processed_ts for Slack source."""

        repo = make_repository(temp_db)

        # Set timestamp
        repo.update_last_processed_ts(
            channel="C123", ts=1234567890.0, source_id=MessageSource.SLACK
        )

        # Retrieve
        ts = repo.get_last_processed_ts(channel="C123", source_id=MessageSource.SLACK)

        assert ts == 1234567890.0

    def test_get_last_processed_ts_telegram(self, temp_db: Path) -> None:
        """Test get_last_processed_ts for Telegram source."""

        repo = make_repository(temp_db)

        # Set timestamp
        repo.update_last_processed_ts(
            channel="test_channel", ts=9876543210.0, source_id=MessageSource.TELEGRAM
        )

        # Retrieve
        ts = repo.get_last_processed_ts(
            channel="test_channel", source_id=MessageSource.TELEGRAM
        )

        assert ts == 9876543210.0

    def test_state_isolation_between_sources(self, temp_db: Path) -> None:
        """Test that Slack and Telegram state are isolated."""

        repo = make_repository(temp_db)

        # Set different timestamps for same channel in different sources
        repo.update_last_processed_ts(
            channel="shared_id", ts=1111111111.0, source_id=MessageSource.SLACK
        )
        repo.update_last_processed_ts(
            channel="shared_id", ts=2222222222.0, source_id=MessageSource.TELEGRAM
        )

        # Retrieve both
        slack_ts = repo.get_last_processed_ts(
            channel="shared_id", source_id=MessageSource.SLACK
        )
        telegram_ts = repo.get_last_processed_ts(
            channel="shared_id", source_id=MessageSource.TELEGRAM
        )

        assert slack_ts == 1111111111.0
        assert telegram_ts == 2222222222.0

    def test_get_last_processed_ts_returns_none_if_not_found(
        self, temp_db: Path
    ) -> None:
        """Test that get_last_processed_ts returns None for unknown channels."""

        repo = make_repository(temp_db)

        ts = repo.get_last_processed_ts(
            channel="nonexistent", source_id=MessageSource.SLACK
        )

        assert ts is None

    def test_legacy_get_last_processed_ts_defaults_to_slack(
        self, temp_db: Path
    ) -> None:
        """Test that calling get_last_processed_ts without source_id defaults to Slack."""

        repo = make_repository(temp_db)

        # Set Slack timestamp
        repo.update_last_processed_ts(
            channel="C123", ts=1234567890.0, source_id=MessageSource.SLACK
        )

        # Retrieve without source_id (legacy call)
        ts = repo.get_last_processed_ts(channel="C123")

        assert ts == 1234567890.0


class TestCandidatesAndEventsSourceTracking:
    """Test that candidates and events preserve source_id correctly."""

    def test_save_candidates_preserves_telegram_source(self, temp_db: Path) -> None:
        """Test saving candidates from Telegram messages."""

        repo = make_repository(temp_db)

        candidates = [
            create_test_candidate(
                message_id="tg123",
                channel="test_channel",
                source_id=MessageSource.TELEGRAM,
                score=0.8,
            )
        ]

        repo.save_candidates(candidates)

        # Retrieve and check source_id
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT source_id FROM event_candidates WHERE message_id='tg123'"
        )
        source_id = cursor.fetchone()[0]
        conn.close()

        assert source_id == MessageSource.TELEGRAM.value

    def test_save_events_preserves_telegram_source(self, temp_db: Path) -> None:
        """Test saving events from Telegram candidates."""

        repo = make_repository(temp_db)

        events = [
            create_test_event(
                message_id="tg456",
                channel="test_channel",
                source_id=MessageSource.TELEGRAM,
                title="Telegram Event",
                category="product",
                confidence=0.9,
                importance=5,
            )
        ]

        repo.save_events(events)

        # Retrieve and check source_id
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT source_id FROM events WHERE message_id='tg456'")
        source_id = cursor.fetchone()[0]
        conn.close()

        assert source_id == MessageSource.TELEGRAM.value

    def test_save_events_merges_source_channels_on_update(self, temp_db: Path) -> None:
        """Updating an existing event should persist the union of source channels."""

        repo = make_repository(temp_db)

        initial_event = create_test_event(
            message_id="union",
            channel="channel-a",
            source_id=MessageSource.TELEGRAM,
            dedup_key="union-key",
            cluster_key="union-cluster",
        )
        repo.save_events([initial_event])

        merged_event = initial_event.model_copy(
            update={
                "source_channels": ["channel-a", "channel-b"],
                "source_id": MessageSource.TELEGRAM,
            }
        )

        repo.save_events([merged_event])

        with sqlite3.connect(str(temp_db)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT source_channels, source_id FROM events WHERE dedup_key = ?",
                ("union-key",),
            )
            row = cursor.fetchone()

        assert row is not None
        stored_channels = set(json.loads(row[0]))
        assert stored_channels == {"channel-a", "channel-b"}
        assert row[1] == MessageSource.TELEGRAM.value

    def test_get_candidates_filters_by_source(self, temp_db: Path) -> None:
        """Test retrieving candidates filtered by source_id."""

        repo = make_repository(temp_db)

        # Save mixed source candidates
        candidates = [
            create_test_candidate(
                message_id=f"slack{i}",
                channel="test",
                source_id=MessageSource.SLACK,
                text_normalized=f"slack {i}",
                score=0.8,
            )
            for i in range(3)
        ] + [
            create_test_candidate(
                message_id=f"tg{i}",
                channel="test",
                source_id=MessageSource.TELEGRAM,
                text_normalized=f"tg {i}",
                score=0.8,
            )
            for i in range(2)
        ]
        repo.save_candidates(candidates)

        # Get only Telegram candidates
        telegram_candidates = repo.get_candidates_by_source(
            source_id=MessageSource.TELEGRAM
        )

        assert len(telegram_candidates) == 2
        assert all(c.source_id == MessageSource.TELEGRAM for c in telegram_candidates)

    def test_get_events_filters_by_source(self, temp_db: Path) -> None:
        """Test retrieving events filtered by source_id."""

        repo = make_repository(temp_db)

        # Save mixed source events
        events = [
            create_test_event(
                message_id=f"slack{i}",
                channel="test",
                source_id=MessageSource.SLACK,
                title=f"Slack Event {i}",
                category="product",
                confidence=0.9,
                importance=5,
            )
            for i in range(3)
        ] + [
            create_test_event(
                message_id=f"tg{i}",
                channel="test",
                source_id=MessageSource.TELEGRAM,
                title=f"Telegram Event {i}",
                category="product",
                confidence=0.9,
                importance=5,
            )
            for i in range(2)
        ]
        repo.save_events(events)

        # Get only Telegram events
        telegram_events = repo.get_events_by_source(source_id=MessageSource.TELEGRAM)

        assert len(telegram_events) == 2
        assert all(e.source_id == MessageSource.TELEGRAM for e in telegram_events)


class TestDashboardQueries:
    """Tests for repository helpers used by dashboard presentation layer."""

    def test_get_recent_slack_messages_returns_most_recent_first(
        self, temp_db: Path, sample_slack_message: SlackMessage
    ) -> None:
        """Repository should return Slack messages ordered by recency."""

        repo = make_repository(temp_db)

        older_message = sample_slack_message.model_copy(
            update={
                "message_id": "older",
                "ts": "123.456",
                "ts_dt": sample_slack_message.ts_dt - timedelta(hours=1),
                "ingested_at": sample_slack_message.ingested_at - timedelta(hours=1),
            }
        )
        repo.save_messages([older_message, sample_slack_message])

        recent_messages = repo.get_recent_slack_messages(limit=1)

        assert len(recent_messages) == 1
        assert recent_messages[0].message_id == sample_slack_message.message_id

    def test_get_recent_candidates_orders_by_score(self, temp_db: Path) -> None:
        """Repository should surface top-scoring candidates first."""

        repo = make_repository(temp_db)

        high_score = create_test_candidate(message_id="high", channel="test", score=0.9)
        low_score = create_test_candidate(message_id="low", channel="test", score=0.1)
        repo.save_candidates([low_score, high_score])

        recent_candidates = repo.get_recent_candidates(limit=1)

        assert len(recent_candidates) == 1
        assert recent_candidates[0].message_id == "high"

    def test_get_recent_events_orders_by_extraction_time(self, temp_db: Path) -> None:
        """Repository should return most recently extracted events first."""

        repo = make_repository(temp_db)

        base_time = datetime.now(tz=UTC)
        older_event = create_test_event(
            message_id="older",
            channel="test",
            extracted_at=base_time - timedelta(hours=1),
        )
        newer_event = create_test_event(
            message_id="newer",
            channel="test",
            extracted_at=base_time,
        )
        repo.save_events([older_event, newer_event])

        recent_events = repo.get_recent_events(limit=1)

        assert len(recent_events) == 1
        assert recent_events[0].message_id == "newer"


class TestCandidateLeasing:
    """Ensure candidates are leased atomically for extraction."""

    def test_candidates_marked_processing_until_final_status(
        self, temp_db: Path
    ) -> None:
        """Repository should reserve candidates to avoid duplicate work."""

        repo = make_repository(temp_db)
        candidate_one = create_test_candidate(message_id="m1", channel="C1", score=0.9)
        candidate_two = create_test_candidate(message_id="m2", channel="C2", score=0.5)
        repo.save_candidates([candidate_one, candidate_two])

        first_batch = repo.get_candidates_for_extraction(batch_size=1)
        assert [c.message_id for c in first_batch] == ["m1"]

        with sqlite3.connect(str(temp_db)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status FROM event_candidates WHERE message_id = ?",
                ("m1",),
            )
            status_row = cursor.fetchone()
            assert status_row is not None
            assert status_row[0] == CandidateStatus.PROCESSING.value

        # Second lease should return the remaining candidate only
        second_batch = repo.get_candidates_for_extraction(batch_size=5)
        assert [c.message_id for c in second_batch] == ["m2"]

        # Mark the first candidate complete and ensure it is not re-leased
        repo.update_candidate_status("m1", CandidateStatus.LLM_OK.value)
        third_batch = repo.get_candidates_for_extraction(batch_size=10)
        assert [c.message_id for c in third_batch] == []

    def test_processing_candidates_released_after_timeout(self, temp_db: Path) -> None:
        """Stuck processing candidates should return to the queue after TTL."""

        repo = make_repository(temp_db)
        candidate = create_test_candidate(
            message_id="m-timeout", channel="C1", score=0.9
        )
        repo.save_candidates([candidate])

        stale_started_at = datetime.now(tz=UTC) - timedelta(hours=2)
        with sqlite3.connect(str(temp_db)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE event_candidates
                SET status = ?, processing_started_at = ?, lease_attempts = 1
                WHERE message_id = ?
                """,
                (
                    CandidateStatus.PROCESSING.value,
                    stale_started_at.isoformat(),
                    "m-timeout",
                ),
            )
            conn.commit()

        leased = repo.get_candidates_for_extraction(batch_size=1)

        assert [c.message_id for c in leased] == ["m-timeout"]
        assert leased[0].status is CandidateStatus.PROCESSING
        assert leased[0].lease_attempts == 2
        assert leased[0].processing_started_at is not None
        assert leased[0].processing_started_at > stale_started_at

    def test_lease_attempts_increment_each_time(self, temp_db: Path) -> None:
        """Lease attempts counter increases whenever a candidate is leased."""

        repo = make_repository(temp_db)
        candidate = create_test_candidate(message_id="m-retry", channel="C1", score=0.9)
        repo.save_candidates([candidate])

        first_lease = repo.get_candidates_for_extraction(batch_size=1)
        assert first_lease[0].lease_attempts == 1

        # Simulate failure by returning candidate to NEW state
        with sqlite3.connect(str(temp_db)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE event_candidates
                SET status = ?, processing_started_at = NULL
                WHERE message_id = ?
                """,
                (CandidateStatus.NEW.value, "m-retry"),
            )
            conn.commit()

        second_lease = repo.get_candidates_for_extraction(batch_size=1)
        assert second_lease[0].lease_attempts == 2
