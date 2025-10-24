"""Tests for multi-source domain models.

Tests MessageSource enum, MessageRecord, TelegramMessage, and source_id tracking.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

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


class TestMessageSource:
    """Test MessageSource enum."""

    def test_message_source_enum_values(self) -> None:
        """Test MessageSource has correct values."""
        assert MessageSource.SLACK.value == "slack"
        assert MessageSource.TELEGRAM.value == "telegram"

    def test_message_source_enum_from_string(self) -> None:
        """Test MessageSource can be created from string."""
        assert MessageSource("slack") == MessageSource.SLACK
        assert MessageSource("telegram") == MessageSource.TELEGRAM

    def test_message_source_invalid_raises_error(self) -> None:
        """Test invalid source_id raises ValueError."""
        with pytest.raises(ValueError):
            MessageSource("invalid")


class TestSlackMessage:
    """Test SlackMessage still works (backward compatibility)."""

    def test_slack_message_creation(self) -> None:
        """Test SlackMessage can be created."""
        msg = SlackMessage(
            message_id="test123",
            channel="C123",
            ts="1234567890.123456",
            ts_dt=datetime.now(tz=UTC),
            text="Test message",
        )
        assert msg.message_id == "test123"
        assert msg.channel == "C123"

    def test_slack_message_has_source_id(self) -> None:
        """Test SlackMessage has source_id field."""
        msg = SlackMessage(
            message_id="test123",
            channel="C123",
            ts="1234567890.123456",
            ts_dt=datetime.now(tz=UTC),
            text="Test message",
            source_id=MessageSource.SLACK,
        )
        assert msg.source_id == MessageSource.SLACK


class TestTelegramMessage:
    """Test TelegramMessage model."""

    def test_telegram_message_creation(self) -> None:
        """Test TelegramMessage can be created."""
        msg = TelegramMessage(
            message_id="tg_123",
            channel="@test_channel",
            message_date=datetime.now(tz=UTC),
            text="Test telegram message",
            source_id=MessageSource.TELEGRAM,
        )
        assert msg.message_id == "tg_123"
        assert msg.channel == "@test_channel"
        assert msg.source_id == MessageSource.TELEGRAM

    def test_telegram_message_with_optional_fields(self) -> None:
        """Test TelegramMessage with optional fields."""
        msg = TelegramMessage(
            message_id="tg_123",
            channel="@test_channel",
            message_date=datetime.now(tz=UTC),
            text="Test",
            source_id=MessageSource.TELEGRAM,
            sender_id="user123",
            sender_name="Test User",
            forward_from_channel="@original_channel",
            media_type="photo",
            views=100,
        )
        assert msg.sender_id == "user123"
        assert msg.sender_name == "Test User"
        assert msg.forward_from_channel == "@original_channel"
        assert msg.media_type == "photo"
        assert msg.views == 100


class TestEventCandidateSourceTracking:
    """Test EventCandidate preserves source_id."""

    def test_event_candidate_with_source_id(self) -> None:
        """Test EventCandidate stores source_id."""
        candidate = EventCandidate(
            message_id="test123",
            channel="C123",
            ts_dt=datetime.now(tz=UTC),
            text_norm="normalized text",
            links_norm=["https://example.com"],
            anchors=["ABC-123"],
            score=50.0,
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.SLACK,
        )
        assert candidate.source_id == MessageSource.SLACK

    def test_event_candidate_telegram_source(self) -> None:
        """Test EventCandidate with Telegram source."""
        candidate = EventCandidate(
            message_id="tg_123",
            channel="@test",
            ts_dt=datetime.now(tz=UTC),
            text_norm="telegram message",
            links_norm=[],
            anchors=[],
            score=30.0,
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.TELEGRAM,
        )
        assert candidate.source_id == MessageSource.TELEGRAM


class TestEventSourceTracking:
    """Test Event preserves source_id."""

    def test_event_with_source_id(self) -> None:
        """Test Event stores source_id."""
        event = Event(
            event_id=uuid4(),
            message_id="test123",
            source_channels=["releases"],
            action=ActionType.LAUNCH,
            object_name_raw="Test Product",
            category=EventCategory.PRODUCT,
            status=EventStatus.CONFIRMED,
            change_type=ChangeType.LAUNCH,
            environment=Environment.PROD,
            time_source=TimeSource.EXPLICIT,
            time_confidence=0.9,
            summary="Test summary",
            confidence=0.85,
            importance=75,
            cluster_key="test-cluster",
            dedup_key="test-dedup",
            source_id=MessageSource.SLACK,
        )
        assert event.source_id == MessageSource.SLACK

    def test_event_telegram_source(self) -> None:
        """Test Event with Telegram source."""
        event = Event(
            event_id=uuid4(),
            message_id="tg_123",
            source_channels=["crypto_news"],
            action=ActionType.LAUNCH,
            object_name_raw="Crypto Update",
            category=EventCategory.PRODUCT,
            status=EventStatus.CONFIRMED,
            change_type=ChangeType.LAUNCH,
            environment=Environment.PROD,
            time_source=TimeSource.EXPLICIT,
            time_confidence=0.9,
            summary="Crypto summary",
            confidence=0.85,
            importance=75,
            cluster_key="crypto-cluster",
            dedup_key="crypto-dedup",
            source_id=MessageSource.TELEGRAM,
        )
        assert event.source_id == MessageSource.TELEGRAM

    def test_event_source_id_defaults_to_slack(self) -> None:
        """Test Event defaults source_id to SLACK for backward compat."""
        event = Event(
            event_id=uuid4(),
            message_id="test123",
            source_channels=["releases"],
            action=ActionType.LAUNCH,
            object_name_raw="Test",
            category=EventCategory.PRODUCT,
            status=EventStatus.CONFIRMED,
            change_type=ChangeType.LAUNCH,
            environment=Environment.PROD,
            time_source=TimeSource.EXPLICIT,
            time_confidence=0.9,
            summary="Test",
            confidence=0.85,
            importance=75,
            cluster_key="test",
            dedup_key="test",
            # source_id not provided - should default to SLACK
        )
        assert event.source_id == MessageSource.SLACK

    def test_event_extracted_at_is_timezone_aware(self) -> None:
        """Event.extracted_at should always carry UTC timezone information."""

        event = Event(
            event_id=uuid4(),
            message_id="utc_check",
            source_channels=["releases"],
            action=ActionType.LAUNCH,
            object_name_raw="Test",
            category=EventCategory.PRODUCT,
            status=EventStatus.CONFIRMED,
            change_type=ChangeType.LAUNCH,
            environment=Environment.PROD,
            time_source=TimeSource.EXPLICIT,
            time_confidence=0.9,
            summary="Test",
            confidence=0.85,
            importance=75,
            cluster_key="test",
            dedup_key="test",
        )

        assert event.extracted_at.tzinfo is not None
        assert event.extracted_at.utcoffset() == timedelta(0)
