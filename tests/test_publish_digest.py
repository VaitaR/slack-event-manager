"""Unit tests for publish digest use case.

Tests confidence filtering, max events limit, category sorting, and block building.
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import pytz

from src.adapters.slack_client import SlackClient
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import Settings
from src.domain.models import DigestResult, Event, EventCategory
from src.use_cases.publish_digest import (
    build_digest_blocks,
    build_event_block,
    chunk_blocks,
    format_event_date,
    get_confidence_icon,
    publish_digest_use_case,
    sort_events_for_digest,
)


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    with patch("src.config.settings.load_config_yaml") as mock_config:
        mock_config.return_value = {
            "llm": {"model": "gpt-5-nano", "temperature": 1.0},
            "database": {"path": "data/test.db"},
            "slack": {"digest_channel_id": "C06B5NJLY4B"},
            "processing": {"tz_default": "Europe/Amsterdam"},
            "digest": {
                "max_events": 10,
                "min_confidence": 0.7,
                "lookback_hours": 48,
                "category_priorities": {
                    "product": 1,
                    "risk": 2,
                    "process": 3,
                    "marketing": 4,
                    "org": 5,
                    "unknown": 6,
                },
            },
        }
        settings = Settings(
            slack_bot_token="xoxb-test",
            openai_api_key="sk-test",
        )
        return settings


@pytest.fixture
def sample_events() -> list[Event]:
    """Create sample events with various categories and confidence scores."""
    base_date = datetime(2025, 10, 13, 10, 0, 0, tzinfo=pytz.UTC)

    return [
        Event(
            event_id=uuid4(),
            message_id="msg1",
            source_msg_event_idx=0,
            dedup_key="key1",
            event_date=base_date,
            category=EventCategory.PRODUCT,
            title="Product Release v2.0",
            summary="Major product release with new features",
            confidence=0.95,
        ),
        Event(
            event_id=uuid4(),
            message_id="msg2",
            source_msg_event_idx=0,
            dedup_key="key2",
            event_date=base_date + timedelta(hours=1),
            category=EventCategory.RISK,
            title="Security Incident",
            summary="Critical security issue detected",
            confidence=0.85,
        ),
        Event(
            event_id=uuid4(),
            message_id="msg3",
            source_msg_event_idx=0,
            dedup_key="key3",
            event_date=base_date + timedelta(hours=2),
            category=EventCategory.PROCESS,
            title="Process Update",
            summary="Updated workflow process",
            confidence=0.75,
        ),
        Event(
            event_id=uuid4(),
            message_id="msg4",
            source_msg_event_idx=0,
            dedup_key="key4",
            event_date=base_date + timedelta(hours=3),
            category=EventCategory.MARKETING,
            title="Campaign Launch",
            summary="Marketing campaign started",
            confidence=0.65,
        ),
        Event(
            event_id=uuid4(),
            message_id="msg5",
            source_msg_event_idx=0,
            dedup_key="key5",
            event_date=base_date + timedelta(hours=4),
            category=EventCategory.ORG,
            title="Team Restructure",
            summary="Organizational changes",
            confidence=0.55,
        ),
        Event(
            event_id=uuid4(),
            message_id="msg6",
            source_msg_event_idx=0,
            dedup_key="key6",
            event_date=base_date + timedelta(hours=5),
            category=EventCategory.UNKNOWN,
            title="Unknown Event",
            summary="Unclear event type",
            confidence=0.45,
        ),
    ]


def test_get_confidence_icon() -> None:
    """Test confidence icon selection."""
    # Arrange & Act & Assert
    assert get_confidence_icon(0.95) == "✅"
    assert get_confidence_icon(0.8) == "✅"
    assert get_confidence_icon(0.75) == "⚠️"
    assert get_confidence_icon(0.6) == "⚠️"
    assert get_confidence_icon(0.5) == "❓"
    assert get_confidence_icon(0.2) == "❓"


def test_format_event_date() -> None:
    """Test event date formatting."""
    # Arrange
    dt = datetime(2025, 10, 15, 8, 0, tzinfo=pytz.UTC)

    # Act
    result = format_event_date(dt, "Europe/Amsterdam")

    # Assert
    assert "15.10.2025" in result
    assert "10:00" in result


def test_sort_events_for_digest_by_date(sample_events: list[Event]) -> None:
    """Test sorting events by date."""
    # Arrange
    shuffled = sample_events[::-1]  # Reverse order

    # Act
    sorted_events = sort_events_for_digest(shuffled)

    # Assert
    for i in range(len(sorted_events) - 1):
        assert sorted_events[i].event_date <= sorted_events[i + 1].event_date


def test_sort_events_for_digest_by_category(sample_events: list[Event]) -> None:
    """Test sorting events by category priority."""
    # Arrange
    base_date = datetime(2025, 10, 13, 10, 0, 0, tzinfo=pytz.UTC)
    same_date_events = [
        Event(
            event_id=uuid4(),
            message_id="msg1",
            source_msg_event_idx=0,
            dedup_key="key1",
            event_date=base_date,
            category=EventCategory.MARKETING,
            title="Marketing",
            summary="Test",
            confidence=0.8,
        ),
        Event(
            event_id=uuid4(),
            message_id="msg2",
            source_msg_event_idx=0,
            dedup_key="key2",
            event_date=base_date,
            category=EventCategory.PRODUCT,
            title="Product",
            summary="Test",
            confidence=0.8,
        ),
        Event(
            event_id=uuid4(),
            message_id="msg3",
            source_msg_event_idx=0,
            dedup_key="key3",
            event_date=base_date,
            category=EventCategory.RISK,
            title="Risk",
            summary="Test",
            confidence=0.8,
        ),
    ]

    # Act
    sorted_events = sort_events_for_digest(same_date_events)

    # Assert - product should be first, then risk, then marketing
    assert sorted_events[0].category == EventCategory.PRODUCT
    assert sorted_events[1].category == EventCategory.RISK
    assert sorted_events[2].category == EventCategory.MARKETING


def test_sort_events_for_digest_by_confidence(sample_events: list[Event]) -> None:
    """Test sorting events by confidence when date and category are same."""
    # Arrange
    base_date = datetime(2025, 10, 13, 10, 0, 0, tzinfo=pytz.UTC)
    same_category_events = [
        Event(
            event_id=uuid4(),
            message_id="msg1",
            source_msg_event_idx=0,
            dedup_key="key1",
            event_date=base_date,
            category=EventCategory.PRODUCT,
            title="Low confidence",
            summary="Test",
            confidence=0.6,
        ),
        Event(
            event_id=uuid4(),
            message_id="msg2",
            source_msg_event_idx=0,
            dedup_key="key2",
            event_date=base_date,
            category=EventCategory.PRODUCT,
            title="High confidence",
            summary="Test",
            confidence=0.9,
        ),
    ]

    # Act
    sorted_events = sort_events_for_digest(same_category_events)

    # Assert - higher confidence first
    assert sorted_events[0].confidence == 0.9
    assert sorted_events[1].confidence == 0.6


def test_build_event_block() -> None:
    """Test building event block with compact format."""
    # Arrange
    event = Event(
        event_id=uuid4(),
        message_id="msg1",
        source_msg_event_idx=0,
        dedup_key="key1",
        event_date=datetime(2025, 10, 13, 10, 0, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Test Event",
        summary="This is a test event",
        confidence=0.85,
        links=["https://example.com"],
    )

    # Act
    block = build_event_block(event, "Europe/Amsterdam")

    # Assert - Compact format: only emoji + title
    assert block["type"] == "section"
    assert "text" in block
    assert block["text"]["type"] == "mrkdwn"
    assert "🚀" in block["text"]["text"]  # Product category emoji
    assert "Test Event" in block["text"]["text"]
    assert block["text"]["text"] == "🚀 Test Event"  # Exact format


def test_build_digest_blocks_with_events(sample_events: list[Event]) -> None:
    """Test building digest blocks with events."""
    # Arrange
    events = sample_events[:3]  # Use first 3 events

    # Act
    blocks = build_digest_blocks(events, "13.10.2025", "Europe/Amsterdam")

    # Assert
    assert len(blocks) > 0
    assert blocks[0]["type"] == "header"
    assert "События 13.10.2025" in blocks[0]["text"]["text"]
    assert any(block["type"] == "divider" for block in blocks)
    assert any(block["type"] == "context" for block in blocks)


def test_build_digest_blocks_empty() -> None:
    """Test building digest blocks with no events."""
    # Arrange
    events: list[Event] = []

    # Act
    blocks = build_digest_blocks(events, "13.10.2025", "Europe/Amsterdam")

    # Assert
    assert len(blocks) > 0
    assert blocks[0]["type"] == "header"
    assert any("Нет новых событий" in str(block) for block in blocks)


def test_chunk_blocks_under_limit() -> None:
    """Test chunking blocks under the limit."""
    # Arrange
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": f"Block {i}"}} for i in range(10)]

    # Act
    chunks = chunk_blocks(blocks, max_blocks=50)

    # Assert
    assert len(chunks) == 1
    assert len(chunks[0]) == 10


def test_chunk_blocks_over_limit() -> None:
    """Test chunking blocks over the limit."""
    # Arrange
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": f"Block {i}"}} for i in range(100)]

    # Act
    chunks = chunk_blocks(blocks, max_blocks=50)

    # Assert
    assert len(chunks) == 2
    assert len(chunks[0]) == 50
    assert len(chunks[1]) == 50


def test_publish_digest_use_case_dry_run(
    sample_events: list[Event], mock_settings: Settings
) -> None:
    """Test publish digest in dry-run mode."""
    # Arrange
    mock_slack_client = Mock(spec=SlackClient)
    mock_repository = Mock(spec=SQLiteRepository)
    mock_repository.get_events_in_window_filtered.return_value = sample_events[:3]

    # Act
    result = publish_digest_use_case(
        slack_client=mock_slack_client,
        repository=mock_repository,
        settings=mock_settings,
        dry_run=True,
    )

    # Assert
    assert result.events_included == 3
    assert result.messages_posted == 0
    assert result.channel == mock_settings.slack_digest_channel_id
    mock_slack_client.post_message.assert_not_called()


def test_publish_digest_use_case_with_posting(
    sample_events: list[Event], mock_settings: Settings
) -> None:
    """Test publish digest with actual posting."""
    # Arrange
    mock_slack_client = Mock(spec=SlackClient)
    mock_slack_client.post_message.return_value = "1234567890.123456"
    mock_repository = Mock(spec=SQLiteRepository)
    mock_repository.get_events_in_window_filtered.return_value = sample_events[:3]

    # Act
    result = publish_digest_use_case(
        slack_client=mock_slack_client,
        repository=mock_repository,
        settings=mock_settings,
        dry_run=False,
    )

    # Assert
    assert result.events_included == 3
    assert result.messages_posted == 1
    mock_slack_client.post_message.assert_called_once()


def test_publish_digest_use_case_confidence_filter(
    sample_events: list[Event], mock_settings: Settings
) -> None:
    """Test publish digest with confidence filtering."""
    # Arrange
    mock_slack_client = Mock(spec=SlackClient)
    mock_repository = Mock(spec=SQLiteRepository)
    # Only return events with confidence >= 0.7
    filtered_events = [e for e in sample_events if e.confidence >= 0.7]
    mock_repository.get_events_in_window_filtered.return_value = filtered_events

    # Act
    result = publish_digest_use_case(
        slack_client=mock_slack_client,
        repository=mock_repository,
        settings=mock_settings,
        min_confidence=0.7,
        dry_run=True,
    )

    # Assert
    assert result.events_included <= len(filtered_events)
    mock_repository.get_events_in_window_filtered.assert_called_once()
    call_args = mock_repository.get_events_in_window_filtered.call_args
    assert call_args.kwargs["min_confidence"] == 0.7


def test_publish_digest_use_case_max_events_limit(
    sample_events: list[Event], mock_settings: Settings
) -> None:
    """Test publish digest with max events limit."""
    # Arrange
    mock_slack_client = Mock(spec=SlackClient)
    mock_repository = Mock(spec=SQLiteRepository)
    mock_repository.get_events_in_window_filtered.return_value = sample_events

    # Act
    result = publish_digest_use_case(
        slack_client=mock_slack_client,
        repository=mock_repository,
        settings=mock_settings,
        max_events=3,
        dry_run=True,
    )

    # Assert
    assert result.events_included == 3


def test_publish_digest_use_case_unlimited_events(
    sample_events: list[Event], mock_settings: Settings
) -> None:
    """Test publish digest with no max events limit."""
    # Arrange
    mock_slack_client = Mock(spec=SlackClient)
    mock_repository = Mock(spec=SQLiteRepository)
    mock_repository.get_events_in_window_filtered.return_value = sample_events

    # Act
    result = publish_digest_use_case(
        slack_client=mock_slack_client,
        repository=mock_repository,
        settings=mock_settings,
        max_events=None,
        dry_run=True,
    )

    # Assert
    assert result.events_included == len(sample_events)


def test_publish_digest_use_case_custom_channel(
    sample_events: list[Event], mock_settings: Settings
) -> None:
    """Test publish digest with custom channel."""
    # Arrange
    mock_slack_client = Mock(spec=SlackClient)
    mock_repository = Mock(spec=SQLiteRepository)
    mock_repository.get_events_in_window_filtered.return_value = sample_events[:3]
    custom_channel = "C12345678"

    # Act
    result = publish_digest_use_case(
        slack_client=mock_slack_client,
        repository=mock_repository,
        settings=mock_settings,
        target_channel=custom_channel,
        dry_run=True,
    )

    # Assert
    assert result.channel == custom_channel


def test_publish_digest_use_case_uses_settings_defaults(
    sample_events: list[Event], mock_settings: Settings
) -> None:
    """Test that publish digest uses settings defaults when parameters not provided."""
    # Arrange
    mock_slack_client = Mock(spec=SlackClient)
    mock_repository = Mock(spec=SQLiteRepository)
    mock_repository.get_events_in_window_filtered.return_value = sample_events

    # Act
    result = publish_digest_use_case(
        slack_client=mock_slack_client,
        repository=mock_repository,
        settings=mock_settings,
        dry_run=True,
    )

    # Assert
    call_args = mock_repository.get_events_in_window_filtered.call_args
    assert call_args.kwargs["min_confidence"] == mock_settings.digest_min_confidence
    assert result.events_included <= (mock_settings.digest_max_events or len(sample_events))

