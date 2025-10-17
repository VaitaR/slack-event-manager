"""Unit tests for publish digest use case.

Tests confidence filtering, max events limit, category sorting, and block building.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import pytz

from src.adapters.slack_client import SlackClient
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import Settings
from src.domain.models import (
    ActionType,
    ChangeType,
    Environment,
    Event,
    EventCategory,
    EventStatus,
    TimeSource,
)
from src.use_cases.publish_digest import (
    build_digest_blocks,
    build_event_block,
    chunk_blocks,
    format_event_date,
    get_confidence_icon,
    publish_digest_use_case,
    sort_events_for_digest,
)


def create_test_event(
    category: EventCategory = EventCategory.PRODUCT,
    confidence: float = 0.95,
    importance: int = 75,
    event_date: datetime | None = None,
    title_suffix: str = "",
) -> Event:
    """Helper to create test event with new structure."""
    if event_date is None:
        event_date = datetime(2025, 10, 13, 10, 0, 0, tzinfo=pytz.UTC)

    return Event(
        event_id=uuid4(),
        message_id=f"msg_{uuid4().hex[:8]}",
        source_channels=["test-channel"],
        extracted_at=datetime.utcnow(),
        action=ActionType.LAUNCH,
        object_name_raw=f"Test Feature {title_suffix}",
        category=category,
        status=EventStatus.COMPLETED,
        change_type=ChangeType.LAUNCH,
        environment=Environment.PROD,
        actual_start=event_date,
        time_source=TimeSource.EXPLICIT,
        time_confidence=0.9,
        summary=f"Test feature {title_suffix}",
        confidence=confidence,
        importance=importance,
        cluster_key=f"cluster_{uuid4().hex[:8]}",
        dedup_key=f"dedup_{uuid4().hex[:8]}",
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
        create_test_event(
            category=EventCategory.PRODUCT,
            confidence=0.95,
            importance=80,
            event_date=base_date,
            title_suffix="v2.0",
        ),
        create_test_event(
            category=EventCategory.RISK,
            confidence=0.85,
            importance=85,
            event_date=base_date + timedelta(hours=1),
            title_suffix="Security",
        ),
        create_test_event(
            category=EventCategory.PROCESS,
            confidence=0.75,
            importance=70,
            event_date=base_date + timedelta(hours=2),
            title_suffix="Process",
        ),
        create_test_event(
            category=EventCategory.MARKETING,
            confidence=0.65,
            importance=65,
            event_date=base_date + timedelta(hours=3),
            title_suffix="Campaign",
        ),
        create_test_event(
            category=EventCategory.ORG,
            confidence=0.55,
            importance=55,
            event_date=base_date + timedelta(hours=4),
            title_suffix="Restructure",
        ),
        create_test_event(
            category=EventCategory.UNKNOWN,
            confidence=0.45,
            importance=45,
            event_date=base_date + timedelta(hours=5),
            title_suffix="Unknown",
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
        time_i = (
            sorted_events[i].actual_start
            or sorted_events[i].actual_end
            or sorted_events[i].planned_start
            or sorted_events[i].planned_end
            or sorted_events[i].extracted_at
        )
        time_next = (
            sorted_events[i + 1].actual_start
            or sorted_events[i + 1].actual_end
            or sorted_events[i + 1].planned_start
            or sorted_events[i + 1].planned_end
            or sorted_events[i + 1].extracted_at
        )
        assert time_i <= time_next


def test_sort_events_for_digest_by_category(sample_events: list[Event]) -> None:
    """Test sorting events by category priority."""
    # Arrange
    base_date = datetime(2025, 10, 13, 10, 0, 0, tzinfo=pytz.UTC)
    same_date_events = [
        create_test_event(
            event_date=base_date,
            category=EventCategory.MARKETING,
            title_suffix="Marketing",
            confidence=0.8,
        ),
        create_test_event(
            event_date=base_date,
            category=EventCategory.PRODUCT,
            title_suffix="Product",
            confidence=0.8,
        ),
        create_test_event(
            event_date=base_date,
            category=EventCategory.RISK,
            title_suffix="Risk",
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
    """Test sorting events by importance when date and category are same."""
    # Arrange
    base_date = datetime(2025, 10, 13, 10, 0, 0, tzinfo=pytz.UTC)
    same_category_events = [
        create_test_event(
            event_date=base_date,
            category=EventCategory.PRODUCT,
            title_suffix="Low",
            confidence=0.6,
            importance=60,
        ),
        create_test_event(
            event_date=base_date,
            category=EventCategory.PRODUCT,
            title_suffix="High",
            confidence=0.9,
            importance=90,
        ),
    ]

    # Act
    sorted_events = sort_events_for_digest(same_category_events)

    # Assert - higher importance first
    assert sorted_events[0].importance == 90
    assert sorted_events[1].importance == 60


def test_build_event_block() -> None:
    """Test building event block with compact format."""
    # Arrange
    event = create_test_event(
        category=EventCategory.PRODUCT,
        confidence=0.85,
        importance=75,
        event_date=datetime(2025, 10, 13, 10, 0, 0, tzinfo=pytz.UTC),
        title_suffix="Test",
    )

    # Act
    block = build_event_block(event, "Europe/Amsterdam")

    # Assert - Compact format: only emoji + title
    assert block["type"] == "section"
    assert "text" in block
    assert block["text"]["type"] == "mrkdwn"
    assert "🚀" in block["text"]["text"]  # Product category emoji
    assert "Launch: Test Feature Test" in block["text"]["text"]


def test_build_digest_blocks_with_events(sample_events: list[Event]) -> None:
    """Test building digest blocks with events."""
    # Arrange
    events = sample_events[:3]  # Use first 3 events

    # Act
    blocks = build_digest_blocks(events, "13.10.2025", "Europe/Amsterdam")

    # Assert
    assert len(blocks) > 0
    assert blocks[0]["type"] == "header"
    assert "Events 13.10.2025" in blocks[0]["text"]["text"]
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
    assert any("No new events" in str(block) for block in blocks)


def test_chunk_blocks_under_limit() -> None:
    """Test chunking blocks under the limit."""
    # Arrange
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"Block {i}"}}
        for i in range(10)
    ]

    # Act
    chunks = chunk_blocks(blocks, max_blocks=50)

    # Assert
    assert len(chunks) == 1
    assert len(chunks[0]) == 10


def test_chunk_blocks_over_limit() -> None:
    """Test chunking blocks over the limit."""
    # Arrange
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"Block {i}"}}
        for i in range(100)
    ]

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
    assert result.events_included <= (
        mock_settings.digest_max_events or len(sample_events)
    )
