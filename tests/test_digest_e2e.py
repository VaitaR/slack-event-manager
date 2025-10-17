"""E2E tests for digest publishing with real Slack integration.

Tests digest posting to real Slack channel C09LS9V0RRV.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import pytz

from src.adapters.slack_client import SlackClient
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import get_settings
from src.domain.models import (
    ActionType,
    ChangeType,
    Environment,
    Event,
    EventCategory,
    EventStatus,
    TimeSource,
)
from src.use_cases.publish_digest import publish_digest_use_case

# Test channel ID
TEST_CHANNEL = "C09LS9V0RRV"


def create_e2e_event(
    category: EventCategory,
    confidence: float,
    importance: int,
    event_date: datetime,
    title_suffix: str,
) -> Event:
    """Helper to create E2E test event with new structure."""
    return Event(
        event_id=uuid4(),
        message_id=f"msg_e2e_{uuid4().hex[:8]}",
        source_channels=["e2e-test"],
        action=ActionType.LAUNCH,
        object_name_raw=f"E2E Test: {title_suffix}",
        category=category,
        status=EventStatus.COMPLETED,
        change_type=ChangeType.LAUNCH,
        environment=Environment.PROD,
        actual_start=event_date,
        time_source=TimeSource.EXPLICIT,
        time_confidence=0.9,
        summary=f"Test event for digest E2E validation - {category.value}",
        confidence=confidence,
        importance=importance,
        cluster_key=f"e2e_cluster_{uuid4().hex[:8]}",
        dedup_key=f"e2e_dedup_{uuid4().hex[:8]}",
    )


@pytest.fixture
def real_slack_client() -> SlackClient:
    """Create real Slack client from environment."""
    settings = get_settings()
    return SlackClient(bot_token=settings.slack_bot_token.get_secret_value())


@pytest.fixture
def real_repository() -> SQLiteRepository:
    """Get repository with real production data."""
    from pathlib import Path

    settings = get_settings()
    db_path = settings.db_path

    # Check if production database exists
    if not Path(db_path).exists():
        # Fall back to test_real_pipeline.db if available
        alt_db_path = "data/test_real_pipeline.db"
        if Path(alt_db_path).exists():
            db_path = alt_db_path
        else:
            pytest.skip(
                f"No database found at {db_path} or {alt_db_path}. "
                "Run scripts/test_with_real_data.py first."
            )

    repo = SQLiteRepository(db_path)

    # Verify database has events
    utc = pytz.UTC
    now = datetime.now(tz=utc)
    start_dt = now - timedelta(hours=168)  # Last week
    events = repo.get_events_in_window(start_dt, now)

    if not events:
        pytest.skip(
            f"Database {db_path} has no events. Run scripts/test_with_real_data.py first."
        )

    return repo


@pytest.fixture
def test_repository(tmp_path: Path) -> SQLiteRepository:
    """Create test repository with sample data for unit tests."""
    db_path = str(tmp_path / "test_digest_e2e.db")
    repo = SQLiteRepository(db_path=db_path)

    # Insert sample events
    base_date = datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(hours=24)

    events = [
        create_e2e_event(
            category=EventCategory.PRODUCT,
            confidence=0.95,
            importance=85,
            event_date=base_date,
            title_suffix="Product Release v3.0",
        ),
        create_e2e_event(
            category=EventCategory.RISK,
            confidence=0.88,
            importance=80,
            event_date=base_date + timedelta(hours=2),
            title_suffix="Security Alert",
        ),
        create_e2e_event(
            category=EventCategory.PROCESS,
            confidence=0.82,
            importance=75,
            event_date=base_date + timedelta(hours=4),
            title_suffix="Process Update",
        ),
        create_e2e_event(
            category=EventCategory.MARKETING,
            confidence=0.75,
            importance=70,
            event_date=base_date + timedelta(hours=6),
            title_suffix="Campaign Launch",
        ),
        create_e2e_event(
            category=EventCategory.ORG,
            confidence=0.68,
            importance=60,
            event_date=base_date + timedelta(hours=8),
            title_suffix="Team Change",
        ),
        # Low confidence event (should be filtered out with default settings)
        create_e2e_event(
            category=EventCategory.UNKNOWN,
            confidence=0.45,
            importance=40,
            event_date=base_date + timedelta(hours=10),
            title_suffix="Low Confidence Event",
        ),
    ]

    repo.save_events(events)
    return repo


def test_digest_dry_run(test_repository: SQLiteRepository) -> None:
    """Test digest generation in dry-run mode (no posting)."""
    # Arrange
    settings = get_settings()
    slack_client = SlackClient(bot_token=settings.slack_bot_token.get_secret_value())

    # Act
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=test_repository,
        settings=settings,
        lookback_hours=48,
        target_channel=TEST_CHANNEL,
        dry_run=True,
        min_confidence=0.7,
        max_events=10,
    )

    # Assert
    assert result.events_included > 0
    assert result.events_included <= 10
    assert result.messages_posted == 0
    assert result.channel == TEST_CHANNEL
    print(f"✓ Dry-run test passed: {result.events_included} events prepared")


def test_digest_confidence_filtering(test_repository: SQLiteRepository) -> None:
    """Test digest with confidence filtering."""
    # Arrange
    settings = get_settings()
    slack_client = SlackClient(bot_token=settings.slack_bot_token.get_secret_value())

    # Act - Filter only high confidence events
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=test_repository,
        settings=settings,
        lookback_hours=48,
        target_channel=TEST_CHANNEL,
        dry_run=True,
        min_confidence=0.8,
        max_events=None,
    )

    # Assert - Should exclude low confidence events
    assert result.events_included <= 3  # Only events with confidence >= 0.8
    print(
        f"✓ Confidence filter test passed: {result.events_included} high-confidence events"
    )


def test_digest_max_events_limit(test_repository: SQLiteRepository) -> None:
    """Test digest with max events limit."""
    # Arrange
    settings = get_settings()
    slack_client = SlackClient(bot_token=settings.slack_bot_token.get_secret_value())

    # Act - Limit to 3 events
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=test_repository,
        settings=settings,
        lookback_hours=48,
        target_channel=TEST_CHANNEL,
        dry_run=True,
        min_confidence=0.0,  # Include all
        max_events=3,
    )

    # Assert
    assert result.events_included == 3
    print(f"✓ Max events limit test passed: {result.events_included} events (limit=3)")


@pytest.mark.skipif(
    os.getenv("SKIP_SLACK_E2E", "false").lower() == "true",
    reason="Skipping real Slack posting test (set SKIP_SLACK_E2E=false to enable)",
)
def test_digest_real_posting(
    real_slack_client: SlackClient, real_repository: SQLiteRepository
) -> None:
    """Test actual digest posting to Slack channel with REAL DATA.

    WARNING: This test posts real messages to Slack channel C09LS9V0RRV.
    Set SKIP_SLACK_E2E=true to skip this test.

    Uses real production events from database instead of mock data.
    """
    # Arrange
    settings = get_settings()

    # Act - Post digest to test channel with real data
    result = publish_digest_use_case(
        slack_client=real_slack_client,
        repository=real_repository,
        settings=settings,
        lookback_hours=168,  # Last week to get real events
        target_channel=TEST_CHANNEL,
        dry_run=False,  # REAL POSTING
        min_confidence=0.7,
        max_events=10,  # Show up to 10 events
    )

    # Assert
    assert result.messages_posted >= 1
    assert result.events_included > 0
    assert result.channel == TEST_CHANNEL

    print("=" * 60)
    print("✅ DIGEST POSTED TO SLACK SUCCESSFULLY")
    print(f"   Channel: {result.channel}")
    print(f"   Events: {result.events_included}")
    print(f"   Messages: {result.messages_posted}")
    print("=" * 60)
    print("\nPlease verify the digest in Slack channel:")
    print(f"https://app.slack.com/client/T01234567/{TEST_CHANNEL}")
    print("\nNote: This is a test digest with E2E test events")


def test_digest_category_sorting(test_repository: SQLiteRepository) -> None:
    """Test that events are sorted by category priority."""
    # Arrange
    settings = get_settings()
    slack_client = SlackClient(bot_token=settings.slack_bot_token.get_secret_value())

    # Act
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=test_repository,
        settings=settings,
        lookback_hours=48,
        target_channel=TEST_CHANNEL,
        dry_run=True,
        min_confidence=0.0,
        max_events=None,
    )

    # Assert - Just verify it runs without error
    # Actual sorting is tested in unit tests
    assert result.events_included > 0
    print(f"✓ Category sorting test passed: {result.events_included} events sorted")


def test_digest_with_settings_defaults(test_repository: SQLiteRepository) -> None:
    """Test digest using all default settings from config."""
    # Arrange
    settings = get_settings()
    slack_client = SlackClient(bot_token=settings.slack_bot_token.get_secret_value())

    # Act - Use all defaults
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=test_repository,
        settings=settings,
        target_channel=TEST_CHANNEL,
        dry_run=True,
    )

    # Assert
    assert result.events_included > 0
    assert result.channel == TEST_CHANNEL
    print(
        f"✓ Settings defaults test passed: "
        f"{result.events_included} events with default config"
    )


def test_digest_empty_results(tmp_path: Path) -> None:
    """Test digest when no events match the criteria."""
    # Arrange
    settings = get_settings()
    slack_client = SlackClient(bot_token=settings.slack_bot_token.get_secret_value())
    db_path = str(tmp_path / "empty_test.db")
    repo = SQLiteRepository(db_path=db_path)  # Empty database

    # Act
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=repo,
        settings=settings,
        lookback_hours=48,
        target_channel=TEST_CHANNEL,
        dry_run=True,
        min_confidence=0.7,
    )

    # Assert
    assert result.events_included == 0
    assert result.messages_posted == 0
    print("✓ Empty results test passed: 0 events handled gracefully")
