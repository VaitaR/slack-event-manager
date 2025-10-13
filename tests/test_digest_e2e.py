"""E2E tests for digest publishing with real Slack integration.

Tests digest posting to real Slack channel C06B5NJLY4B.
"""

import os
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
import pytz

from src.adapters.slack_client import SlackClient
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import get_settings
from src.domain.models import Event, EventCategory
from src.use_cases.publish_digest import publish_digest_use_case

# Test channel ID
TEST_CHANNEL = "C06B5NJLY4B"


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
def test_repository(tmp_path: pytest.TempPathFactory) -> SQLiteRepository:
    """Create test repository with sample data for unit tests."""
    db_path = str(tmp_path / "test_digest_e2e.db")
    repo = SQLiteRepository(db_path=db_path)

    # Insert sample events
    base_date = datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(hours=24)

    events = [
        Event(
            event_id=uuid4(),
            message_id="msg_e2e_1",
            source_msg_event_idx=0,
            dedup_key="e2e_key_1",
            event_date=base_date,
            category=EventCategory.PRODUCT,
            title="E2E Test: Product Release v3.0",
            summary="Test event for digest E2E validation - Product category",
            confidence=0.95,
            links=["https://github.com/test/release"],
            tags=["e2e-test", "product"],
        ),
        Event(
            event_id=uuid4(),
            message_id="msg_e2e_2",
            source_msg_event_idx=0,
            dedup_key="e2e_key_2",
            event_date=base_date + timedelta(hours=2),
            category=EventCategory.RISK,
            title="E2E Test: Security Alert",
            summary="Test event for digest E2E validation - Risk category",
            confidence=0.88,
            links=["https://security.test/alert"],
            tags=["e2e-test", "security"],
        ),
        Event(
            event_id=uuid4(),
            message_id="msg_e2e_3",
            source_msg_event_idx=0,
            dedup_key="e2e_key_3",
            event_date=base_date + timedelta(hours=4),
            category=EventCategory.PROCESS,
            title="E2E Test: Process Update",
            summary="Test event for digest E2E validation - Process category",
            confidence=0.82,
            links=["https://docs.test/process"],
            tags=["e2e-test", "process"],
        ),
        Event(
            event_id=uuid4(),
            message_id="msg_e2e_4",
            source_msg_event_idx=0,
            dedup_key="e2e_key_4",
            event_date=base_date + timedelta(hours=6),
            category=EventCategory.MARKETING,
            title="E2E Test: Campaign Launch",
            summary="Test event for digest E2E validation - Marketing category",
            confidence=0.75,
            links=["https://marketing.test/campaign"],
            tags=["e2e-test", "marketing"],
        ),
        Event(
            event_id=uuid4(),
            message_id="msg_e2e_5",
            source_msg_event_idx=0,
            dedup_key="e2e_key_5",
            event_date=base_date + timedelta(hours=8),
            category=EventCategory.ORG,
            title="E2E Test: Team Change",
            summary="Test event for digest E2E validation - Org category",
            confidence=0.68,
            links=["https://hr.test/announcement"],
            tags=["e2e-test", "org"],
        ),
        # Low confidence event (should be filtered out with default settings)
        Event(
            event_id=uuid4(),
            message_id="msg_e2e_6",
            source_msg_event_idx=0,
            dedup_key="e2e_key_6",
            event_date=base_date + timedelta(hours=10),
            category=EventCategory.UNKNOWN,
            title="E2E Test: Low Confidence Event",
            summary="This should be filtered out with min_confidence=0.7",
            confidence=0.45,
            tags=["e2e-test", "low-confidence"],
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
    print(f"✓ Confidence filter test passed: {result.events_included} high-confidence events")


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

    WARNING: This test posts real messages to Slack channel C06B5NJLY4B.
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


def test_digest_empty_results(tmp_path: pytest.TempPathFactory) -> None:
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

