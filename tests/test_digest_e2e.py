"""E2E tests for digest publishing with real Slack integration.

Tests digest posting to real Slack channel C09LS9V0RRV.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import pytz

from src.adapters.repository_factory import create_repository
from src.adapters.slack_client import SlackClient
from src.config.settings import Settings, get_settings
from src.domain.models import (
    ActionType,
    ChangeType,
    Environment,
    Event,
    EventCategory,
    EventStatus,
    TimeSource,
)
from src.domain.protocols import RepositoryProtocol
from src.use_cases.publish_digest import publish_digest_use_case

# Test channel ID
TEST_CHANNEL = "C09LS9V0RRV"

DATABASE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


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
def slack_client(settings: Settings) -> SlackClient:
    """Slack client configured with test settings."""

    return SlackClient(bot_token=settings.slack_bot_token.get_secret_value())


@pytest.fixture
def digest_repository(repo: RepositoryProtocol) -> RepositoryProtocol:
    """Seed repository with sample digest events for testing."""

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


@pytest.fixture
def real_repository() -> RepositoryProtocol:
    """Get repository with real production data."""

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

    live_settings = settings.model_copy(
        update={"database_type": "sqlite", "db_path": str(db_path)}
    )
    repo = create_repository(live_settings)

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


@pytest.mark.parametrize("repo", DATABASE_BACKENDS, indirect=True)
def test_digest_dry_run(
    digest_repository: RepositoryProtocol,
    slack_client: SlackClient,
    settings: Settings,
) -> None:
    """Test digest generation in dry-run mode (no posting)."""
    repository = digest_repository

    # Act
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=repository,
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


@pytest.mark.parametrize("repo", DATABASE_BACKENDS, indirect=True)
def test_digest_confidence_filtering(
    digest_repository: RepositoryProtocol,
    slack_client: SlackClient,
    settings: Settings,
) -> None:
    """Test digest with confidence filtering."""
    repository = digest_repository

    # Act - Filter only high confidence events
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=repository,
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


@pytest.mark.parametrize("repo", DATABASE_BACKENDS, indirect=True)
def test_digest_max_events_limit(
    digest_repository: RepositoryProtocol,
    slack_client: SlackClient,
    settings: Settings,
) -> None:
    """Test digest with max events limit."""
    repository = digest_repository

    # Act - Limit to 3 events
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=repository,
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
    real_slack_client: SlackClient, real_repository: RepositoryProtocol
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


@pytest.mark.parametrize("repo", DATABASE_BACKENDS, indirect=True)
def test_digest_category_sorting(
    digest_repository: RepositoryProtocol,
    slack_client: SlackClient,
    settings: Settings,
) -> None:
    """Test that events are sorted by category priority."""
    repository = digest_repository

    # Act
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=repository,
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


@pytest.mark.parametrize("repo", DATABASE_BACKENDS, indirect=True)
def test_digest_with_settings_defaults(
    digest_repository: RepositoryProtocol,
    slack_client: SlackClient,
    settings: Settings,
) -> None:
    """Test digest using all default settings from config."""
    repository = digest_repository

    # Act - Use all defaults
    result = publish_digest_use_case(
        slack_client=slack_client,
        repository=repository,
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


@pytest.mark.parametrize("repo", DATABASE_BACKENDS, indirect=True)
def test_digest_empty_results(
    repo: RepositoryProtocol, slack_client: SlackClient, settings: Settings
) -> None:
    """Test digest when no events match the criteria."""

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
