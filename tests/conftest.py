"""Pytest configuration and shared fixtures."""

import os
from datetime import datetime
from unittest.mock import Mock

import pytest
import pytz

from src.domain.models import (
    CandidateStatus,
    ChannelConfig,
    Event,
    EventCandidate,
    EventCategory,
    ScoringFeatures,
    SlackMessage,
)


@pytest.fixture
def sample_slack_message() -> SlackMessage:
    """Sample Slack message for testing."""
    return SlackMessage(
        message_id="test_msg_123",
        channel="C123456",
        ts="1728000000.123456",
        ts_dt=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        user="U123456",
        is_bot=False,
        subtype=None,
        text="Release v1.0 is scheduled for Oct 15",
        blocks_text="Release v1.0 is scheduled for Oct 15",
        text_norm="release v1.0 is scheduled for oct 15",
        links_raw=["https://github.com/org/repo/issues/42"],
        links_norm=["https://github.com/org/repo/issues/42"],
        anchors=["org/repo#42"],
        reactions={"rocket": 5, "eyes": 2},
        reply_count=3,
        ingested_at=datetime(2025, 10, 10, 10, 5, tzinfo=pytz.UTC),
    )


@pytest.fixture
def sample_channel_config() -> ChannelConfig:
    """Sample channel configuration."""
    return ChannelConfig(
        channel_id="C123456",
        channel_name="releases",
        threshold_score=15.0,
        whitelist_keywords=["release", "deploy", "launch"],
        keyword_weight=10.0,
        mention_weight=8.0,
        reply_weight=5.0,
        reaction_weight=3.0,
        anchor_weight=4.0,
        link_weight=2.0,
        file_weight=3.0,
        bot_penalty=-15.0,
    )


@pytest.fixture
def sample_scoring_features() -> ScoringFeatures:
    """Sample scoring features."""
    return ScoringFeatures(
        has_keywords=True,
        keyword_count=1,
        has_mention=False,
        reply_count=3,
        reaction_count=7,
        anchor_count=1,
        link_count=1,
        has_files=False,
        is_bot=False,
        channel_name="releases",
    )


@pytest.fixture
def sample_event_candidate() -> EventCandidate:
    """Sample event candidate."""
    return EventCandidate(
        message_id="test_msg_123",
        channel="C123456",
        ts_dt=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        text_norm="release v1.0 scheduled for oct 15",
        links_norm=["https://github.com/org/repo/issues/42"],
        anchors=["org/repo#42"],
        score=25.0,
        status=CandidateStatus.NEW,
        features=ScoringFeatures(
            has_keywords=True,
            keyword_count=1,
            has_mention=False,
            reply_count=3,
            reaction_count=7,
            anchor_count=1,
            link_count=1,
            has_files=False,
            is_bot=False,
            channel_name="releases",
        ),
    )


@pytest.fixture
def sample_event() -> Event:
    """Sample event for testing."""
    return Event(
        message_id="test_msg_123",
        source_msg_event_idx=0,
        dedup_key="abc123def456",
        event_date=datetime(2025, 10, 15, 8, 0, tzinfo=pytz.UTC),
        event_end=None,
        category=EventCategory.PRODUCT,
        title="Release v1.0",
        summary="Major release with new features",
        impact_area=["payments", "wallet"],
        tags=["release", "v1.0"],
        links=["https://github.com/org/repo/issues/42"],
        anchors=["org/repo#42"],
        confidence=0.9,
        source_channels=["#releases"],
        ingested_at=datetime(2025, 10, 10, 10, 5, tzinfo=pytz.UTC),
    )


@pytest.fixture
def mock_slack_client() -> Mock:
    """Mock Slack client."""
    mock = Mock()
    mock.fetch_messages.return_value = []
    mock.get_user_info.return_value = {"real_name": "Test User", "name": "testuser"}
    mock.post_message.return_value = "1728000000.123456"
    return mock


@pytest.fixture
def mock_llm_client() -> Mock:
    """Mock LLM client."""
    mock = Mock()
    return mock


@pytest.fixture
def mock_repository() -> Mock:
    """Mock repository."""
    mock = Mock()
    mock.save_messages.return_value = 1
    mock.get_watermark.return_value = None
    mock.update_watermark.return_value = None
    mock.get_new_messages_for_candidates.return_value = []
    mock.save_candidates.return_value = 1
    mock.get_candidates_for_extraction.return_value = []
    mock.update_candidate_status.return_value = None
    mock.save_events.return_value = 1
    mock.get_events_in_window.return_value = []
    mock.save_llm_call.return_value = None
    mock.get_daily_llm_cost.return_value = 0.0
    mock.get_cached_llm_response.return_value = None
    return mock


@pytest.fixture
def postgres_available() -> bool:
    """Check if PostgreSQL is available for testing."""
    return (
        os.environ.get("POSTGRES_PASSWORD") is not None
        and os.environ.get("TEST_POSTGRES", "0") == "1"
    )


@pytest.fixture
def postgres_test_db():
    """Setup and teardown PostgreSQL test database.

    Requires:
        - POSTGRES_PASSWORD environment variable
        - TEST_POSTGRES=1 environment variable
        - PostgreSQL running on localhost:5432

    Yields:
        PostgresRepository instance with clean test database
    """
    # Check if PostgreSQL testing is enabled
    if not os.environ.get("POSTGRES_PASSWORD"):
        pytest.skip("POSTGRES_PASSWORD not set - skipping PostgreSQL tests")
    if os.environ.get("TEST_POSTGRES", "0") != "1":
        pytest.skip("TEST_POSTGRES=1 not set - skipping PostgreSQL tests")

    from src.adapters.postgres_repository import PostgresRepository

    # Connect to test database
    test_db_name = "slack_events_test"
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        pytest.skip("POSTGRES_PASSWORD not set")

    repo = PostgresRepository(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=test_db_name,
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=password,
        minconn=1,
        maxconn=2,
    )

    # Clean up: drop all tables before test
    with repo._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DROP TABLE IF EXISTS ingestion_state CASCADE;
                DROP TABLE IF EXISTS channel_watermarks CASCADE;
                DROP TABLE IF EXISTS llm_calls CASCADE;
                DROP TABLE IF EXISTS events CASCADE;
                DROP TABLE IF EXISTS event_candidates CASCADE;
                DROP TABLE IF EXISTS raw_slack_messages CASCADE;
            """)
            conn.commit()

    # Run migrations
    import subprocess

    subprocess.run(
        ["alembic", "upgrade", "head"],
        check=True,
        capture_output=True,
    )

    yield repo

    # Cleanup after test
    with repo._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DROP TABLE IF EXISTS ingestion_state CASCADE;
                DROP TABLE IF EXISTS channel_watermarks CASCADE;
                DROP TABLE IF EXISTS llm_calls CASCADE;
                DROP TABLE IF EXISTS events CASCADE;
                DROP TABLE IF EXISTS event_candidates CASCADE;
                DROP TABLE IF EXISTS raw_slack_messages CASCADE;
            """)
            conn.commit()

    repo.close()
