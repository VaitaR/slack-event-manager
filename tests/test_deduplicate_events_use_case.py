"""Tests for deduplicate_events_use_case."""

from datetime import datetime
from unittest.mock import Mock

import pytz
from structlog.testing import capture_logs

from src.config.settings import Settings
from src.domain.models import Event, MessageSource
from src.use_cases.deduplicate_events import deduplicate_events_use_case


def create_mock_settings() -> Settings:
    """Create mock settings for testing."""
    return Settings(
        llm_model="gpt-5-nano",
        llm_temperature=1.0,
        llm_timeout_seconds=30,
        daily_budget_usd=10.0,
        db_path=":memory:",
        database_type="sqlite",
        slack_bot_token="test-token",
        slack_digest_channel_id="C1234567890",
        processing_tz_default="Europe/Amsterdam",
        dedup_date_window_hours=48,
        dedup_title_similarity=0.8,
        logging_level="INFO",
    )


def create_test_event(
    message_id: str = "test_msg",
    object_name: str = "Test Event",
    links: list[str] | None = None,
    source_id: MessageSource = MessageSource.SLACK,
) -> Event:
    """Create test event with computed cluster_key and dedup_key."""
    from uuid import uuid4

    # Create a minimal event first to compute keys
    temp_event = Event(
        event_id=uuid4(),
        message_id=message_id,
        source_channels=["#test"],
        extracted_at=datetime.utcnow(),
        action="Launch",
        object_name_raw=object_name,
        category="product",
        status="completed",
        change_type="launch",
        environment="prod",
        actual_start=datetime(2025, 10, 15, 10, 0, tzinfo=pytz.UTC),
        time_source="explicit",
        time_confidence=0.9,
        summary="Test event",
        confidence=0.8,
        importance=75,
        links=links or ["https://example.com"],
        source_id=source_id,
        cluster_key="",  # Will be computed
        dedup_key="",  # Will be computed
    )

    # Compute cluster_key and dedup_key using deduplicator service
    from src.services.deduplicator import generate_cluster_key, generate_dedup_key

    cluster_key = generate_cluster_key(temp_event)
    dedup_key = generate_dedup_key(temp_event)

    # Return complete event with computed keys
    return Event(
        event_id=temp_event.event_id,
        message_id=temp_event.message_id,
        source_channels=temp_event.source_channels,
        extracted_at=temp_event.extracted_at,
        action=temp_event.action,
        object_name_raw=temp_event.object_name_raw,
        category=temp_event.category,
        status=temp_event.status,
        change_type=temp_event.change_type,
        environment=temp_event.environment,
        actual_start=temp_event.actual_start,
        time_source=temp_event.time_source,
        time_confidence=temp_event.time_confidence,
        summary=temp_event.summary,
        confidence=temp_event.confidence,
        importance=temp_event.importance,
        links=temp_event.links,
        source_id=temp_event.source_id,
        cluster_key=cluster_key,
        dedup_key=dedup_key,
    )


def test_deduplicate_events_empty_list() -> None:
    """Test deduplication with empty event list."""
    repository = Mock()
    repository.query_events.return_value = []

    settings = create_mock_settings()
    result = deduplicate_events_use_case(repository, settings)

    assert result.new_events == 0
    assert result.merged_events == 0
    assert result.total_events == 0
    repository.query_events.assert_called_once()


def test_deduplicate_events_no_merges_needed() -> None:
    """Test deduplication when no events should merge."""
    event1 = create_test_event("msg1", "Event A", ["https://link1.com"])
    event2 = create_test_event("msg2", "Event B", ["https://link2.com"])

    repository = Mock()
    repository.query_events.return_value = [event1, event2]

    settings = create_mock_settings()
    result = deduplicate_events_use_case(repository, settings)

    # No merges: 2 initial events, 2 final events, 0 merged
    assert result.new_events == 2
    assert result.merged_events == 0
    assert result.total_events == 2

    # Verify invariant: new_events + merged_events == initial_count
    initial_count = 2
    assert result.new_events + result.merged_events == initial_count


def test_deduplicate_events_with_merges() -> None:
    """Test deduplication when events should merge."""
    # Two similar events that should merge
    event1 = create_test_event("msg1", "Release v1.0", ["https://release.com"])
    event2 = create_test_event("msg2", "Release v1.0", ["https://release.com"])

    repository = Mock()
    repository.query_events.return_value = [event1, event2]

    settings = create_mock_settings()
    result = deduplicate_events_use_case(repository, settings)

    # Should merge: 2 initial events, 1 final event, 1 merged
    assert result.new_events == 1
    assert result.merged_events == 1
    assert result.total_events == 1

    # Verify invariant: new_events + merged_events == initial_count
    initial_count = 2
    assert result.new_events + result.merged_events == initial_count


def test_deduplicate_events_complex_scenario() -> None:
    """Test complex scenario with multiple merges."""
    # Create 5 events: 3 should merge into 1, 2 remain unique
    event1 = create_test_event("msg1", "Feature Launch", ["https://feature.com"])
    event2 = create_test_event("msg2", "Feature Launch", ["https://feature.com"])
    event3 = create_test_event("msg3", "Feature Launch", ["https://feature.com"])
    event4 = create_test_event("msg4", "Bug Fix", ["https://bug.com"])
    event5 = create_test_event("msg5", "Security Update", ["https://security.com"])

    repository = Mock()
    repository.query_events.return_value = [event1, event2, event3, event4, event5]

    settings = create_mock_settings()
    result = deduplicate_events_use_case(repository, settings)

    # Expected: 5 initial, 3 final (1 merged group + 2 unique), 2 merged
    assert result.new_events == 3
    assert result.merged_events == 2
    assert result.total_events == 3

    # Verify invariant: new_events + merged_events == initial_count
    initial_count = 5
    assert result.new_events + result.merged_events == initial_count


def test_deduplicate_events_invariant_verification() -> None:
    """Test that the invariant new_events + merged_events == initial_count is always maintained."""
    # Test case 1: No merges
    event1 = create_test_event("msg1", "Event A")
    event2 = create_test_event("msg2", "Event B")

    repository = Mock()
    repository.query_events.return_value = [event1, event2]

    settings = create_mock_settings()
    result = deduplicate_events_use_case(repository, settings)

    initial_count = 2
    assert result.new_events + result.merged_events == initial_count

    # Test case 2: All events merge
    event3 = create_test_event("msg3", "Same Event", ["https://same.com"])
    event4 = create_test_event("msg4", "Same Event", ["https://same.com"])

    repository.query_events.return_value = [event3, event4]
    result = deduplicate_events_use_case(repository, settings)

    initial_count = 2
    assert result.new_events + result.merged_events == initial_count

    # Test case 3: Partial merges
    event5 = create_test_event("msg5", "Feature", ["https://feature.com"])
    event6 = create_test_event("msg6", "Feature", ["https://feature.com"])
    event7 = create_test_event("msg7", "Bug Fix", ["https://bug.com"])

    repository.query_events.return_value = [event5, event6, event7]
    result = deduplicate_events_use_case(repository, settings)

    initial_count = 3
    assert result.new_events + result.merged_events == initial_count


def test_deduplicate_events_source_filtering() -> None:
    """Test deduplication with source filtering."""
    event1 = create_test_event("msg1", "Event A", source_id=MessageSource.SLACK)
    create_test_event("msg2", "Event B", source_id=MessageSource.TELEGRAM)

    repository = Mock()
    # Mock should return only Slack events when source_id filter is applied
    repository.query_events.return_value = [event1]  # Only Slack event

    settings = create_mock_settings()
    result = deduplicate_events_use_case(
        repository, settings, source_id=MessageSource.SLACK
    )

    # Should only process Slack events
    assert result.new_events == 1
    assert result.merged_events == 0
    assert result.total_events == 1

    # Verify source_id was passed to query_events
    repository.query_events.assert_called_once()
    call_args = repository.query_events.call_args[0][0]
    assert call_args.source_id == MessageSource.SLACK.value


def test_deduplicate_events_logs_structured_summary() -> None:
    """Use case should emit structured logs instead of printing to stdout."""

    repository = Mock()
    repository.query_events.return_value = []

    settings = create_mock_settings()

    with capture_logs() as logs:
        deduplicate_events_use_case(repository, settings)

    event_names = {entry["event"] for entry in logs}
    assert "deduplication_started" in event_names
    assert "deduplication_finished" in event_names
