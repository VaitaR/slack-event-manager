"""Tests for dashboard data retrieval use cases."""

from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import pytz

from src.domain.models import (
    CandidateStatus,
    Event,
    EventCandidate,
    EventStatus,
    MessageSource,
    ScoringFeatures,
    SlackMessage,
)
from src.use_cases.dashboard_queries import (
    fetch_recent_candidates,
    fetch_recent_events,
    fetch_recent_messages,
)


def _make_slack_message(message_id: str) -> SlackMessage:
    """Create a sample SlackMessage for tests."""

    return SlackMessage(
        message_id=message_id,
        channel="C123",
        ts="123.456",
        ts_dt=datetime(2025, 10, 15, 10, 0, tzinfo=pytz.UTC),
        user="U123",
        text="hello",
        text_norm="hello",
        blocks_text="hello",
        links_raw=[],
        links_norm=[],
        anchors=[],
    )


def _make_candidate(message_id: str) -> EventCandidate:
    """Create a sample candidate for tests."""

    return EventCandidate(
        message_id=message_id,
        channel="C123",
        ts_dt=datetime(2025, 10, 15, 10, 0, tzinfo=pytz.UTC),
        text_norm="candidate",
        links_norm=[],
        anchors=[],
        score=0.9,
        status=CandidateStatus.NEW,
        features=ScoringFeatures(),
    )


def _make_event(message_id: str) -> Event:
    """Create a sample event for tests."""

    return Event(
        event_id=uuid4(),
        message_id=message_id,
        source_channels=["C123"],
        extracted_at=datetime(2025, 10, 15, 11, 0, tzinfo=pytz.UTC),
        action="Launch",
        object_name_raw="Test",
        category="product",
        status=EventStatus.PLANNED,
        change_type="launch",
        environment="prod",
        time_source="explicit",
        time_confidence=0.9,
        summary="summary",
        confidence=0.9,
        importance=5,
        links=[],
        cluster_key="cluster",
        dedup_key="dedup",
        source_id=MessageSource.SLACK,
    )


def test_fetch_recent_messages_delegates_to_repository() -> None:
    """Use case should fetch messages through repository abstraction."""

    repo = Mock()
    expected = [_make_slack_message("m1")]
    repo.get_recent_slack_messages.return_value = expected

    result = fetch_recent_messages(repo, limit=5)

    repo.get_recent_slack_messages.assert_called_once_with(limit=5)
    assert result == expected


def test_fetch_recent_candidates_delegates_to_repository() -> None:
    """Use case should fetch candidates through repository abstraction."""

    repo = Mock()
    expected = [_make_candidate("m1")]
    repo.get_recent_candidates.return_value = expected

    result = fetch_recent_candidates(repo, limit=10)

    repo.get_recent_candidates.assert_called_once_with(limit=10)
    assert result == expected


def test_fetch_recent_events_delegates_to_repository() -> None:
    """Use case should fetch events through repository abstraction."""

    repo = Mock()
    expected = [_make_event("m1")]
    repo.get_recent_events.return_value = expected

    result = fetch_recent_events(repo, limit=15)

    repo.get_recent_events.assert_called_once_with(limit=15)
    assert result == expected
