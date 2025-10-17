"""Tests for new deduplication logic (cluster_key and dedup_key).

Critical tests for initiative-level and instance-level key generation.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

import pytz

from src.domain.models import (
    ActionType,
    ChangeType,
    Environment,
    Event,
    EventCategory,
    EventStatus,
    TimeSource,
)
from src.services.deduplicator import generate_cluster_key, generate_dedup_key


def create_test_event(**kwargs: Any) -> Event:
    """Create test event with defaults."""
    defaults: dict[str, Any] = {
        "event_id": uuid4(),
        "message_id": "test123",
        "source_channels": ["test"],
        "extracted_at": datetime.utcnow(),
        "action": ActionType.LAUNCH,
        "object_name_raw": "Test Feature",
        "category": EventCategory.PRODUCT,
        "status": EventStatus.COMPLETED,
        "change_type": ChangeType.LAUNCH,
        "environment": Environment.PROD,
        "actual_start": datetime(2025, 10, 15, 10, 0, tzinfo=pytz.UTC),
        "time_source": TimeSource.EXPLICIT,
        "time_confidence": 0.9,
        "summary": "Test",
        "confidence": 0.8,
        "importance": 75,
        "cluster_key": "",
        "dedup_key": "",
    }
    defaults.update(kwargs)
    return Event(**defaults)


def test_cluster_key_same_for_same_initiative() -> None:
    """Test that cluster_key is same for same initiative regardless of status."""
    event1 = create_test_event(
        status=EventStatus.PLANNED,
        planned_start=datetime(2025, 10, 20, tzinfo=pytz.UTC),
    )
    event2 = create_test_event(
        status=EventStatus.STARTED,
        actual_start=datetime(2025, 10, 20, tzinfo=pytz.UTC),
    )

    key1 = generate_cluster_key(event1)
    key2 = generate_cluster_key(event2)

    # Same initiative = same cluster_key
    assert key1 == key2


def test_cluster_key_different_for_different_object() -> None:
    """Test that cluster_key differs for different objects."""
    event1 = create_test_event(object_name_raw="Feature A")
    event2 = create_test_event(object_name_raw="Feature B")

    key1 = generate_cluster_key(event1)
    key2 = generate_cluster_key(event2)

    assert key1 != key2


def test_cluster_key_different_for_different_action() -> None:
    """Test that cluster_key differs for different actions."""
    event1 = create_test_event(action=ActionType.LAUNCH)
    event2 = create_test_event(action=ActionType.DEPLOY)

    key1 = generate_cluster_key(event1)
    key2 = generate_cluster_key(event2)

    assert key1 != key2


def test_cluster_key_with_anchor() -> None:
    """Test that anchor is included in cluster_key."""
    event1 = create_test_event(anchors=["TEST-123"])
    event2 = create_test_event(anchors=["TEST-456"])

    key1 = generate_cluster_key(event1)
    key2 = generate_cluster_key(event2)

    # Different anchors = different cluster
    assert key1 != key2


def test_dedup_key_different_for_different_status() -> None:
    """Test that dedup_key differs for different status of same initiative."""
    event1 = create_test_event(
        status=EventStatus.PLANNED,
        planned_start=datetime(2025, 10, 20, tzinfo=pytz.UTC),
    )
    event2 = create_test_event(
        status=EventStatus.STARTED,
        actual_start=datetime(2025, 10, 20, tzinfo=pytz.UTC),
    )

    key1 = generate_dedup_key(event1)
    key2 = generate_dedup_key(event2)

    # Same initiative, different status = different dedup_key
    assert key1 != key2


def test_dedup_key_different_for_different_time() -> None:
    """Test that dedup_key differs for different times."""
    event1 = create_test_event(
        actual_start=datetime(2025, 10, 20, 10, 0, tzinfo=pytz.UTC)
    )
    event2 = create_test_event(
        actual_start=datetime(2025, 10, 21, 10, 0, tzinfo=pytz.UTC)
    )

    key1 = generate_dedup_key(event1)
    key2 = generate_dedup_key(event2)

    # Different times = different dedup_key
    assert key1 != key2


def test_dedup_key_different_for_different_environment() -> None:
    """Test that dedup_key differs for different environments."""
    event1 = create_test_event(environment=Environment.PROD)
    event2 = create_test_event(environment=Environment.STAGING)

    key1 = generate_dedup_key(event1)
    key2 = generate_dedup_key(event2)

    # Different environments = different dedup_key
    assert key1 != key2


def test_dedup_key_same_for_identical_instance() -> None:
    """Test that dedup_key is same for identical event instances."""
    event1 = create_test_event(
        status=EventStatus.COMPLETED,
        actual_start=datetime(2025, 10, 20, 10, 0, tzinfo=pytz.UTC),
        environment=Environment.PROD,
    )
    event2 = create_test_event(
        status=EventStatus.COMPLETED,
        actual_start=datetime(2025, 10, 20, 10, 0, tzinfo=pytz.UTC),
        environment=Environment.PROD,
    )

    key1 = generate_dedup_key(event1)
    key2 = generate_dedup_key(event2)

    # Identical instances = same dedup_key
    assert key1 == key2


def test_dedup_key_includes_cluster_key() -> None:
    """Test that events with different cluster have different dedup keys."""
    event1 = create_test_event(object_name_raw="Feature A")
    event2 = create_test_event(object_name_raw="Feature B")

    key1 = generate_dedup_key(event1)
    key2 = generate_dedup_key(event2)

    # Different clusters = different dedup_keys
    assert key1 != key2


def test_lifecycle_tracking_scenario() -> None:
    """Test real lifecycle scenario: planned -> started -> completed."""
    # Same feature, different lifecycle stages
    base_kwargs = {
        "action": ActionType.DEPLOY,
        "object_name_raw": "Backend API v2.0",
        "anchors": ["DEPLOY-456"],
    }

    planned = create_test_event(
        **base_kwargs,
        status=EventStatus.PLANNED,
        planned_start=datetime(2025, 10, 25, tzinfo=pytz.UTC),
        environment=Environment.PROD,
    )

    started = create_test_event(
        **base_kwargs,
        status=EventStatus.STARTED,
        actual_start=datetime(2025, 10, 25, 9, 0, tzinfo=pytz.UTC),
        environment=Environment.PROD,
    )

    completed = create_test_event(
        **base_kwargs,
        status=EventStatus.COMPLETED,
        actual_start=datetime(2025, 10, 25, 9, 0, tzinfo=pytz.UTC),
        actual_end=datetime(2025, 10, 25, 10, 0, tzinfo=pytz.UTC),
        environment=Environment.PROD,
    )

    # All same initiative
    cluster_planned = generate_cluster_key(planned)
    cluster_started = generate_cluster_key(started)
    cluster_completed = generate_cluster_key(completed)

    assert cluster_planned == cluster_started == cluster_completed

    # But different instances
    dedup_planned = generate_dedup_key(planned)
    dedup_started = generate_dedup_key(started)
    dedup_completed = generate_dedup_key(completed)

    assert dedup_planned != dedup_started
    assert dedup_started != dedup_completed
    assert dedup_planned != dedup_completed
