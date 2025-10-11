"""Tests for deduplicator service."""

from datetime import datetime

import pytz

from src.domain.models import Event, EventCategory
from src.services import deduplicator


def test_generate_dedup_key(sample_event: Event) -> None:
    """Test dedup key generation."""
    key = deduplicator.generate_dedup_key(sample_event)

    assert isinstance(key, str)
    assert len(key) == 40  # SHA1 hex digest length


def test_generate_dedup_key_deterministic(sample_event: Event) -> None:
    """Test dedup key is deterministic."""
    key1 = deduplicator.generate_dedup_key(sample_event)
    key2 = deduplicator.generate_dedup_key(sample_event)

    assert key1 == key2


def test_has_overlap_with_common_elements() -> None:
    """Test overlap detection with common elements."""
    list1 = ["a", "b", "c"]
    list2 = ["c", "d", "e"]

    assert deduplicator.has_overlap(list1, list2) is True


def test_has_overlap_no_common_elements() -> None:
    """Test overlap detection without common elements."""
    list1 = ["a", "b"]
    list2 = ["c", "d"]

    assert deduplicator.has_overlap(list1, list2) is False


def test_has_overlap_empty_lists() -> None:
    """Test overlap detection with empty lists."""
    assert deduplicator.has_overlap([], []) is False
    assert deduplicator.has_overlap(["a"], []) is False


def test_should_merge_events_same_message_id() -> None:
    """Test Rule 1: Same message_id should NOT merge."""
    event1 = Event(
        message_id="same_msg",
        source_msg_event_idx=0,
        dedup_key="key1",
        event_date=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Event 1",
        summary="Summary",
        links=["https://example.com"],
        anchors=[],
        confidence=0.9,
        source_channels=["#test"],
    )

    event2 = Event(
        message_id="same_msg",  # Same message!
        source_msg_event_idx=1,
        dedup_key="key2",
        event_date=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Event 2",
        summary="Summary",
        links=["https://example.com"],
        anchors=[],
        confidence=0.9,
        source_channels=["#test"],
    )

    assert deduplicator.should_merge_events(event1, event2) is False


def test_should_merge_events_no_anchor_overlap() -> None:
    """Test no merge without anchor/link overlap."""
    event1 = Event(
        message_id="msg1",
        source_msg_event_idx=0,
        dedup_key="key1",
        event_date=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Release v1.0",
        summary="Summary",
        links=["https://example1.com"],
        anchors=["PROJ-1"],
        confidence=0.9,
        source_channels=["#test"],
    )

    event2 = Event(
        message_id="msg2",
        source_msg_event_idx=0,
        dedup_key="key2",
        event_date=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Release v1.0",
        summary="Summary",
        links=["https://example2.com"],  # Different link
        anchors=["PROJ-2"],  # Different anchor
        confidence=0.9,
        source_channels=["#test"],
    )

    assert deduplicator.should_merge_events(event1, event2) is False


def test_should_merge_events_date_too_far() -> None:
    """Test no merge when date delta too large."""
    event1 = Event(
        message_id="msg1",
        source_msg_event_idx=0,
        dedup_key="key1",
        event_date=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Release v1.0",
        summary="Summary",
        links=["https://example.com"],
        anchors=["PROJ-1"],
        confidence=0.9,
        source_channels=["#test"],
    )

    event2 = Event(
        message_id="msg2",
        source_msg_event_idx=0,
        dedup_key="key2",
        event_date=datetime(2025, 10, 15, 10, 0, tzinfo=pytz.UTC),  # 5 days later
        category=EventCategory.PRODUCT,
        title="Release v1.0",
        summary="Summary",
        links=["https://example.com"],  # Same link
        anchors=["PROJ-1"],  # Same anchor
        confidence=0.9,
        source_channels=["#test"],
    )

    # Default window is 48 hours
    assert (
        deduplicator.should_merge_events(event1, event2, date_window_hours=48) is False
    )


def test_should_merge_events_title_too_different() -> None:
    """Test no merge when title similarity too low."""
    event1 = Event(
        message_id="msg1",
        source_msg_event_idx=0,
        dedup_key="key1",
        event_date=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Release v1.0",
        summary="Summary",
        links=["https://example.com"],
        anchors=["PROJ-1"],
        confidence=0.9,
        source_channels=["#test"],
    )

    event2 = Event(
        message_id="msg2",
        source_msg_event_idx=0,
        dedup_key="key2",
        event_date=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Completely Different Title",  # Very different
        summary="Summary",
        links=["https://example.com"],
        anchors=["PROJ-1"],
        confidence=0.9,
        source_channels=["#test"],
    )

    assert (
        deduplicator.should_merge_events(event1, event2, title_similarity_threshold=0.8)
        is False
    )


def test_should_merge_events_valid_merge() -> None:
    """Test valid merge case."""
    event1 = Event(
        message_id="msg1",
        source_msg_event_idx=0,
        dedup_key="key1",
        event_date=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Release v1.0",
        summary="Summary",
        links=["https://example.com"],
        anchors=["PROJ-1"],
        confidence=0.9,
        source_channels=["#releases"],
    )

    event2 = Event(
        message_id="msg2",
        source_msg_event_idx=0,
        dedup_key="key2",
        event_date=datetime(2025, 10, 10, 12, 0, tzinfo=pytz.UTC),  # 2 hours later
        category=EventCategory.PRODUCT,
        title="Release v1.0",  # Same title
        summary="Summary",
        links=["https://example.com"],  # Same link
        anchors=["PROJ-1"],  # Same anchor
        confidence=0.8,
        source_channels=["#updates"],
    )

    assert deduplicator.should_merge_events(event1, event2) is True


def test_merge_events_combines_attributes() -> None:
    """Test event merging combines attributes correctly."""
    event1 = Event(
        message_id="msg1",
        source_msg_event_idx=0,
        dedup_key="key1",
        event_date=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Release v1.0",
        summary="Summary 1",
        links=["https://link1.com"],
        anchors=["PROJ-1"],
        tags=["tag1"],
        confidence=0.8,
        source_channels=["#releases"],
        version=1,
    )

    event2 = Event(
        message_id="msg2",
        source_msg_event_idx=0,
        dedup_key="key2",
        event_date=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Release v1.0",
        summary="Summary 2",
        links=["https://link2.com"],
        anchors=["PROJ-2"],
        tags=["tag2"],
        confidence=0.9,  # Higher
        source_channels=["#updates"],
        version=1,
    )

    merged = deduplicator.merge_events(event1, event2)

    # Check unions
    assert set(merged.links) == {"https://link1.com", "https://link2.com"}
    assert set(merged.tags) == {"tag1", "tag2"}
    assert set(merged.source_channels) == {"#releases", "#updates"}
    assert set(merged.anchors) == {"PROJ-1", "PROJ-2"}

    # Check max values
    assert merged.confidence == 0.9
    assert merged.version == 2  # Incremented

    # Check kept attributes
    assert merged.title == event1.title
    assert merged.event_id == event1.event_id


def test_find_merge_candidates() -> None:
    """Test finding merge candidates."""
    new_event = Event(
        message_id="new_msg",
        source_msg_event_idx=0,
        dedup_key="new_key",
        event_date=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        category=EventCategory.PRODUCT,
        title="Release v1.0",
        summary="Summary",
        links=["https://example.com"],
        anchors=["PROJ-1"],
        confidence=0.9,
        source_channels=["#test"],
    )

    existing_events = [
        Event(
            message_id="existing1",
            source_msg_event_idx=0,
            dedup_key="key1",
            event_date=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
            category=EventCategory.PRODUCT,
            title="Release v1.0",
            summary="Summary",
            links=["https://example.com"],
            anchors=["PROJ-1"],
            confidence=0.9,
            source_channels=["#test"],
        ),
        Event(
            message_id="existing2",
            source_msg_event_idx=0,
            dedup_key="key2",
            event_date=datetime(2025, 10, 10, 12, 0, tzinfo=pytz.UTC),
            category=EventCategory.PRODUCT,
            title="Different Event",
            summary="Summary",
            links=["https://other.com"],
            anchors=["PROJ-2"],
            confidence=0.9,
            source_channels=["#test"],
        ),
    ]

    candidates = deduplicator.find_merge_candidates(new_event, existing_events)

    assert len(candidates) == 1
    assert candidates[0].message_id == "existing1"


def test_deduplicate_event_list() -> None:
    """Test deduplicating a list of events."""
    events = [
        Event(
            message_id="msg1",
            source_msg_event_idx=0,
            dedup_key="key1",
            event_date=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
            category=EventCategory.PRODUCT,
            title="Release v1.0",
            summary="Summary",
            links=["https://example.com"],
            anchors=["PROJ-1"],
            confidence=0.9,
            source_channels=["#test"],
        ),
        Event(
            message_id="msg2",
            source_msg_event_idx=0,
            dedup_key="key2",
            event_date=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
            category=EventCategory.PRODUCT,
            title="Release v1.0",  # Should merge with first
            summary="Summary",
            links=["https://example.com"],
            anchors=["PROJ-1"],
            confidence=0.8,
            source_channels=["#test"],
        ),
        Event(
            message_id="msg3",
            source_msg_event_idx=0,
            dedup_key="key3",
            event_date=datetime(2025, 10, 10, 12, 0, tzinfo=pytz.UTC),
            category=EventCategory.PRODUCT,
            title="Different Event",  # Should NOT merge
            summary="Summary",
            links=["https://other.com"],
            anchors=["PROJ-2"],
            confidence=0.9,
            source_channels=["#test"],
        ),
    ]

    deduplicated = deduplicator.deduplicate_event_list(events)

    # Should have 2 events (first two merged, third separate)
    assert len(deduplicated) == 2
