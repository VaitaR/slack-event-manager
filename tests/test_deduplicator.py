"""Tests for deduplicator service."""

from datetime import datetime

import pytz

from src.domain.models import Event, MessageSource
from src.services import deduplicator
from src.services.title_renderer import TitleRenderer
from tests.conftest import create_test_event


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
    event1 = create_test_event(
        message_id="same_msg",
        object_name="Event 1",
        dedup_key="key1",
        actual_start=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        links=["https://example.com"],
        anchors=[],
    )

    event2 = create_test_event(
        message_id="same_msg",  # Same message!
        object_name="Event 2",
        dedup_key="key2",
        actual_start=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
        links=["https://example.com"],
        anchors=[],
    )

    assert deduplicator.should_merge_events(event1, event2) is False


def test_should_merge_events_no_anchor_overlap() -> None:
    """Test no merge without anchor/link overlap."""
    event1 = create_test_event(
        message_id="msg1",
        object_name="Release v1.0",
        dedup_key="key1",
        actual_start=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        links=["https://example1.com"],
        anchors=["PROJ-1"],
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Release v1.0",
        dedup_key="key2",
        actual_start=datetime(2025, 10, 10, 10, 30, tzinfo=pytz.UTC),
        links=["https://example2.com"],
        anchors=["PROJ-2"],
    )

    assert deduplicator.should_merge_events(event1, event2) is False


def test_should_merge_events_date_too_far() -> None:
    """Test no merge if date delta > 48 hours."""
    event1 = create_test_event(
        message_id="msg1",
        object_name="Release v1.0",
        dedup_key="key1",
        actual_start=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        links=["https://example.com"],
        anchors=["PROJ-1"],
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Release v1.0",
        dedup_key="key2",
        actual_start=datetime(2025, 10, 13, 10, 0, tzinfo=pytz.UTC),  # 3 days later
        links=["https://example.com"],
        anchors=["PROJ-1"],
    )

    assert (
        deduplicator.should_merge_events(event1, event2, date_window_hours=48) is False
    )


def test_merge_events_refreshes_dedup_key() -> None:
    """Merged events must have dedup_key recomputed after time updates."""

    later_time = datetime(2025, 10, 20, 10, 0, tzinfo=pytz.UTC)
    earlier_time = datetime(2025, 10, 19, 9, 0, tzinfo=pytz.UTC)

    event1 = create_test_event(
        message_id="msg-target",
        object_name="Release v1.0",
        dedup_key="temp",
        planned_start=later_time,
        actual_start=None,
        links=["https://example.com"],
        anchors=["PROJ-1"],
    ).model_copy(update={"actual_start": None, "planned_start": later_time})

    event1 = event1.model_copy(
        update={"dedup_key": deduplicator.generate_dedup_key(event1)}
    )

    event2 = create_test_event(
        message_id="msg-new",
        object_name="Release v1.0",
        dedup_key="temp2",
        actual_start=earlier_time,
        links=["https://example.com"],
        anchors=["PROJ-1"],
    )

    event2 = event2.model_copy(
        update={"dedup_key": deduplicator.generate_dedup_key(event2)}
    )

    merged = deduplicator.merge_events(event1, event2)

    assert merged.actual_start == earlier_time
    assert merged.dedup_key == deduplicator.generate_dedup_key(merged)


def test_should_merge_events_with_link_overlap() -> None:
    """Test merge with link overlap."""
    event1 = create_test_event(
        message_id="msg1",
        object_name="Release v1.0",
        dedup_key="key1",
        actual_start=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        links=["https://example.com/release"],
        anchors=[],
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Release v1.0",
        dedup_key="key2",
        actual_start=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
        links=["https://example.com/release"],
        anchors=[],
    )

    assert deduplicator.should_merge_events(event1, event2) is True


def test_should_merge_events_with_anchor_overlap() -> None:
    """Test merge with anchor overlap."""
    event1 = create_test_event(
        message_id="msg1",
        object_name="Fix critical bug",
        dedup_key="key1",
        actual_start=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        links=[],
        anchors=["PROJ-123"],
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Fix critical bug",
        dedup_key="key2",
        actual_start=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
        links=[],
        anchors=["PROJ-123", "PROJ-456"],
    )

    assert deduplicator.should_merge_events(event1, event2) is True


def test_should_merge_events_meets_similarity_threshold() -> None:
    """Events should merge when similarity meets the threshold."""
    renderer = TitleRenderer()
    event1 = create_test_event(
        message_id="msg1",
        object_name="Payments Dashboard",
        qualifiers=["beta"],
        anchors=["PROJ-123"],
        links=["https://example.com/a"],
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Payments Dashboard",
        qualifiers=["beta release"],
        anchors=["PROJ-123"],
        links=["https://example.com/a"],
    )

    assert (
        deduplicator.should_merge_events(
            event1,
            event2,
            title_similarity_threshold=0.89,
            title_renderer=renderer,
        )
        is True
    )


def test_should_not_merge_events_below_similarity_threshold() -> None:
    """Events should not merge when similarity falls below the threshold."""
    renderer = TitleRenderer()
    event1 = create_test_event(
        message_id="msg1",
        object_name="Payments Dashboard",
        qualifiers=["beta"],
        anchors=["PROJ-123"],
        links=["https://example.com/a"],
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Payments Dashboard",
        qualifiers=["beta release"],
        anchors=["PROJ-123"],
        links=["https://example.com/a"],
    )

    assert (
        deduplicator.should_merge_events(
            event1,
            event2,
            title_similarity_threshold=0.9,
            title_renderer=renderer,
        )
        is False
    )


def test_should_not_merge_events_from_different_sources() -> None:
    """Events from different sources must never merge."""
    renderer = TitleRenderer()
    event1 = create_test_event(
        message_id="msg1",
        object_name="Release v1.0",
        dedup_key="key1",
        links=["https://example.com/release"],
        anchors=["PROJ-999"],
        source_id=MessageSource.SLACK,
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Release v1.0",
        dedup_key="key2",
        links=["https://example.com/release"],
        anchors=["PROJ-999"],
        source_id=MessageSource.TELEGRAM,
    )

    assert (
        deduplicator.should_merge_events(
            event1,
            event2,
            title_similarity_threshold=0.5,
            title_renderer=renderer,
        )
        is False
    )


def test_merge_events_combines_fields() -> None:
    """Test event merging combines relevant fields."""
    event1 = create_test_event(
        message_id="msg1",
        object_name="Release v1.0",
        dedup_key="key1",
        actual_start=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        links=["https://link1.com"],
        anchors=["PROJ-1"],
        source_channels=["#releases"],
        confidence=0.9,
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Release v1.0",
        dedup_key="key2",
        actual_start=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
        links=["https://link2.com"],
        anchors=["PROJ-2"],
        source_channels=["#announcements"],
        confidence=0.95,
    )

    merged = deduplicator.merge_events(event1, event2)

    # Verify combined fields
    assert merged.confidence == 0.95  # Max confidence
    assert set(merged.links) == {"https://link1.com", "https://link2.com"}
    assert set(merged.anchors) == {"PROJ-1", "PROJ-2"}
    assert set(merged.source_channels) == {"#releases", "#announcements"}


def test_merge_events_prefers_earlier_date() -> None:
    """Test merged event uses earlier date."""
    later_date = datetime(2025, 10, 10, 12, 0, tzinfo=pytz.UTC)
    earlier_date = datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC)

    event1 = create_test_event(
        message_id="msg1",
        object_name="Release",
        dedup_key="key1",
        actual_start=later_date,
        links=["https://example.com"],
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Release",
        dedup_key="key2",
        actual_start=earlier_date,
        links=["https://example.com"],
    )

    merged = deduplicator.merge_events(event1, event2)

    assert merged.actual_start == earlier_date


def test_deduplicate_event_list_same_message_no_merge() -> None:
    """Test events from same message are not merged."""
    event1 = create_test_event(
        message_id="msg1",
        object_name="Event 1",
        dedup_key="key1",
        links=["https://example.com"],
    )

    event2 = create_test_event(
        message_id="msg1",  # Same message
        object_name="Event 2",
        dedup_key="key2",
        links=["https://example.com"],
    )

    events = [event1, event2]
    deduplicated = deduplicator.deduplicate_event_list(events)

    # Both events should remain
    assert len(deduplicated) == 2


def test_deduplicate_event_list_merges_similar() -> None:
    """Test similar events from different messages are merged."""
    event1 = create_test_event(
        message_id="msg1",
        object_name="Release",
        dedup_key="key1",
        actual_start=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
        links=["https://example.com/release"],
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Release",
        dedup_key="key2",
        actual_start=datetime(2025, 10, 10, 11, 0, tzinfo=pytz.UTC),
        links=["https://example.com/release"],
    )

    events = [event1, event2]
    deduplicated = deduplicator.deduplicate_event_list(events)

    # Should merge into 1 event
    assert len(deduplicated) == 1
    merged = deduplicated[0]
    assert "msg1" in merged.message_id or "msg2" in merged.message_id


def test_deduplicate_event_list_preserves_unique() -> None:
    """Test unique events are preserved."""
    event1 = create_test_event(
        message_id="msg1",
        object_name="Release v1.0",
        dedup_key="key1",
        links=["https://release1.com"],
        anchors=["PROJ-1"],
    )

    event2 = create_test_event(
        message_id="msg2",
        object_name="Security fix",
        dedup_key="key2",
        links=["https://security.com"],
        anchors=["SEC-100"],
    )

    event3 = create_test_event(
        message_id="msg3",
        object_name="Performance update",
        dedup_key="key3",
        links=["https://perf.com"],
        anchors=["PERF-42"],
    )

    events = [event1, event2, event3]
    deduplicated = deduplicator.deduplicate_event_list(events)

    # All should remain
    assert len(deduplicated) == 3


def test_deduplicate_event_list_empty() -> None:
    """Test deduplication with empty list."""
    events: list[Event] = []
    deduplicated = deduplicator.deduplicate_event_list(events)

    assert deduplicated == []


def test_deduplicate_event_list_single() -> None:
    """Test deduplication with single event."""
    event = create_test_event()

    events = [event]
    deduplicated = deduplicator.deduplicate_event_list(events)

    assert len(deduplicated) == 1
    assert deduplicated[0] == event
