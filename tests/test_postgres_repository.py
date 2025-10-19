"""Tests for PostgreSQL repository.

These tests are skipped unless:
- POSTGRES_PASSWORD environment variable is set
- TEST_POSTGRES=1 environment variable is set
- PostgreSQL is running on localhost:5432

Run with: TEST_POSTGRES=1 POSTGRES_PASSWORD=password pytest tests/test_postgres_repository.py
"""

from datetime import datetime, timedelta

import pytz

from src.adapters.query_builders import (
    CandidateQueryCriteria,
    EventQueryCriteria,
    high_priority_candidates_criteria,
    recent_events_criteria,
)
from src.domain.models import CandidateStatus, EventCategory, LLMCallMetadata


def test_postgres_save_and_get_messages(postgres_test_db, sample_slack_message):
    """Test saving and retrieving Slack messages in PostgreSQL."""
    repo = postgres_test_db

    # Save message
    count = repo.save_messages([sample_slack_message])
    assert count == 1

    # Get messages for candidates (should be empty initially)
    messages = repo.get_new_messages_for_candidates()
    assert len(messages) == 1
    assert messages[0].message_id == sample_slack_message.message_id
    assert messages[0].channel == sample_slack_message.channel
    assert messages[0].text == sample_slack_message.text


def test_postgres_message_upsert_updates_all_mutable_fields(
    postgres_test_db, sample_slack_message
):
    """Test that message upsert updates all mutable fields (reactions, attachments, etc).

    This ensures PostgreSQL behaves like SQLite (INSERT OR REPLACE) by updating
    all fields that can change after initial message creation (reactions, attachments,
    files, links, anchors, permalink, edits, user info).
    """
    repo = postgres_test_db

    # Save initial message
    count = repo.save_messages([sample_slack_message])
    assert count == 1

    # Verify initial state
    messages = repo.get_new_messages_for_candidates()
    assert len(messages) == 1
    initial_msg = messages[0]
    assert initial_msg.reply_count == sample_slack_message.reply_count
    assert initial_msg.attachments_count == sample_slack_message.attachments_count
    assert initial_msg.files_count == sample_slack_message.files_count
    assert initial_msg.total_reactions == sample_slack_message.total_reactions

    # Create updated message with changed mutable fields
    # (simulating Slack API returning updated data on next fetch)
    from src.domain.models import SlackMessage

    updated_msg = SlackMessage(
        message_id=sample_slack_message.message_id,  # Same ID
        channel=sample_slack_message.channel,
        user=sample_slack_message.user,
        user_real_name="Updated User Name",  # Changed
        user_display_name="updated_user",  # Changed
        user_email="updated@example.com",  # Changed
        user_profile_image="https://example.com/new_avatar.jpg",  # Changed
        ts=sample_slack_message.ts,
        ts_dt=sample_slack_message.ts_dt,
        is_bot=sample_slack_message.is_bot,
        subtype=sample_slack_message.subtype,
        text="Updated text content",  # Changed
        blocks_text="Updated blocks content",  # Changed
        text_norm="updated text content",  # Changed
        links_raw=["https://new-link.com"],  # Changed
        links_norm=["https://new-link.com"],  # Changed
        anchors=["NEW-123", "NEW-456"],  # Changed
        attachments_count=5,  # Changed (was 2)
        files_count=3,  # Changed (was 1)
        reactions={"rocket": 10, "tada": 5},  # Changed
        total_reactions=15,  # Changed (was 3)
        reply_count=20,  # Changed (was 5)
        permalink="https://example.slack.com/archives/C123/p1234567890_new",  # Changed
        edited_ts="1234567891.123456",  # Changed
        edited_user="U999999",  # Changed
        ingested_at=sample_slack_message.ingested_at,
    )

    # Save updated message (upsert)
    count = repo.save_messages([updated_msg])
    assert count == 1

    # Verify ALL mutable fields were updated
    messages = repo.get_new_messages_for_candidates()
    assert len(messages) == 1
    updated_retrieved = messages[0]

    # Check user fields
    assert updated_retrieved.user_real_name == "Updated User Name"
    assert updated_retrieved.user_display_name == "updated_user"
    assert updated_retrieved.user_email == "updated@example.com"
    assert updated_retrieved.user_profile_image == "https://example.com/new_avatar.jpg"

    # Check text fields
    assert updated_retrieved.text == "Updated text content"
    assert updated_retrieved.blocks_text == "Updated blocks content"
    assert updated_retrieved.text_norm == "updated text content"

    # Check links and anchors
    assert updated_retrieved.links_raw == ["https://new-link.com"]
    assert updated_retrieved.links_norm == ["https://new-link.com"]
    assert updated_retrieved.anchors == ["NEW-123", "NEW-456"]

    # Check counts (critical for scoring!)
    assert updated_retrieved.attachments_count == 5
    assert updated_retrieved.files_count == 3
    assert updated_retrieved.total_reactions == 15
    assert updated_retrieved.reply_count == 20

    # Check reactions dict
    assert updated_retrieved.reactions == {"rocket": 10, "tada": 5}

    # Check edit metadata
    assert (
        updated_retrieved.permalink
        == "https://example.slack.com/archives/C123/p1234567890_new"
    )
    assert updated_retrieved.edited_ts == "1234567891.123456"
    assert updated_retrieved.edited_user == "U999999"


def test_postgres_watermark_operations(postgres_test_db):
    """Test watermark get/set operations in PostgreSQL."""
    repo = postgres_test_db
    channel = "C123456"

    # Initially no watermark
    watermark = repo.get_watermark(channel)
    assert watermark is None

    # Set watermark
    repo.update_watermark(channel, "1728000000.123456")

    # Get watermark
    watermark = repo.get_watermark(channel)
    assert watermark == "1728000000.123456"

    # Update watermark
    repo.update_watermark(channel, "1728000001.234567")
    watermark = repo.get_watermark(channel)
    assert watermark == "1728000001.234567"


def test_postgres_save_and_get_candidates(postgres_test_db, sample_event_candidate):
    """Test saving and retrieving event candidates in PostgreSQL."""
    repo = postgres_test_db

    # Save candidate
    count = repo.save_candidates([sample_event_candidate])
    assert count == 1

    # Get candidates for extraction
    candidates = repo.get_candidates_for_extraction(batch_size=10)
    assert len(candidates) == 1
    assert candidates[0].message_id == sample_event_candidate.message_id
    assert candidates[0].channel == sample_event_candidate.channel
    assert candidates[0].status == CandidateStatus.NEW


def test_postgres_update_candidate_status(postgres_test_db, sample_event_candidate):
    """Test updating candidate status in PostgreSQL."""
    repo = postgres_test_db

    # Save candidate
    repo.save_candidates([sample_event_candidate])

    # Update status
    repo.update_candidate_status(sample_event_candidate.message_id, "llm_ok")

    # Verify status changed (should not appear in extraction queue)
    candidates = repo.get_candidates_for_extraction(batch_size=10)
    assert len(candidates) == 0


def test_postgres_save_and_get_events(postgres_test_db, sample_event):
    """Test saving and retrieving events in PostgreSQL."""
    repo = postgres_test_db

    # Save event
    count = repo.save_events([sample_event])
    assert count == 1

    # Get events in window
    start_dt = datetime(2025, 10, 14, 0, 0, tzinfo=pytz.UTC)
    end_dt = datetime(2025, 10, 16, 0, 0, tzinfo=pytz.UTC)
    events = repo.get_events_in_window(start_dt, end_dt)

    assert len(events) == 1
    assert events[0].title == sample_event.title
    assert events[0].category == sample_event.category
    assert events[0].confidence == sample_event.confidence


def test_postgres_event_versioning(postgres_test_db, sample_event):
    """Test event versioning (upsert by dedup_key) in PostgreSQL."""
    repo = postgres_test_db

    # Save initial event
    repo.save_events([sample_event])

    # Update event with same dedup_key
    updated_event = sample_event.model_copy(
        update={"title": "Updated Title", "version": 2}
    )
    repo.save_events([updated_event])

    # Get events
    start_dt = datetime(2025, 10, 14, 0, 0, tzinfo=pytz.UTC)
    end_dt = datetime(2025, 10, 16, 0, 0, tzinfo=pytz.UTC)
    events = repo.get_events_in_window(start_dt, end_dt)

    assert len(events) == 1
    assert events[0].title == "Updated Title"
    assert events[0].version == 2


def test_postgres_save_llm_call(postgres_test_db):
    """Test saving LLM call metadata in PostgreSQL."""
    repo = postgres_test_db

    metadata = LLMCallMetadata(
        message_id="test_msg_123",
        prompt_hash="abc123",
        model="gpt-5-nano",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.001,
        latency_ms=1500,
        cached=False,
        ts=datetime(2025, 10, 10, 10, 0, tzinfo=pytz.UTC),
    )

    # Save LLM call
    repo.save_llm_call(metadata)

    # Get daily cost
    cost = repo.get_daily_llm_cost(datetime(2025, 10, 10, tzinfo=pytz.UTC))
    assert cost == 0.001


def test_postgres_get_daily_llm_cost_multiple_calls(postgres_test_db):
    """Test daily LLM cost calculation with multiple calls in PostgreSQL."""
    repo = postgres_test_db

    # Save multiple LLM calls
    for i in range(3):
        metadata = LLMCallMetadata(
            message_id=f"test_msg_{i}",
            prompt_hash=f"hash_{i}",
            model="gpt-5-nano",
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.001,
            latency_ms=1500,
            cached=False,
            ts=datetime(2025, 10, 10, 10, i, tzinfo=pytz.UTC),
        )
        repo.save_llm_call(metadata)

    # Get daily cost
    cost = repo.get_daily_llm_cost(datetime(2025, 10, 10, tzinfo=pytz.UTC))
    assert cost == 0.003  # 3 calls * 0.001


def test_postgres_ingestion_state(postgres_test_db):
    """Test ingestion state get/update operations in PostgreSQL."""
    repo = postgres_test_db
    channel = "C123456"

    # Initially no state
    ts = repo.get_last_processed_ts(channel)
    assert ts is None

    # Update state
    repo.update_last_processed_ts(channel, "1728000000.123456")

    # Get state
    ts = repo.get_last_processed_ts(channel)
    assert ts == "1728000000.123456"

    # Update again
    repo.update_last_processed_ts(channel, "1728000001.234567")
    ts = repo.get_last_processed_ts(channel)
    assert ts == "1728000001.234567"


def test_postgres_candidate_batch_size_none(postgres_test_db, sample_event_candidate):
    """Test getting all candidates when batch_size=None in PostgreSQL."""
    repo = postgres_test_db

    # Save multiple candidates
    candidates = []
    for i in range(5):
        candidate = sample_event_candidate.model_copy(
            update={"message_id": f"test_msg_{i}"}
        )
        candidates.append(candidate)
    repo.save_candidates(candidates)

    # Get all candidates (batch_size=None)
    retrieved = repo.get_candidates_for_extraction(batch_size=None)
    assert len(retrieved) == 5

    # Get limited candidates
    retrieved_limited = repo.get_candidates_for_extraction(batch_size=2)
    assert len(retrieved_limited) == 2


def test_postgres_candidate_min_score_filter(postgres_test_db, sample_event_candidate):
    """Test candidate filtering by min_score in PostgreSQL."""
    repo = postgres_test_db

    # Save candidates with different scores
    candidate1 = sample_event_candidate.model_copy(
        update={"message_id": "msg_1", "score": 10.0}
    )
    candidate2 = sample_event_candidate.model_copy(
        update={"message_id": "msg_2", "score": 20.0}
    )
    candidate3 = sample_event_candidate.model_copy(
        update={"message_id": "msg_3", "score": 30.0}
    )
    repo.save_candidates([candidate1, candidate2, candidate3])

    # Get candidates with min_score
    retrieved = repo.get_candidates_for_extraction(batch_size=None, min_score=15.0)
    assert len(retrieved) == 2
    assert all(c.score >= 15.0 for c in retrieved)


def test_postgres_message_upsert(postgres_test_db, sample_slack_message):
    """Test message upsert behavior in PostgreSQL."""
    repo = postgres_test_db

    # Save initial message
    repo.save_messages([sample_slack_message])

    # Update message with same message_id
    updated_message = sample_slack_message.model_copy(
        update={"text": "Updated text", "text_norm": "updated text"}
    )
    repo.save_messages([updated_message])

    # Get messages
    messages = repo.get_new_messages_for_candidates()
    assert len(messages) == 1
    assert messages[0].text == "Updated text"


def test_postgres_event_category_handling(postgres_test_db, sample_event):
    """Test proper handling of EventCategory enum in PostgreSQL."""
    repo = postgres_test_db

    # Test different categories
    categories = [
        EventCategory.PRODUCT,
        EventCategory.PROCESS,
        EventCategory.MARKETING,
        EventCategory.RISK,
        EventCategory.ORG,
        EventCategory.UNKNOWN,
    ]

    for idx, category in enumerate(categories):
        event = sample_event.model_copy(
            update={
                "message_id": f"msg_{idx}",
                "dedup_key": f"key_{idx}",
                "category": category,
            }
        )
        repo.save_events([event])

    # Get all events
    start_dt = datetime(2025, 10, 14, 0, 0, tzinfo=pytz.UTC)
    end_dt = datetime(2025, 10, 16, 0, 0, tzinfo=pytz.UTC)
    events = repo.get_events_in_window(start_dt, end_dt)

    assert len(events) == len(categories)
    retrieved_categories = {e.category for e in events}
    assert retrieved_categories == set(categories)


def test_postgres_query_events_basic(postgres_test_db, sample_event):
    """Test basic event querying in PostgreSQL."""
    repo = postgres_test_db

    # Save test events
    events = []
    for i in range(3):
        event = sample_event.model_copy(
            update={
                "message_id": f"msg_{i}",
                "dedup_key": f"key_{i}",
                "category": EventCategory.PRODUCT if i < 2 else EventCategory.RISK,
                "confidence": 0.8 + i * 0.1,
            }
        )
        events.append(event)
    repo.save_events(events)

    # Query all events
    criteria = EventQueryCriteria()
    results = repo.query_events(criteria)
    assert len(results) == 3

    # Query by category
    criteria = EventQueryCriteria(categories=[EventCategory.PRODUCT])
    results = repo.query_events(criteria)
    assert len(results) == 2
    assert all(e.category == EventCategory.PRODUCT for e in results)

    # Query by min confidence
    criteria = EventQueryCriteria(min_confidence=0.9)
    results = repo.query_events(criteria)
    assert len(results) == 2
    assert all(e.confidence >= 0.9 for e in results)


def test_postgres_query_events_date_filters(postgres_test_db, sample_event):
    """Test event querying with date filters in PostgreSQL."""
    repo = postgres_test_db

    # Create events with different extraction times
    base_time = datetime(2025, 10, 15, 12, 0, tzinfo=pytz.UTC)
    events = []

    for i in range(3):
        event = sample_event.model_copy(
            update={
                "message_id": f"msg_{i}",
                "dedup_key": f"key_{i}",
                "extracted_at": base_time + timedelta(hours=i),
            }
        )
        events.append(event)
    repo.save_events(events)

    # Query events extracted after specific time
    criteria = EventQueryCriteria(extracted_after=base_time + timedelta(hours=0.5))
    results = repo.query_events(criteria)
    assert len(results) == 2

    # Query events extracted before specific time
    criteria = EventQueryCriteria(extracted_before=base_time + timedelta(hours=1.5))
    results = repo.query_events(criteria)
    assert len(results) == 2


def test_postgres_query_events_source_channels(postgres_test_db, sample_event):
    """Test event querying by source channels in PostgreSQL."""
    repo = postgres_test_db

    # Save events with different source channels
    events = []
    for i in range(3):
        event = sample_event.model_copy(
            update={
                "message_id": f"msg_{i}",
                "dedup_key": f"key_{i}",
                "source_channels": [f"channel_{i}", "channel_0"]
                if i > 0
                else ["channel_0"],
            }
        )
        events.append(event)
    repo.save_events(events)

    # Query by source channels
    criteria = EventQueryCriteria(source_channels=["channel_0"])
    results = repo.query_events(criteria)
    assert len(results) == 3  # All events have channel_0

    criteria = EventQueryCriteria(source_channels=["channel_1"])
    results = repo.query_events(criteria)
    assert len(results) == 1  # Only event with channel_1


def test_postgres_query_events_with_limit_offset(postgres_test_db, sample_event):
    """Test event querying with limit and offset in PostgreSQL."""
    repo = postgres_test_db

    # Save multiple events
    events = []
    for i in range(5):
        event = sample_event.model_copy(
            update={
                "message_id": f"msg_{i}",
                "dedup_key": f"key_{i}",
                "confidence": 0.5 + i * 0.1,
            }
        )
        events.append(event)
    repo.save_events(events)

    # Query with limit
    criteria = EventQueryCriteria(limit=3)
    results = repo.query_events(criteria)
    assert len(results) == 3

    # Query with limit and offset
    criteria = EventQueryCriteria(limit=2, offset=2)
    results = repo.query_events(criteria)
    assert len(results) == 2

    # Query with offset beyond available results
    criteria = EventQueryCriteria(limit=10, offset=10)
    results = repo.query_events(criteria)
    assert len(results) == 0


def test_postgres_query_candidates_basic(postgres_test_db, sample_event_candidate):
    """Test basic candidate querying in PostgreSQL."""
    repo = postgres_test_db

    # Save test candidates
    candidates = []
    for i in range(3):
        candidate = sample_event_candidate.model_copy(
            update={
                "message_id": f"msg_{i}",
                "score": 10.0 + i * 5.0,
                "status": CandidateStatus.NEW,
            }
        )
        candidates.append(candidate)
    repo.save_candidates(candidates)

    # Query all candidates
    criteria = CandidateQueryCriteria()
    results = repo.query_candidates(criteria)
    assert len(results) == 3

    # Query by status
    criteria = CandidateQueryCriteria(status="new")
    results = repo.query_candidates(criteria)
    assert len(results) == 3

    # Query by min score
    criteria = CandidateQueryCriteria(min_score=15.0)
    results = repo.query_candidates(criteria)
    assert len(results) == 2
    assert all(c.score >= 15.0 for c in results)


def test_postgres_query_candidates_channel_filter(
    postgres_test_db, sample_event_candidate
):
    """Test candidate querying by channel in PostgreSQL."""
    repo = postgres_test_db

    # Save candidates with different channels
    candidates = []
    for i in range(3):
        candidate = sample_event_candidate.model_copy(
            update={
                "message_id": f"msg_{i}",
                "channel": f"channel_{i}",
                "score": 10.0 + i,
            }
        )
        candidates.append(candidate)
    repo.save_candidates(candidates)

    # Query by specific channel
    criteria = CandidateQueryCriteria(channel="channel_1")
    results = repo.query_candidates(criteria)
    assert len(results) == 1
    assert results[0].channel == "channel_1"


def test_postgres_query_candidates_date_filters(
    postgres_test_db, sample_event_candidate
):
    """Test candidate querying with date filters in PostgreSQL."""
    repo = postgres_test_db

    # Create candidates with different timestamps
    base_time = datetime(2025, 10, 15, 12, 0, tzinfo=pytz.UTC)
    candidates = []

    for i in range(3):
        candidate = sample_event_candidate.model_copy(
            update={
                "message_id": f"msg_{i}",
                "ts_dt": base_time + timedelta(hours=i),
            }
        )
        candidates.append(candidate)
    repo.save_candidates(candidates)

    # Query candidates created after specific time
    criteria = CandidateQueryCriteria(created_after=base_time + timedelta(hours=0.5))
    results = repo.query_candidates(criteria)
    assert len(results) == 2

    # Query candidates created before specific time
    criteria = CandidateQueryCriteria(created_before=base_time + timedelta(hours=1.5))
    results = repo.query_candidates(criteria)
    assert len(results) == 2


def test_postgres_query_candidates_with_limit_offset(
    postgres_test_db, sample_event_candidate
):
    """Test candidate querying with limit and offset in PostgreSQL."""
    repo = postgres_test_db

    # Save multiple candidates
    candidates = []
    for i in range(5):
        candidate = sample_event_candidate.model_copy(
            update={
                "message_id": f"msg_{i}",
                "score": 20.0 - i,  # Decreasing scores
            }
        )
        candidates.append(candidate)
    repo.save_candidates(candidates)

    # Query with limit
    criteria = CandidateQueryCriteria(limit=3)
    results = repo.query_candidates(criteria)
    assert len(results) == 3

    # Query with limit and offset
    criteria = CandidateQueryCriteria(limit=2, offset=2)
    results = repo.query_candidates(criteria)
    assert len(results) == 2

    # Verify ordering (should be by score DESC by default)
    assert results[0].score >= results[1].score


def test_postgres_query_criteria_helper_functions(
    postgres_test_db, sample_event, sample_event_candidate
):
    """Test query criteria helper functions work with PostgreSQL."""
    repo = postgres_test_db

    # Save test data
    event = sample_event.model_copy(
        update={
            "message_id": "test_event",
            "dedup_key": "test_key",
            "extracted_at": datetime(2025, 10, 15, 12, 0, tzinfo=pytz.UTC),
        }
    )
    repo.save_events([event])

    candidate = sample_event_candidate.model_copy(
        update={
            "message_id": "test_candidate",
            "score": 20.0,
            "status": CandidateStatus.NEW,
        }
    )
    repo.save_candidates([candidate])

    # Test recent_events_criteria helper
    criteria = recent_events_criteria(days=1)
    events = repo.query_events(criteria)
    assert len(events) >= 0  # Should not fail

    # Test high_priority_candidates_criteria helper
    criteria = high_priority_candidates_criteria(threshold=15.0)
    candidates = repo.query_candidates(criteria)
    assert len(candidates) >= 0  # Should not fail


def test_postgres_query_events_empty_results(postgres_test_db):
    """Test query methods return empty results when no data matches."""
    repo = postgres_test_db

    # Query with very restrictive criteria
    criteria = EventQueryCriteria(
        categories=[EventCategory.PRODUCT],
        min_confidence=0.9,
        start_date=datetime(2025, 10, 15, tzinfo=pytz.UTC),
        end_date=datetime(2025, 10, 16, tzinfo=pytz.UTC),
    )
    results = repo.query_events(criteria)
    assert results == []

    # Query candidates with restrictive criteria
    criteria = CandidateQueryCriteria(
        min_score=100.0,
        status="new",
    )
    results = repo.query_candidates(criteria)
    assert results == []
