"""Tests for PostgreSQL repository adapter.

These tests require a PostgreSQL instance to be running.
Set POSTGRES_PASSWORD environment variable to run these tests.
"""

import os
from datetime import datetime

import pytest
import pytz

from src.domain.models import (
    CandidateStatus,
    Event,
    EventCategory,
    LLMCallMetadata,
)

# Skip all tests if PostgreSQL is not configured
pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_PASSWORD"),
    reason="PostgreSQL test database not configured",
)


def test_postgres_connection(postgres_repository):
    """Test that PostgreSQL connection works."""
    # This test passes if the fixture doesn't raise an exception
    assert postgres_repository is not None
    assert postgres_repository.pool is not None


def test_save_and_retrieve_messages(postgres_repository, sample_slack_message):
    """Test saving and retrieving Slack messages."""
    # Save message
    count = postgres_repository.save_messages([sample_slack_message])
    assert count == 1

    # Retrieve via get_new_messages_for_candidates
    messages = postgres_repository.get_new_messages_for_candidates()
    assert len(messages) == 1
    assert messages[0].message_id == sample_slack_message.message_id
    assert messages[0].text == sample_slack_message.text


def test_save_duplicate_messages(postgres_repository, sample_slack_message):
    """Test that saving duplicate messages is idempotent."""
    # Save same message twice
    count1 = postgres_repository.save_messages([sample_slack_message])
    count2 = postgres_repository.save_messages([sample_slack_message])

    assert count1 == 1
    assert count2 == 1

    # Should still have only one message
    messages = postgres_repository.get_new_messages_for_candidates()
    assert len(messages) == 1


def test_watermark_operations(postgres_repository):
    """Test watermark get and update operations."""
    channel = "C123456"

    # Initially should be None
    watermark = postgres_repository.get_watermark(channel)
    assert watermark is None

    # Update watermark
    postgres_repository.update_watermark(channel, "1728000000.123456")

    # Retrieve updated watermark
    watermark = postgres_repository.get_watermark(channel)
    assert watermark == "1728000000.123456"

    # Update again
    postgres_repository.update_watermark(channel, "1728000001.123456")
    watermark = postgres_repository.get_watermark(channel)
    assert watermark == "1728000001.123456"


def test_save_and_retrieve_candidates(postgres_repository, sample_event_candidate):
    """Test saving and retrieving event candidates."""
    # Save candidate
    count = postgres_repository.save_candidates([sample_event_candidate])
    assert count == 1

    # Retrieve candidates
    candidates = postgres_repository.get_candidates_for_extraction(batch_size=10)
    assert len(candidates) == 1
    assert candidates[0].message_id == sample_event_candidate.message_id
    assert candidates[0].score == sample_event_candidate.score
    assert candidates[0].status == CandidateStatus.NEW


def test_update_candidate_status(postgres_repository, sample_event_candidate):
    """Test updating candidate status."""
    # Save candidate
    postgres_repository.save_candidates([sample_event_candidate])

    # Update status
    postgres_repository.update_candidate_status(
        sample_event_candidate.message_id, "llm_ok"
    )

    # Verify status updated (should not appear in new candidates)
    candidates = postgres_repository.get_candidates_for_extraction(batch_size=10)
    assert len(candidates) == 0


def test_save_and_retrieve_events(postgres_repository, sample_event):
    """Test saving and retrieving events."""
    # Save event
    count = postgres_repository.save_events([sample_event])
    assert count == 1

    # Retrieve events in window
    start_dt = sample_event.event_date.replace(hour=0, minute=0, second=0)
    end_dt = sample_event.event_date.replace(hour=23, minute=59, second=59)
    events = postgres_repository.get_events_in_window(start_dt, end_dt)

    assert len(events) == 1
    assert events[0].event_id == sample_event.event_id
    assert events[0].title == sample_event.title
    assert events[0].category == EventCategory.PRODUCT


def test_event_deduplication(postgres_repository, sample_event):
    """Test event deduplication by dedup_key."""
    # Save event
    postgres_repository.save_events([sample_event])

    # Create modified event with same dedup_key
    modified_event = Event(
        event_id=sample_event.event_id,
        version=2,  # Incremented version
        message_id=sample_event.message_id,
        source_msg_event_idx=sample_event.source_msg_event_idx,
        dedup_key=sample_event.dedup_key,  # Same dedup_key
        event_date=sample_event.event_date,
        event_end=None,
        category=sample_event.category,
        title="Updated Title",  # Changed title
        summary="Updated summary",
        impact_area=sample_event.impact_area,
        tags=sample_event.tags,
        links=sample_event.links,
        anchors=sample_event.anchors,
        confidence=sample_event.confidence,
        source_channels=sample_event.source_channels,
        ingested_at=sample_event.ingested_at,
    )

    # Save modified event
    postgres_repository.save_events([modified_event])

    # Should still have only one event with updated title
    start_dt = sample_event.event_date.replace(hour=0, minute=0, second=0)
    end_dt = sample_event.event_date.replace(hour=23, minute=59, second=59)
    events = postgres_repository.get_events_in_window(start_dt, end_dt)

    assert len(events) == 1
    assert events[0].title == "Updated Title"
    assert events[0].version == 2


def test_llm_call_tracking(postgres_repository):
    """Test LLM call metadata tracking."""
    metadata = LLMCallMetadata(
        message_id="test_msg_123",
        prompt_hash="abc123",
        model="gpt-5-nano",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.001,
        latency_ms=1500,
        cached=False,
        ts=datetime(2025, 10, 15, 10, 0, tzinfo=pytz.UTC),
    )

    # Save LLM call
    postgres_repository.save_llm_call(metadata)

    # Get daily cost
    date = datetime(2025, 10, 15, tzinfo=pytz.UTC)
    cost = postgres_repository.get_daily_llm_cost(date)
    assert cost == 0.001


def test_llm_response_caching(postgres_repository):
    """Test LLM response caching."""
    prompt_hash = "test_hash_123"

    # Initially should be None
    cached = postgres_repository.get_cached_llm_response(prompt_hash)
    assert cached is None

    # Save LLM call with prompt hash
    metadata = LLMCallMetadata(
        message_id="test_msg",
        prompt_hash=prompt_hash,
        model="gpt-5-nano",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.001,
        latency_ms=1500,
        cached=False,
        ts=datetime.utcnow().replace(tzinfo=pytz.UTC),
    )
    postgres_repository.save_llm_call(metadata)

    # Save response
    response_json = '{"events": []}'
    postgres_repository.save_llm_response(prompt_hash, response_json)

    # Retrieve cached response
    cached = postgres_repository.get_cached_llm_response(prompt_hash)
    assert cached == response_json


def test_ingestion_state(postgres_repository):
    """Test ingestion state tracking."""
    channel_id = "C123456"

    # Initially should be None
    last_ts = postgres_repository.get_last_processed_ts(channel_id)
    assert last_ts is None

    # Update last processed timestamp
    postgres_repository.update_last_processed_ts(channel_id, 1728000000.123456)

    # Retrieve timestamp
    last_ts = postgres_repository.get_last_processed_ts(channel_id)
    assert last_ts == 1728000000.123456


def test_connection_pool_behavior(postgres_repository):
    """Test that connection pooling works correctly."""
    # Make multiple calls to ensure pool is used
    for i in range(5):
        channel = f"C{i}"
        postgres_repository.update_watermark(channel, f"100000{i}.123456")
        watermark = postgres_repository.get_watermark(channel)
        assert watermark == f"100000{i}.123456"

    # Pool should still be active
    assert postgres_repository.pool is not None


def test_jsonb_field_handling(postgres_repository, sample_slack_message):
    """Test that JSONB fields are properly handled."""
    # Save message with complex JSON fields
    postgres_repository.save_messages([sample_slack_message])

    # Retrieve and verify JSON fields
    messages = postgres_repository.get_new_messages_for_candidates()
    assert len(messages) == 1

    message = messages[0]
    assert isinstance(message.links_raw, list)
    assert isinstance(message.links_norm, list)
    assert isinstance(message.anchors, list)
    assert isinstance(message.reactions, dict)


def test_timestamp_with_timezone_handling(postgres_repository, sample_slack_message):
    """Test that TIMESTAMP WITH TIME ZONE fields preserve timezone info."""
    # Save message
    postgres_repository.save_messages([sample_slack_message])

    # Retrieve and verify timezone is preserved
    messages = postgres_repository.get_new_messages_for_candidates()
    assert len(messages) == 1

    message = messages[0]
    assert message.ts_dt.tzinfo == pytz.UTC
    assert message.ingested_at.tzinfo == pytz.UTC
