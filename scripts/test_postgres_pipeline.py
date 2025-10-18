#!/usr/bin/env python3
"""Test full pipeline with PostgreSQL database.

This script tests the complete pipeline:
1. Ingest mock messages
2. Build candidates
3. Extract events (mocked LLM)
4. Deduplicate events
5. Verify data in PostgreSQL
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adapters.postgres_repository import PostgresRepository
from src.adapters.repository_factory import create_repository
from src.config.settings import get_settings
from src.domain.models import (
    CandidateStatus,
    EventCandidate,
    LLMEvent,
    ScoringFeatures,
    SlackMessage,
)
from src.services.deduplicator import (
    generate_cluster_key,
    generate_dedup_key,
)
from src.services.text_normalizer import normalize_text
from src.use_cases.deduplicate_events import deduplicate_events_use_case

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_mock_messages() -> list[SlackMessage]:
    """Create mock Slack messages for testing."""
    now = datetime.now(pytz.UTC)

    messages = [
        SlackMessage(
            message_id=f"msg_{uuid4().hex[:8]}",
            channel="C123456",
            ts=str(now.timestamp()),
            ts_dt=now,
            user="U123456",
            text="🚀 Launching Payment Gateway v2.0 tomorrow at 10:00 UTC. This will enable faster transactions. https://github.com/org/repo/issues/123",
            reactions={"rocket": 5, "eyes": 3},
            total_reactions=8,
            reply_count=2,
            links_raw=["https://github.com/org/repo/issues/123"],
            anchors=["org/repo#123"],
            user_real_name="John Doe",
            user_display_name="johndoe",
        ),
        SlackMessage(
            message_id=f"msg_{uuid4().hex[:8]}",
            channel="C123456",
            ts=str((now - timedelta(hours=2)).timestamp()),
            ts_dt=now - timedelta(hours=2),
            user="U789012",
            text="⚠️ INCIDENT: Database migration failed in production. Rolling back now. ETA 30 minutes. JIRA: PROD-456",
            reactions={"warning": 8, "fire": 4},
            total_reactions=12,
            reply_count=5,
            anchors=["PROD-456"],
            user_real_name="Jane Smith",
            user_display_name="janesmith",
        ),
        SlackMessage(
            message_id=f"msg_{uuid4().hex[:8]}",
            channel="C123456",
            ts=str((now - timedelta(hours=5)).timestamp()),
            ts_dt=now - timedelta(hours=5),
            user="U345678",
            text="📢 New marketing campaign launching next week: Summer Sale 2025. Check the deck: https://docs.google.com/presentation/d/abc123",
            reactions={"tada": 3},
            total_reactions=3,
            reply_count=1,
            links_raw=["https://docs.google.com/presentation/d/abc123"],
            anchors=["gdoc:abc123"],
            user_real_name="Mike Johnson",
            user_display_name="mikej",
        ),
    ]

    return messages


def create_mock_llm_events(messages: list[SlackMessage]) -> dict[str, list[LLMEvent]]:
    """Create mock LLM extraction results."""
    llm_events = {}

    # Event 1: Payment Gateway Launch
    llm_events[messages[0].message_id] = [
        LLMEvent(
            action="launch",
            object_name="Payment Gateway v2.0",
            category="product",
            status="planned",
            change_type="launch",
            environment="prod",
            planned_start="tomorrow at 10:00 UTC",
            summary="Launching Payment Gateway v2.0 to enable faster transactions",
            links=["https://github.com/org/repo/issues/123"],
            anchors=["org/repo#123"],
            impact_area=["payments", "transactions"],
            confidence=0.95,
        )
    ]

    # Event 2: Database Incident
    llm_events[messages[1].message_id] = [
        LLMEvent(
            action="rollback",
            object_name="Database Migration",
            category="risk",
            status="in_progress",
            change_type="rollback",
            environment="prod",
            severity="critical",
            actual_start="now",
            summary="Database migration failed in production, rolling back",
            anchors=["PROD-456"],
            impact_area=["database", "production"],
            confidence=0.98,
        )
    ]

    # Event 3: Marketing Campaign
    llm_events[messages[2].message_id] = [
        LLMEvent(
            action="launch",
            object_name="Summer Sale 2025",
            category="marketing",
            status="planned",
            change_type="launch",
            environment="unknown",
            planned_start="next week",
            summary="New marketing campaign: Summer Sale 2025",
            links=["https://docs.google.com/presentation/d/abc123"],
            anchors=["gdoc:abc123"],
            impact_area=["marketing", "sales"],
            confidence=0.85,
        )
    ]

    return llm_events


def test_pipeline() -> None:
    """Test full pipeline with PostgreSQL."""
    logger.info("=" * 80)
    logger.info("Starting PostgreSQL Pipeline Test")
    logger.info("=" * 80)

    # Get settings and repository
    settings = get_settings()
    logger.info(f"Database type: {settings.database_type}")

    if settings.database_type != "postgres":
        logger.error("DATABASE_TYPE must be 'postgres' for this test")
        sys.exit(1)

    repo = create_repository(settings)
    if not isinstance(repo, PostgresRepository):
        logger.error("Repository is not PostgresRepository")
        sys.exit(1)

    logger.info("✓ PostgreSQL repository initialized")

    # Step 1: Create and ingest mock messages
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Ingesting mock messages")
    logger.info("=" * 80)

    messages = create_mock_messages()
    logger.info(f"Created {len(messages)} mock messages")

    saved_count = repo.save_messages(messages)
    logger.info(f"✓ Ingested {saved_count} messages")

    for msg in messages:
        logger.info(f"  - {msg.message_id[:16]}... - {msg.text[:60]}...")

    # Step 2: Build candidates
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Building candidates")
    logger.info("=" * 80)

    candidates = []
    for msg in messages:
        normalized_text = normalize_text(msg.text)
        features = ScoringFeatures(
            anchor_count=len(msg.anchors),
            link_count=len(msg.links_raw),
            reaction_count=msg.total_reactions,
            reply_count=msg.reply_count or 0,
            mention_count=0,
            keyword_count=0,
        )
        # Calculate simple score based on features
        score = (
            features.anchor_count * 4.0
            + features.link_count * 2.0
            + features.reaction_count * 0.5
            + features.reply_count * 1.0
        )

        candidate = EventCandidate(
            candidate_id=uuid4(),
            message_id=msg.message_id,
            channel_name="releases",  # Mock channel name
            normalized_text=normalized_text,
            features=features,
            score=score,
            status=CandidateStatus.NEW,
        )
        candidates.append(candidate)
        logger.info(
            f"  - Created candidate: {candidate.candidate_id} (score: {score:.2f})"
        )

    saved_count = repo.save_candidates(candidates)
    logger.info(f"✓ Saved {saved_count} candidates")

    # Step 3: Mock LLM extraction and convert to Events
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Extracting events (mocked LLM)")
    logger.info("=" * 80)

    llm_events_map = create_mock_llm_events(messages)

    from src.use_cases.extract_events import convert_llm_event_to_event

    events = []
    for msg in messages:
        if msg.message_id in llm_events_map:
            for llm_event in llm_events_map[msg.message_id]:
                event = convert_llm_event_to_event(
                    llm_event=llm_event,
                    message_id=msg.message_id,
                    message_ts_dt=msg.ts_dt,
                    channel_name="releases",
                )

                # Generate cluster and dedup keys
                event.cluster_key = generate_cluster_key(event)
                event.dedup_key = generate_dedup_key(event)

                events.append(event)
                logger.info(
                    f"  - Extracted event: {event.action.value} {event.object_name_raw} "
                    f"(category: {event.category.value}, confidence: {event.confidence:.2f})"
                )

    saved_count = repo.save_events(events)
    logger.info(f"✓ Saved {saved_count} events")

    # Step 4: Verify data in PostgreSQL
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Verifying data in PostgreSQL")
    logger.info("=" * 80)

    # Check messages
    all_messages = repo.get_all_messages()
    logger.info(f"✓ Messages in DB: {len(all_messages)}")
    assert len(all_messages) >= len(messages), "Not all messages were saved"

    # Check candidates
    all_candidates = repo.get_candidates_for_extraction(batch_size=None)
    logger.info(f"✓ Candidates in DB: {len(all_candidates)}")
    assert len(all_candidates) >= len(candidates), "Not all candidates were saved"

    # Check events
    start_date = datetime.now(pytz.UTC) - timedelta(days=1)
    end_date = datetime.now(pytz.UTC) + timedelta(days=7)
    all_events = repo.get_events_in_window(start_date, end_date)
    logger.info(f"✓ Events in DB: {len(all_events)}")
    assert len(all_events) >= len(events), "Not all events were saved"

    # Print event details
    logger.info("\nEvent Details:")
    for event in all_events:
        logger.info(
            f"  - {event.action.value} {event.object_name_raw} "
            f"({event.category.value}, confidence: {event.confidence:.2f}, "
            f"importance: {event.importance})"
        )

    # Step 5: Test deduplication
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Testing deduplication")
    logger.info("=" * 80)

    # Add a duplicate event
    import copy

    duplicate_event = copy.deepcopy(events[0])
    duplicate_event.event_id = uuid4()  # New ID but same content
    repo.save_events([duplicate_event])
    logger.info("✓ Added duplicate event")

    # Run deduplication
    deduplicated = deduplicate_events_use_case(
        repository=repo,
        date_window_hours=48,
        title_similarity_threshold=0.8,
    )
    logger.info(f"✓ Deduplicated {deduplicated} events")

    # Final verification
    logger.info("\n" + "=" * 80)
    logger.info("Final Statistics")
    logger.info("=" * 80)

    final_events = repo.get_events_in_window(start_date, end_date)
    logger.info(f"Total messages: {len(all_messages)}")
    logger.info(f"Total candidates: {len(all_candidates)}")
    logger.info(f"Total events: {len(final_events)}")
    logger.info(f"Events deduplicated: {deduplicated}")

    # Category breakdown
    category_counts = {}
    for event in final_events:
        cat = event.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info("\nEvents by category:")
    for cat, count in sorted(category_counts.items()):
        logger.info(f"  {cat}: {count}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ PostgreSQL Pipeline Test PASSED")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        logger.error(f"❌ Pipeline test FAILED: {e}", exc_info=True)
        sys.exit(1)
