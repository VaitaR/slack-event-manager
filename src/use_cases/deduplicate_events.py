"""Deduplicate events use case.

Applies deduplication rules to merge similar events.
"""

import sys
from datetime import datetime, timedelta

import pytz

from src.adapters.query_builders import EventQueryCriteria
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import Settings
from src.domain.models import DeduplicationResult, MessageSource
from src.services import deduplicator


def deduplicate_events_use_case(
    repository: SQLiteRepository,
    settings: Settings,
    lookback_days: int = 7,
    source_id: MessageSource | None = None,
) -> DeduplicationResult:
    """Deduplicate events within lookback window.

    1. Fetch all events from lookback window (optionally filtered by source)
    2. For each event:
       a. Check: same message_id with other events? → no merge (Rule 1)
       b. Find merge candidates: anchor/link overlap + date Δ
       c. Fuzzy match title (rapidfuzz ≥0.8)
       d. If merge: combine attributes, increment version
       e. Generate dedup_key
       f. Upsert to events table
    3. Return counts

    Args:
        repository: Data repository
        settings: Application settings
        lookback_days: Days to look back for deduplication
        source_id: Optional source filter for strict isolation (prevents cross-source merging)

    Returns:
        DeduplicationResult with counts

    Example:
        >>> # Deduplicate all events
        >>> result = deduplicate_events_use_case(repo, settings)
        >>> result.merged_events
        3

        >>> # Deduplicate only Slack events (strict isolation)
        >>> result = deduplicate_events_use_case(repo, settings, source_id=MessageSource.SLACK)
    """
    # Get events from lookback window using Criteria pattern
    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    extracted_after = now - timedelta(days=lookback_days)

    # Use EventQueryCriteria with optional source filtering
    criteria = EventQueryCriteria(
        extracted_after=extracted_after,
        source_id=source_id.value
        if source_id
        else None,  # Filter by source if provided
        order_by="extracted_at",
        order_desc=False,  # Chronological order
    )

    all_events = repository.query_events(criteria)

    if not all_events:
        return DeduplicationResult(new_events=0, merged_events=0, total_events=0)

    initial_count = len(all_events)

    # Log initial state
    print("\n🔍 Deduplication Analysis:")
    print(f"   Initial events: {initial_count}")
    if source_id:
        print(f"   Source filter: {source_id.value} (strict isolation)")
    else:
        print("   Source filter: None (all sources)")
    print(f"   Date window: {settings.dedup_date_window_hours} hours")
    print(f"   Title similarity threshold: {settings.dedup_title_similarity}")
    print("")

    # Show all events before deduplication
    print("   📋 Events BEFORE deduplication:")
    for i, evt in enumerate(all_events, 1):
        title = f"{evt.action.value}: {evt.object_name_raw}"
        primary_time = (
            evt.actual_start
            or evt.actual_end
            or evt.planned_start
            or evt.planned_end
            or evt.extracted_at
        )
        print(f"   {i}. {title[:60]}")
        print(f"      Message ID: {evt.message_id[:8]}...")
        print(f"      Date: {primary_time.isoformat()}")
        print(f"      Links: {evt.links[:2]}")  # Show first 2 links
        print(f"      Dedup key: {evt.dedup_key[:16]}...")
        print("")
    sys.stdout.flush()

    # Deduplicate events with detailed logging
    print("   🔄 Running deduplication...")
    sys.stdout.flush()

    deduplicated_events = deduplicator.deduplicate_event_list(
        all_events,
        date_window_hours=settings.dedup_date_window_hours,
        title_similarity_threshold=settings.dedup_title_similarity,
    )

    final_count = len(deduplicated_events)
    merged_count = initial_count - final_count

    print("\n   ✅ Deduplication complete:")
    print(f"      Initial: {initial_count}")
    print(f"      Final: {final_count}")
    print(f"      Merged: {merged_count}")
    print("")

    # Show events after deduplication
    print("   📋 Events AFTER deduplication:")
    for i, evt in enumerate(deduplicated_events, 1):
        title = f"{evt.action.value}: {evt.object_name_raw}"
        primary_time = (
            evt.actual_start
            or evt.actual_end
            or evt.planned_start
            or evt.planned_end
            or evt.extracted_at
        )
        print(f"   {i}. {title[:60]}")
        print(f"      Message ID: {evt.message_id[:8]}...")
        print(f"      Date: {primary_time.isoformat()}")
        print(f"      Channels: {evt.source_channels}")
        print(f"      Importance: {evt.importance}")
        print("")
    sys.stdout.flush()

    # Save deduplicated events (they will be upserted)
    if deduplicated_events:
        repository.save_events(deduplicated_events)

    return DeduplicationResult(
        new_events=final_count - merged_count,
        merged_events=merged_count,
        total_events=final_count,
    )
