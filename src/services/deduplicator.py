"""Event deduplication service.

Rules:
1. Events from same message_id NEVER merge
2. Inter-message merge if:
   - Anchor/link overlap
   - Date delta <= 48 hours (configurable)
   - Fuzzy title similarity >= 0.8 (configurable)
"""

import hashlib

from rapidfuzz import fuzz

from src.domain.deduplication_constants import (
    DEFAULT_DATE_WINDOW_HOURS,
    DEFAULT_TITLE_SIMILARITY,
    SAME_MESSAGE_NO_MERGE,
)
from src.domain.models import Event


def generate_dedup_key(event: Event) -> str:
    """Generate deterministic dedup key for event.

    Format: sha1(event_date || title[:80].lower() || top_anchor_or_null)

    Args:
        event: Event to generate key for

    Returns:
        SHA1 hex digest

    Example:
        >>> evt = Event(event_date=datetime(...), title="Release v1", ...)
        >>> generate_dedup_key(evt)
        'a3f2b1c0...'
    """
    # Canonical date (ISO format)
    date_str = event.event_date.isoformat()

    # Normalized title (first 80 chars, lowercase)
    title_norm = event.title[:80].lower().strip()

    # Top anchor (first one if available, else empty)
    top_anchor = event.anchors[0] if event.anchors else ""

    # Concatenate and hash
    key_material = f"{date_str}||{title_norm}||{top_anchor}"
    return hashlib.sha1(key_material.encode("utf-8")).hexdigest()


def has_overlap(list1: list[str], list2: list[str]) -> bool:
    """Check if two lists have any common elements.

    Args:
        list1: First list
        list2: Second list

    Returns:
        True if intersection is non-empty

    Example:
        >>> has_overlap(["a", "b"], ["b", "c"])
        True
        >>> has_overlap(["a"], ["b"])
        False
    """
    return bool(set(list1) & set(list2))


def should_merge_events(
    event1: Event,
    event2: Event,
    date_window_hours: int = DEFAULT_DATE_WINDOW_HOURS,
    title_similarity_threshold: float = DEFAULT_TITLE_SIMILARITY,
) -> bool:
    """Determine if two events should be merged.

    Rules:
    - Same message_id: NO merge (Rule 1)
    - Different message_id:
      - Must have anchor/link overlap
      - Date delta <= window
      - Title similarity >= threshold

    Args:
        event1: First event
        event2: Second event
        date_window_hours: Maximum date difference in hours (default: 48)
        title_similarity_threshold: Minimum fuzzy similarity 0.0-1.0 (default: 0.8)

    Returns:
        True if events should merge

    Example:
        >>> evt1 = Event(message_id="m1", title="Release v1.0", ...)
        >>> evt2 = Event(message_id="m2", title="Release v1.0", ...)
        >>> should_merge_events(evt1, evt2)
        True

    Note:
        Default values come from src.domain.deduplication_constants.
        In production, pass values from Settings configuration.
    """
    # Rule 1: Same message_id = NO merge
    if SAME_MESSAGE_NO_MERGE and event1.message_id == event2.message_id:
        return False

    # Check anchor/link overlap
    combined_anchors1 = event1.anchors + event1.links
    combined_anchors2 = event2.anchors + event2.links

    if not has_overlap(combined_anchors1, combined_anchors2):
        return False

    # Check date delta
    date_delta = abs((event1.event_date - event2.event_date).total_seconds() / 3600)
    if date_delta > date_window_hours:
        return False

    # Check fuzzy title similarity
    similarity = fuzz.ratio(event1.title.lower(), event2.title.lower()) / 100.0
    if similarity < title_similarity_threshold:
        return False

    return True


def merge_events(event1: Event, event2: Event) -> Event:
    """Merge two events, combining attributes.

    Strategy:
    - Union: links, tags, source_channels, anchors
    - Max: confidence, version
    - Keep: first event's core attributes (title, summary, date)

    Args:
        event1: Primary event (keeps core attributes)
        event2: Secondary event (contributes additional data)

    Returns:
        Merged event with incremented version

    Example:
        >>> evt1 = Event(links=["a"], confidence=0.8, version=1)
        >>> evt2 = Event(links=["b"], confidence=0.9, version=1)
        >>> merged = merge_events(evt1, evt2)
        >>> merged.version
        2
        >>> set(merged.links)
        {'a', 'b'}
    """
    # Union of lists (deduplicated)
    merged_links = list(set(event1.links + event2.links))[:3]  # Max 3
    merged_tags = list(set(event1.tags + event2.tags))
    merged_channels = list(set(event1.source_channels + event2.source_channels))
    merged_anchors = list(set(event1.anchors + event2.anchors))

    # Max values
    max_confidence = max(event1.confidence, event2.confidence)
    max_version = max(event1.version, event2.version) + 1

    # Create merged event (keeping event1 as base)
    merged = Event(
        event_id=event1.event_id,  # Keep primary ID
        version=max_version,
        message_id=event1.message_id,
        source_msg_event_idx=event1.source_msg_event_idx,
        dedup_key=event1.dedup_key,
        event_date=event1.event_date,
        event_end=event1.event_end or event2.event_end,
        category=event1.category,
        title=event1.title,
        summary=event1.summary,
        impact_area=list(set(event1.impact_area + event2.impact_area)),
        tags=merged_tags,
        links=merged_links,
        anchors=merged_anchors,
        confidence=max_confidence,
        source_channels=merged_channels,
        ingested_at=event1.ingested_at,
    )

    return merged


def find_merge_candidates(
    new_event: Event,
    existing_events: list[Event],
    date_window_hours: int = DEFAULT_DATE_WINDOW_HOURS,
    title_similarity_threshold: float = DEFAULT_TITLE_SIMILARITY,
) -> list[Event]:
    """Find existing events that should merge with new event.

    Args:
        new_event: Event to check
        existing_events: Pool of existing events
        date_window_hours: Date window for consideration
        title_similarity_threshold: Fuzzy match threshold

    Returns:
        List of events that should merge

    Example:
        >>> new_evt = Event(...)
        >>> existing = [Event(...), Event(...)]
        >>> candidates = find_merge_candidates(new_evt, existing)
        >>> len(candidates)
        1
    """
    candidates = []

    for existing in existing_events:
        if should_merge_events(
            new_event,
            existing,
            date_window_hours=date_window_hours,
            title_similarity_threshold=title_similarity_threshold,
        ):
            candidates.append(existing)

    return candidates


def deduplicate_event_list(
    events: list[Event],
    date_window_hours: int = DEFAULT_DATE_WINDOW_HOURS,
    title_similarity_threshold: float = DEFAULT_TITLE_SIMILARITY,
) -> list[Event]:
    """Deduplicate a list of events in-memory.

    Args:
        events: List of events to deduplicate
        date_window_hours: Date window
        title_similarity_threshold: Fuzzy threshold

    Returns:
        Deduplicated list of events

    Example:
        >>> events = [Event(...), Event(...), Event(...)]
        >>> deduped = deduplicate_event_list(events)
        >>> len(deduped) < len(events)
        True
    """
    if not events:
        return []

    # Sort by date to process chronologically
    sorted_events = sorted(events, key=lambda e: e.event_date)

    deduplicated: list[Event] = []
    processed_dedup_keys: set[str] = set()

    for event in sorted_events:
        # Check if already processed (by dedup_key)
        if event.dedup_key in processed_dedup_keys:
            continue

        # Find merge candidates in already processed events
        merge_candidates = find_merge_candidates(
            event,
            deduplicated,
            date_window_hours=date_window_hours,
            title_similarity_threshold=title_similarity_threshold,
        )

        if merge_candidates:
            # Merge with first candidate, remove it, add merged
            target = merge_candidates[0]
            deduplicated.remove(target)
            processed_dedup_keys.discard(target.dedup_key)

            merged = merge_events(target, event)
            deduplicated.append(merged)
            processed_dedup_keys.add(merged.dedup_key)
        else:
            # No merge needed, add as new
            deduplicated.append(event)
            processed_dedup_keys.add(event.dedup_key)

    return deduplicated
