"""Business rules and constants for event deduplication.

This module defines the core business rules used to determine when two events
should be merged together. All deduplication thresholds and rules are centralized
here to ensure consistency across the application.
"""

from typing import Final

# Deduplication time window
DEFAULT_DATE_WINDOW_HOURS: Final[int] = 48
"""Maximum time difference in hours between events for merge consideration.

Business rule: Events occurring within 48 hours (2 days) can be merged if they
have overlapping links/anchors and similar titles. This allows consolidating
announcements that are posted multiple times or updated within a short period.

Example:
    - Event A: "Feature released" at 2025-10-01 10:00
    - Event B: "Feature released (updated)" at 2025-10-02 11:00
    - Time delta: 25 hours → Can merge (< 48h)
"""

# Title similarity threshold
DEFAULT_TITLE_SIMILARITY: Final[float] = 0.8
"""Minimum fuzzy title similarity score for merging (0.0-1.0).

Uses RapidFuzz ratio algorithm to compare title strings. A score of 0.8 means
80% similarity is required. This prevents merging of unrelated events while
allowing minor wording differences.

Business rule: Titles must be at least 80% similar to consider merging events.

Example:
    - "New wallet feature released" vs "New wallet feature rollout"
    - Similarity: 0.85 → Can merge (≥ 0.8)

    - "Wallet update" vs "Backend migration"
    - Similarity: 0.3 → Cannot merge (< 0.8)
"""

# Deduplication rules
SAME_MESSAGE_NO_MERGE: Final[bool] = True
"""Rule 1: Events from same message NEVER merge, even if similar.

Business rule: If multiple events are extracted from a single Slack message,
they represent distinct events that were intentionally mentioned together.
Merging them would lose information.

Example from single message:
    "We released Feature A on Monday and Feature B on Tuesday"
    → Event 1: Feature A (Monday)
    → Event 2: Feature B (Tuesday)
    → Do NOT merge (same message_id)
"""

# Message processing window
DEFAULT_MESSAGE_LOOKBACK_DAYS: Final[int] = 7
"""How many days back to process messages when running deduplication.

This controls which messages are fetched and processed, not which events
are considered for merging. Events can have dates far in the past, but
we only process recently sent messages.

Business rule: Process messages from last 7 days to capture new announcements
while avoiding reprocessing the entire history on each run.

Example:
    - Today: 2025-10-10
    - Process messages from: 2025-10-03 onwards
    - Message dated 2025-10-05 says "We released X on 2025-09-01"
      → Message is processed (within 7 days)
      → Event date is 2025-09-01 (can be old, that's OK)
"""
