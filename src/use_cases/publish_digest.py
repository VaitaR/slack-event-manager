"""Publish digest use case.

Generates and posts daily event digest to Slack.
"""

from datetime import datetime, timedelta
from typing import Any

import pytz

from src.adapters.slack_client import SlackClient
from src.config.settings import Settings
from src.domain.models import DigestResult, Event, EventCategory
from src.domain.protocols import RepositoryProtocol
from src.services.title_renderer import TitleRenderer

# Initialize title renderer
_title_renderer = TitleRenderer()


def format_event_date(event_date: datetime, tz_name: str = "Europe/Amsterdam") -> str:
    """Format event date for display.

    Args:
        event_date: Event datetime in UTC
        tz_name: Target timezone

    Returns:
        Formatted date string

    Example:
        >>> dt = datetime(2025, 10, 15, 8, 0, tzinfo=pytz.UTC)
        >>> format_event_date(dt, "Europe/Amsterdam")
        '15.10.2025 10:00'
    """
    try:
        tz = pytz.timezone(tz_name)
        local_dt = event_date.astimezone(tz)
        return local_dt.strftime("%d.%m.%Y %H:%M")
    except Exception:
        return event_date.strftime("%d.%m.%Y %H:%M UTC")


def get_confidence_icon(confidence: float) -> str:
    """Get emoji icon for confidence level.

    Args:
        confidence: Confidence score (0.0-1.0)

    Returns:
        Emoji string

    Example:
        >>> get_confidence_icon(0.95)
        '✅'
        >>> get_confidence_icon(0.65)
        '⚠️'
    """
    if confidence >= 0.8:
        return "✅"
    elif confidence >= 0.6:
        return "⚠️"
    else:
        return "❓"


def build_event_block(event: Event, tz_name: str) -> dict[str, Any]:
    """Build Slack Block Kit block for an event.

    Args:
        event: Event to format
        tz_name: Timezone for date display

    Returns:
        Block dictionary

    Example:
        >>> evt = Event(...)
        >>> block = build_event_block(evt, "Europe/Amsterdam")
        >>> block["type"]
        'section'
    """
    # Format category emoji
    category_emoji = {
        EventCategory.PRODUCT: "🚀",
        EventCategory.RISK: "⚠️",
        EventCategory.PROCESS: "⚙️",
        EventCategory.MARKETING: "📢",
        EventCategory.ORG: "👥",
        EventCategory.UNKNOWN: "❓",
    }

    emoji = category_emoji.get(event.category, "•")

    # Render title from slots using TitleRenderer
    # Use severity-first format for risk events, otherwise canonical
    format_style = (
        "severity_first" if event.category == EventCategory.RISK else "canonical"
    )
    title = _title_renderer.render_canonical_title(event, format_style=format_style)

    # Compact format: only category emoji + title
    text = f"{emoji} {title}"

    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _get_event_primary_time(event: Event) -> datetime:
    """Get primary time for event (prefer actual, fallback to planned).

    Args:
        event: Event to get time for

    Returns:
        Primary datetime
    """
    # Prefer actual times (started/completed events)
    if event.actual_start:
        return event.actual_start
    if event.actual_end:
        return event.actual_end
    # Fallback to planned times
    if event.planned_start:
        return event.planned_start
    if event.planned_end:
        return event.planned_end
    # Last resort: use extraction time
    return event.extracted_at


def sort_events_for_digest(
    events: list[Event], category_priorities: dict[str, int] | None = None
) -> list[Event]:
    """Sort events for digest display.

    Primary: primary_time (actual > planned) ASC
    Secondary: category priority
    Tertiary: importance DESC (changed from confidence)

    Args:
        events: List of events
        category_priorities: Category priority mapping (lower = higher priority)

    Returns:
        Sorted list

    Example:
        >>> events = [Event(...), Event(...)]
        >>> sorted_events = sort_events_for_digest(events)
    """
    if category_priorities is None:
        category_priorities = {
            "product": 1,
            "risk": 2,
            "process": 3,
            "marketing": 4,
            "org": 5,
            "unknown": 6,
        }

    return sorted(
        events,
        key=lambda e: (
            _get_event_primary_time(e),
            category_priorities.get(e.category.value, 99),
            -e.importance,  # Higher importance first
        ),
    )


def build_digest_blocks(
    events: list[Event], date_str: str, tz_name: str
) -> list[dict[str, Any]]:
    """Build Slack Block Kit blocks for digest.

    Args:
        events: Events to include
        date_str: Date string for header
        tz_name: Timezone

    Returns:
        List of blocks

    Example:
        >>> blocks = build_digest_blocks(events, "15.10.2025", "Europe/Amsterdam")
        >>> len(blocks) > 0
        True
    """
    blocks: list[dict[str, Any]] = []

    # Header
    blocks.append(
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Events {date_str}"},
        }
    )

    blocks.append({"type": "divider"})

    if not events:
        # No events message
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "_No new events._"},
            }
        )
    else:
        # Add event blocks
        for event in events:
            blocks.append(build_event_block(event, tz_name))

        # Footer with count
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Total events: *{len(events)}*",
                    }
                ],
            }
        )

    return blocks


def chunk_blocks(
    blocks: list[dict[str, Any]], max_blocks: int = 50
) -> list[list[dict[str, Any]]]:
    """Chunk blocks to respect Slack limits.

    Args:
        blocks: All blocks
        max_blocks: Maximum blocks per message

    Returns:
        List of block chunks

    Example:
        >>> blocks = [{"type": "section"}, ...]
        >>> chunks = chunk_blocks(blocks, max_blocks=50)
        >>> len(chunks)
        1
    """
    if len(blocks) <= max_blocks:
        return [blocks]

    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []

    for block in blocks:
        current_chunk.append(block)

        if len(current_chunk) >= max_blocks:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def publish_digest_use_case(
    slack_client: SlackClient,
    repository: RepositoryProtocol,
    settings: Settings,
    lookback_hours: int | None = None,
    target_channel: str | None = None,
    dry_run: bool = False,
    min_confidence: float | None = None,
    max_events: int | None = None,
) -> DigestResult:
    """Publish daily event digest to Slack.

    1. Query events for date range with filtering
    2. Sort by date, category, confidence
    3. Apply max events limit
    4. Build Slack Block Kit
    5. Chunk if needed
    6. Post to channel

    Args:
        slack_client: Slack client
        repository: Repository protocol implementation
        settings: Application settings
        lookback_hours: Hours to look back (defaults to settings.digest_lookback_hours)
        target_channel: Override digest channel
        dry_run: If True, don't post, just build
        min_confidence: Minimum confidence score (defaults to settings.digest_min_confidence)
        max_events: Maximum events to include (defaults to settings.digest_max_events)

    Returns:
        DigestResult with counts

    Example:
        >>> result = publish_digest_use_case(slack, repo, settings)
        >>> result.events_included
        10
        >>> result = publish_digest_use_case(
        ...     slack, repo, settings, min_confidence=0.8, max_events=20
        ... )
        >>> result.events_included <= 20
        True
    """
    # Use settings defaults if not provided
    if lookback_hours is None:
        lookback_hours = settings.digest_lookback_hours
    if min_confidence is None:
        min_confidence = settings.digest_min_confidence
    if max_events is None:
        max_events = settings.digest_max_events

    # Get events from window
    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    start_dt = now - timedelta(hours=lookback_hours)
    end_dt = now

    # Fetch events with database-level filtering
    events = repository.get_events_in_window_filtered(
        start_dt=start_dt,
        end_dt=end_dt,
        min_confidence=min_confidence,
        max_events=None,  # Apply limit after sorting for better prioritization
    )

    # Sort events by priority
    sorted_events = sort_events_for_digest(
        events, category_priorities=settings.digest_category_priorities
    )

    # Apply max events limit after sorting
    if max_events is not None and len(sorted_events) > max_events:
        sorted_events = sorted_events[:max_events]

    # Build blocks
    date_str = now.astimezone(pytz.timezone(settings.tz_default)).strftime("%d.%m.%Y")
    blocks = build_digest_blocks(sorted_events, date_str, settings.tz_default)

    # Chunk blocks
    block_chunks = chunk_blocks(blocks, max_blocks=50)

    # Post to Slack
    channel = target_channel or settings.slack_digest_channel_id
    messages_posted = 0

    if not dry_run:
        for chunk in block_chunks:
            try:
                slack_client.post_message(channel, chunk)
                messages_posted += 1
            except Exception as e:
                # Log error but continue
                print(f"Failed to post digest chunk: {e}")

    return DigestResult(
        messages_posted=messages_posted,
        events_included=len(sorted_events),
        channel=channel,
    )
