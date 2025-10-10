"""Publish digest use case.

Generates and posts daily event digest to Slack.
"""

from datetime import datetime, timedelta
from typing import Any

import pytz

from src.adapters.slack_client import SlackClient
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import Settings
from src.domain.models import DigestResult, Event, EventCategory

# Category priorities (lower = higher priority)
CATEGORY_PRIORITY = {
    EventCategory.PRODUCT: 1,
    EventCategory.RISK: 2,
    EventCategory.PROCESS: 3,
    EventCategory.MARKETING: 4,
    EventCategory.ORG: 5,
    EventCategory.UNKNOWN: 6,
}


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

    # Format date
    date_str = format_event_date(event.event_date, tz_name)
    if event.event_end:
        end_str = format_event_date(event.event_end, tz_name)
        date_str = f"{date_str} - {end_str}"

    # Build title line
    confidence_icon = get_confidence_icon(event.confidence)
    title_line = f"{emoji} *{event.title}*"
    if event.confidence < 0.7:
        title_line += f" {confidence_icon}"

    # Build detail line
    details = [date_str]

    # Add links (max 3)
    if event.links:
        for link in event.links[:3]:
            # Extract domain for display
            try:
                from urllib.parse import urlparse
                domain = urlparse(link).netloc or link
                details.append(f"<{link}|{domain}>")
            except Exception:
                details.append(f"<{link}|link>")

    detail_line = " · ".join(details)

    # Build summary (optional, if not too long)
    summary_line = ""
    if event.summary and len(event.summary) <= 200:
        summary_line = f"\n_{event.summary}_"

    text = f"{title_line}\n{detail_line}{summary_line}"

    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def sort_events_for_digest(events: list[Event]) -> list[Event]:
    """Sort events for digest display.

    Primary: event_date ASC
    Secondary: category priority
    Tertiary: confidence DESC

    Args:
        events: List of events

    Returns:
        Sorted list

    Example:
        >>> events = [Event(...), Event(...)]
        >>> sorted_events = sort_events_for_digest(events)
    """
    return sorted(
        events,
        key=lambda e: (
            e.event_date,
            CATEGORY_PRIORITY.get(e.category, 99),
            -e.confidence,
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
            "text": {"type": "plain_text", "text": f"События {date_str}"},
        }
    )

    blocks.append({"type": "divider"})

    if not events:
        # No events message
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "_Нет новых событий._"},
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
                        "text": f"Всего событий: *{len(events)}*",
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
    repository: SQLiteRepository,
    settings: Settings,
    lookback_hours: int = 48,
    target_channel: str | None = None,
    dry_run: bool = False,
) -> DigestResult:
    """Publish daily event digest to Slack.

    1. Query events for date range
    2. Sort by date, category, confidence
    3. Build Slack Block Kit
    4. Chunk if needed
    5. Post to channel

    Args:
        slack_client: Slack client
        repository: Data repository
        settings: Application settings
        lookback_hours: Hours to look back
        target_channel: Override digest channel
        dry_run: If True, don't post, just build

    Returns:
        DigestResult with counts

    Example:
        >>> result = publish_digest_use_case(slack, repo, settings)
        >>> result.events_included
        23
    """
    # Get events from window
    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    start_dt = now - timedelta(hours=lookback_hours)
    end_dt = now

    events = repository.get_events_in_window(start_dt, end_dt)

    # Sort events
    sorted_events = sort_events_for_digest(events)

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

