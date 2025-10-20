"""Ingest messages use case.

Fetches messages from Slack channels and stores them with normalization.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Any

import pytz

from src.adapters.slack_client import SlackClient
from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.domain.models import IngestResult, SlackMessage
from src.domain.protocols import RepositoryProtocol
from src.services import link_extractor, text_normalizer

logger = get_logger(__name__)


def generate_message_id(channel: str, ts: str) -> str:
    """Generate deterministic message ID.

    Args:
        channel: Channel ID
        ts: Message timestamp

    Returns:
        SHA1 hash of channel|ts

    Example:
        >>> generate_message_id("C123", "1234567890.123456")
        'a3f2b1c0...'
    """
    key_material = f"{channel}|{ts}"
    return hashlib.sha1(key_material.encode("utf-8")).hexdigest()


def process_slack_message(
    raw_msg: dict[str, Any],
    channel_id: str,
    user_info: dict[str, Any] | None = None,
    permalink: str | None = None,
) -> SlackMessage:
    """Process raw Slack message into domain model.

    Args:
        raw_msg: Raw Slack message dictionary
        channel_id: Channel ID
        user_info: Optional user info from Slack API
        permalink: Optional permalink URL

    Returns:
        Processed SlackMessage
    """
    # Extract basic fields
    ts = raw_msg.get("ts", "")
    ts_dt = (
        datetime.fromtimestamp(float(ts), tz=pytz.UTC)
        if ts
        else datetime.utcnow().replace(tzinfo=pytz.UTC)
    )
    user = raw_msg.get("user")
    bot_id = raw_msg.get("bot_id")
    is_bot = bool(bot_id or raw_msg.get("subtype") == "bot_message")
    subtype = raw_msg.get("subtype")
    text = raw_msg.get("text", "")

    # Extract user information
    user_real_name: str | None = None
    user_display_name: str | None = None
    user_email: str | None = None
    user_profile_image: str | None = None

    if user_info:
        profile = user_info.get("profile", {})
        user_real_name = user_info.get("real_name") or profile.get("real_name")
        user_display_name = profile.get("display_name") or user_info.get("name")
        user_email = profile.get("email")
        # Get largest available profile image
        user_profile_image = (
            profile.get("image_512")
            or profile.get("image_192")
            or profile.get("image_72")
            or profile.get("image_48")
        )

    # Extract blocks text
    blocks = raw_msg.get("blocks", [])
    blocks_text = text_normalizer.extract_blocks_text(blocks)

    # Combine text sources
    combined_text = text_normalizer.combine_text_sources(text, blocks_text)

    # Extract links and anchors
    links_raw = link_extractor.extract_urls(combined_text)
    links_norm = link_extractor.normalize_links(links_raw)
    anchors = link_extractor.extract_all_anchors(combined_text)

    # Normalize text
    text_norm = text_normalizer.normalize_text(combined_text)

    # Extract attachments and files count
    attachments_count = len(raw_msg.get("attachments", []))
    files_count = len(raw_msg.get("files", []))

    # Extract reactions
    reactions: dict[str, int] = {}
    total_reactions = 0
    for reaction in raw_msg.get("reactions", []):
        emoji = reaction.get("name", "")
        count = reaction.get("count", 0)
        if emoji:
            reactions[emoji] = count
            total_reactions += count

    # Reply count
    reply_count = raw_msg.get("reply_count", 0)

    # Extract edit information
    edited_info = raw_msg.get("edited")
    edited_ts: str | None = None
    edited_user: str | None = None
    if edited_info:
        edited_ts = edited_info.get("ts")
        edited_user = edited_info.get("user")

    # Generate message ID
    message_id = generate_message_id(channel_id, ts)

    return SlackMessage(
        message_id=message_id,
        channel=channel_id,
        ts=ts,
        ts_dt=ts_dt,
        user=user,
        user_real_name=user_real_name,
        user_display_name=user_display_name,
        user_email=user_email,
        user_profile_image=user_profile_image,
        is_bot=is_bot,
        bot_id=bot_id,
        subtype=subtype,
        text=text,
        blocks_text=blocks_text,
        text_norm=text_norm,
        links_raw=links_raw,
        links_norm=links_norm,
        anchors=anchors,
        attachments_count=attachments_count,
        files_count=files_count,
        reactions=reactions,
        total_reactions=total_reactions,
        reply_count=reply_count,
        permalink=permalink,
        edited_ts=edited_ts,
        edited_user=edited_user,
        ingested_at=datetime.utcnow().replace(tzinfo=pytz.UTC),
    )


def ingest_messages_use_case(
    slack_client: SlackClient,
    repository: RepositoryProtocol,
    settings: Settings,
    lookback_hours: int | None = None,
    backfill_from_date: datetime | None = None,
) -> IngestResult:
    """Ingest messages from configured Slack channels.

    For each channel:
    1. Get ingestion state (last_ts)
    2. Determine oldest timestamp:
       - If state exists: use last_ts (incremental)
       - If no state and backfill_from_date: use that date
       - If no state and no backfill: use 30 days ago (default)
    3. Fetch messages from oldest to now
    4. Normalize text
    5. Extract links & anchors
    6. Generate message_id
    7. Save to raw_slack_messages
    8. Update ingestion_state with max timestamp

    Args:
        slack_client: Slack client
        repository: Repository protocol implementation
        settings: Application settings
        lookback_hours: Override default lookback (deprecated, use backfill_from_date)
        backfill_from_date: Optional date to start backfill from (first run only)

    Returns:
        IngestResult with counts and errors

    Example:
        >>> result = ingest_messages_use_case(client, repo, settings)
        >>> result.messages_saved
        42
    """
    if lookback_hours is None:
        lookback_hours = settings.lookback_hours_default

    total_fetched = 0
    total_saved = 0
    channels_processed: list[str] = []
    errors: list[str] = []

    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    now_ts = now.timestamp()

    # Default backfill window: 30 days
    default_backfill_days = 30

    for channel_config in settings.slack_channels:
        channel_id = channel_config.channel_id

        try:
            # Get ingestion state
            last_ts = repository.get_last_processed_ts(channel_id)

            if last_ts is not None:
                # Incremental: use last processed timestamp
                oldest_ts = str(last_ts)
                logger.info(
                    "slack_ingestion_incremental",
                    channel_id=channel_id,
                    oldest_ts=oldest_ts,
                    strategy="incremental",
                )
            elif backfill_from_date:
                # First run with explicit backfill date
                oldest_ts = str(backfill_from_date.timestamp())
                logger.info(
                    "slack_ingestion_backfill",
                    channel_id=channel_id,
                    backfill_date=backfill_from_date.isoformat(),
                    strategy="explicit_backfill",
                )
            else:
                # First run: default 30 days
                oldest_dt = now - timedelta(days=default_backfill_days)
                oldest_ts = str(oldest_dt.timestamp())
                logger.info(
                    "slack_ingestion_first_run",
                    channel_id=channel_id,
                    backfill_days=default_backfill_days,
                    strategy="default_backfill",
                )

            # Fetch messages from Slack
            raw_messages = slack_client.fetch_messages(
                channel_id=channel_id,
                oldest_ts=oldest_ts,
                latest_ts=None,  # up to now
            )

            total_fetched += len(raw_messages)

            if not raw_messages:
                channels_processed.append(channel_id)
                continue

            # Process messages with user info and permalinks
            processed_messages: list[SlackMessage] = []
            for raw_msg in raw_messages:
                # Get user info if available
                user_info = None
                user_id = raw_msg.get("user")
                if user_id and not raw_msg.get("bot_id"):
                    try:
                        user_info = slack_client.get_user_info(user_id)
                    except Exception:
                        # Continue without user info if fetch fails
                        pass

                # Get permalink
                permalink = None
                msg_ts = raw_msg.get("ts")
                if msg_ts:
                    try:
                        permalink = slack_client.get_permalink(channel_id, msg_ts)
                    except Exception:
                        # Continue without permalink if fetch fails
                        pass

                processed_msg = process_slack_message(
                    raw_msg, channel_id, user_info=user_info, permalink=permalink
                )
                processed_messages.append(processed_msg)

            # Save to repository
            saved_count = repository.save_messages(processed_messages)
            total_saved += saved_count

            # Update ingestion_state to latest message timestamp
            if processed_messages:
                latest_ts_str = max(msg.ts for msg in processed_messages)
                latest_ts_float = float(latest_ts_str)
                # Add small epsilon to avoid refetching the same message
                repository.update_last_processed_ts(
                    channel_id, latest_ts_float + 0.000001
                )
                logger.info(
                    "slack_ingestion_state_updated",
                    channel_id=channel_id,
                    latest_ts=latest_ts_str,
                    messages_saved=saved_count,
                )
            elif last_ts is None:
                # First run but no messages: still mark as processed up to now
                repository.update_last_processed_ts(channel_id, now_ts)
                logger.info(
                    "slack_ingestion_state_initialized",
                    channel_id=channel_id,
                    reason="no_messages",
                )

            channels_processed.append(channel_id)

        except Exception as e:
            error_msg = f"Channel {channel_id}: {str(e)}"
            errors.append(error_msg)

    return IngestResult(
        messages_fetched=total_fetched,
        messages_saved=total_saved,
        channels_processed=channels_processed,
        errors=errors,
    )
