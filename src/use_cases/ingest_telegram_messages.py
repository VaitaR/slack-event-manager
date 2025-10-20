"""Ingest Telegram messages use case.

Fetches messages from Telegram channels and stores them with normalization.
Similar to Slack ingestion but adapted for Telegram's message structure.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Any

import pytz

from src.domain.protocols import RepositoryProtocol

# Import Telegram types conditionally to avoid runtime errors
try:
    from telethon.tl.types import (  # type: ignore[import-untyped]
        MessageEntityTextUrl,
        MessageEntityUrl,
    )
except ImportError:
    # Fallback for type checking or when telethon is not available
    MessageEntityTextUrl = Any
    MessageEntityUrl = Any
from src.adapters.telegram_client import TelegramClient
from src.config.settings import Settings
from src.domain.models import IngestResult, MessageSource, TelegramMessage
from src.services import link_extractor, text_normalizer


def generate_telegram_message_id(channel: str, message_id: str) -> str:
    """Generate deterministic message ID for Telegram.

    Args:
        channel: Channel ID or username
        message_id: Telegram message ID (as string)

    Returns:
        SHA1 hash of channel|message_id

    Example:
        >>> generate_telegram_message_id("@channel", "123")
        'a3f2b1c0...'
    """
    key_material = f"{channel}|{message_id}"
    return hashlib.sha1(key_material.encode("utf-8")).hexdigest()


def extract_urls_from_entities(entities: list[Any], text: str) -> list[str]:
    """Extract URLs from Telegram message entities.

    Args:
        entities: List of Telegram entities (MessageEntityUrl, MessageEntityTextUrl)
        text: Message text

    Returns:
        List of extracted URLs
    """
    urls: list[str] = []

    for entity in entities:
        if isinstance(entity, MessageEntityUrl):
            # Extract URL from text using offset and length
            url = text[entity.offset : entity.offset + entity.length]
            urls.append(url)
        elif isinstance(entity, MessageEntityTextUrl):
            # URL is stored in entity.url
            urls.append(entity.url)

    return urls


def build_telegram_post_url(channel_id: str, message_id: str) -> str | None:
    """Build post URL for Telegram message.

    Args:
        channel_id: Channel username (@channel) or numeric ID
        message_id: Message ID

    Returns:
        Post URL or None if channel has no username

    Example:
        >>> build_telegram_post_url("@crypto_news", "123")
        'https://t.me/crypto_news/123'
    """
    # Extract username from @username format
    if channel_id.startswith("@"):
        username = channel_id[1:]  # Remove @ prefix
        return f"https://t.me/{username}/{message_id}"

    # Numeric IDs don't have public URLs
    return None


def process_telegram_message(
    raw_msg: dict[str, Any], channel_id: str
) -> TelegramMessage:
    """Process raw Telegram message into domain model.

    Args:
        raw_msg: Raw Telegram message dictionary from TelegramClient
        channel_id: Channel ID or username

    Returns:
        Processed TelegramMessage
    """
    # Extract basic fields
    message_id = raw_msg.get("message_id", "")
    message_date = raw_msg.get("date")
    if isinstance(message_date, datetime) and message_date.tzinfo is None:
        message_date = message_date.replace(tzinfo=pytz.UTC)
    elif not isinstance(message_date, datetime):
        message_date = datetime.utcnow().replace(tzinfo=pytz.UTC)

    sender_id = raw_msg.get("sender_id")
    text = raw_msg.get("text", "")

    # Extract URLs from entities
    entities = raw_msg.get("entities", [])
    urls_from_entities = extract_urls_from_entities(entities, text)

    # Combine with any URLs in text (regex)
    urls_from_text = link_extractor.extract_urls(text)
    all_urls = list(set(urls_from_entities + urls_from_text))  # Deduplicate

    # Normalize URLs
    links_norm = link_extractor.normalize_links(all_urls)

    # Extract anchors
    anchors = link_extractor.extract_all_anchors(text)

    # Normalize text
    text_norm = text_normalizer.normalize_text(text)

    # Build post URL
    post_url = raw_msg.get("post_url") or build_telegram_post_url(
        channel_id, message_id
    )

    # Extract other fields
    forward_from_channel = raw_msg.get("forward_from_channel")
    forward_from_message_id = raw_msg.get("forward_from_message_id")
    media_type = raw_msg.get("media_type")
    views = raw_msg.get("views", 0)
    reply_count = raw_msg.get("reply_count", 0)

    # Generate deterministic message ID
    unique_id = generate_telegram_message_id(channel_id, message_id)

    return TelegramMessage(
        message_id=unique_id,  # Use hash as primary key
        channel=channel_id,
        message_date=message_date,
        sender_id=sender_id,
        sender_name=None,  # Not extracted yet
        is_bot=False,  # Not extracted yet
        text=text,
        text_norm=text_norm,
        blocks_text=text,  # Use text for scoring compatibility
        forward_from_channel=forward_from_channel,
        forward_from_message_id=forward_from_message_id,
        media_type=media_type,
        links_raw=all_urls,
        links_norm=links_norm,
        anchors=anchors,
        views=views,
        reply_count=reply_count,
        reactions={},  # Not extracted yet
        post_url=post_url,
        ingested_at=datetime.utcnow().replace(tzinfo=pytz.UTC),
    )


def ingest_telegram_messages_use_case(
    telegram_client: TelegramClient,
    repository: RepositoryProtocol,
    settings: Settings,
    backfill_from_date: datetime | None = None,
) -> IngestResult:
    """Ingest messages from configured Telegram channels.

    For each channel:
    1. Get ingestion state (last_message_id)
    2. Determine backfill date:
       - If state exists: fetch from last_message_id (incremental)
       - If no state and backfill_from_date: use that date
       - If no state and no backfill: use 1 day ago (default)
    3. Fetch messages from Telegram
    4. Normalize text
    5. Extract links & anchors
    6. Generate message_id
    7. Save to raw_telegram_messages
    8. Update ingestion_state_telegram with max message_id

    Args:
        telegram_client: Telegram client
        repository: Data repository
        settings: Application settings
        backfill_from_date: Optional date to start backfill from (first run only)

    Returns:
        IngestResult with counts and errors

    Example:
        >>> result = ingest_telegram_messages_use_case(client, repo, settings)
        >>> result.messages_saved
        42
    """
    total_fetched = 0
    total_saved = 0
    channels_processed: list[str] = []
    errors: list[str] = []

    now = datetime.utcnow().replace(tzinfo=pytz.UTC)

    # Default backfill window: 1 day (as per requirements)
    default_backfill_days = 1

    # Get Telegram channels from config
    telegram_channels = getattr(settings, "telegram_channels", [])

    if not telegram_channels:
        print("⚠️  No Telegram channels configured")
        return IngestResult(
            messages_fetched=0,
            messages_saved=0,
            channels_processed=[],
            errors=["No Telegram channels configured"],
        )

    for channel_config in telegram_channels:
        # Handle both dict and object formats
        if isinstance(channel_config, dict):
            channel_id = channel_config.get("channel_id", "")
            enabled = channel_config.get("enabled", True)
            from_date_str = channel_config.get("from_date")
        else:
            # TelegramChannelConfig object
            channel_id = getattr(channel_config, "username", "")
            enabled = getattr(channel_config, "enabled", True)
            from_date_str = getattr(channel_config, "from_date", None)

        if not enabled:
            print(f"⏭️  Channel {channel_id} is disabled, skipping")
            continue

        try:
            # Get ingestion state for Telegram
            last_message_id = repository.get_last_processed_ts(
                channel_id, source_id=MessageSource.TELEGRAM
            )

            # Determine backfill strategy
            if last_message_id is not None:
                # Incremental: fetch from last processed message
                print(
                    f"📈 Channel {channel_id}: Incremental from message_id {last_message_id}"
                )
                # For Telegram, we'll fetch all and filter by message_id
                # Telethon doesn't support min_id in iter_messages easily
                backfill_date = None
            elif backfill_from_date:
                # First run with explicit backfill date
                backfill_date = backfill_from_date
                print(
                    f"📅 Channel {channel_id}: Backfill from {backfill_date.isoformat()}"
                )
            elif from_date_str:
                # Use from_date from config
                backfill_date = datetime.fromisoformat(
                    from_date_str.replace("Z", "+00:00")
                )
                print(
                    f"📅 Channel {channel_id}: Backfill from config {backfill_date.isoformat()}"
                )
            else:
                # First run: default 1 day
                backfill_date = now - timedelta(days=default_backfill_days)
                print(
                    f"🔄 Channel {channel_id}: First run, backfill {default_backfill_days} day(s)"
                )

            # Fetch messages from Telegram
            # Note: Telegram returns newest first, we'll filter by date
            raw_messages = telegram_client.fetch_messages(
                channel_id=channel_id,
                limit=100,  # Fetch up to 100 messages
            )

            total_fetched += len(raw_messages)

            if not raw_messages:
                channels_processed.append(channel_id)
                # Initialize state even if no messages
                if last_message_id is None:
                    repository.update_last_processed_ts(
                        channel_id, 0, source_id=MessageSource.TELEGRAM
                    )
                    print(
                        f"✅ Initialized ingestion_state for {channel_id} (no messages)"
                    )
                continue

            # Filter messages by date if backfill_date is set
            filtered_messages = []
            for raw_msg in raw_messages:
                msg_date = raw_msg.get("date")
                if isinstance(msg_date, datetime):
                    if backfill_date and msg_date < backfill_date:
                        # Stop processing older messages
                        break
                    filtered_messages.append(raw_msg)

            # Filter by last_message_id if incremental
            if last_message_id is not None:
                filtered_messages = [
                    msg
                    for msg in filtered_messages
                    if int(msg.get("message_id", "0")) > int(last_message_id)
                ]

            if not filtered_messages:
                print(f"ℹ️  No new messages for {channel_id}")
                channels_processed.append(channel_id)
                continue

            # Process messages
            processed_messages: list[TelegramMessage] = []
            for raw_msg in filtered_messages:
                processed_msg = process_telegram_message(raw_msg, channel_id)
                processed_messages.append(processed_msg)

            # Save to repository
            saved_count = repository.save_telegram_messages(processed_messages)
            total_saved += saved_count

            # Update ingestion_state to latest message ID
            if processed_messages:
                # Actually, use the real Telegram message ID from raw messages
                telegram_ids = [
                    int(raw_msg.get("message_id", "0")) for raw_msg in filtered_messages
                ]
                max_telegram_id = max(telegram_ids) if telegram_ids else 0

                repository.update_last_processed_ts(
                    channel_id, float(max_telegram_id), source_id=MessageSource.TELEGRAM
                )
                print(
                    f"✅ Updated ingestion_state for {channel_id} to message_id {max_telegram_id}"
                )

            channels_processed.append(channel_id)

        except Exception as e:
            error_msg = f"Channel {channel_id}: {str(e)}"
            errors.append(error_msg)
            print(f"❌ {error_msg}")

    return IngestResult(
        messages_fetched=total_fetched,
        messages_saved=total_saved,
        channels_processed=channels_processed,
        errors=errors,
    )
