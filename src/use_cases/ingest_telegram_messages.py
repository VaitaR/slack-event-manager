"""Ingest Telegram messages use case.

Fetches messages from Telegram channels and stores them with normalization.
Similar to Slack ingestion but adapted for Telegram's message structure.
"""

import hashlib
from datetime import datetime
from typing import Any

import pytz

from src.adapters.telegram_client import TelegramClient
from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.domain.models import IngestResult, MessageSource, TelegramMessage
from src.domain.protocols import RepositoryProtocol
from src.services import link_extractor, text_normalizer

logger = get_logger(__name__)

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
        # Check if entity is MessageEntityUrl (handle None fallbacks)
        if MessageEntityUrl is not None and isinstance(entity, MessageEntityUrl):
            # Extract URL from text using offset and length
            url = text[entity.offset : entity.offset + entity.length]
            urls.append(url)
        # Check if entity is MessageEntityTextUrl (handle None fallbacks)
        elif MessageEntityTextUrl is not None and isinstance(
            entity, MessageEntityTextUrl
        ):
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


def _extract_reply_reference(raw_msg: dict[str, Any]) -> str | None:
    """Extract raw reply target identifier from Telegram payload."""

    reply_to = raw_msg.get("reply_to_message_id") or raw_msg.get("reply_to_msg_id")
    if reply_to:
        return str(reply_to)

    reply_block = raw_msg.get("reply_to")
    if isinstance(reply_block, dict):
        for key in ("reply_to_msg_id", "msg_id", "id"):
            value = reply_block.get(key)
            if value:
                return str(value)

    return None


def _normalize_reactions(raw_reactions: Any) -> dict[str, int]:
    """Convert heterogeneous Telegram reactions payload into a simple mapping."""

    reactions: dict[str, int] = {}

    def _add_reaction(label: str, count: int) -> None:
        if not label:
            return
        reactions[label] = reactions.get(label, 0) + max(count, 0)

    if raw_reactions is None:
        return reactions

    if isinstance(raw_reactions, dict):
        results = raw_reactions.get("results")
        if isinstance(results, list):
            for entry in results:
                if isinstance(entry, dict):
                    emoji = entry.get("reaction") or entry.get("emoji")
                    if isinstance(emoji, dict):
                        emoji = emoji.get("emoticon") or emoji.get("emoji")
                    count = entry.get("count") or entry.get("total_count") or 0
                    if isinstance(count, int):
                        _add_reaction(str(emoji or ""), count)
            return reactions

        for key, value in raw_reactions.items():
            if isinstance(value, int):
                _add_reaction(str(key), value)
        return reactions

    if isinstance(raw_reactions, list):
        for entry in raw_reactions:
            if isinstance(entry, dict):
                emoji = entry.get("reaction") or entry.get("emoji")
                if isinstance(emoji, dict):
                    emoji = emoji.get("emoticon") or emoji.get("emoji")
                count = entry.get("count") or entry.get("total_count") or 0
                if isinstance(count, int):
                    _add_reaction(str(emoji or ""), count)
        return reactions

    if isinstance(raw_reactions, str):
        _add_reaction(raw_reactions, 1)
    return reactions


def _extract_file_mime(raw_msg: dict[str, Any]) -> str | None:
    """Extract MIME type from Telegram payload if available."""

    for key in ("file_mime", "mime_type", "mimeType"):
        value = raw_msg.get(key)
        if isinstance(value, str) and value:
            return value

    media = raw_msg.get("media") or raw_msg.get("document") or raw_msg.get("file")
    if isinstance(media, dict):
        for key in ("mime_type", "mimeType"):
            value = media.get(key)
            if isinstance(value, str) and value:
                return value

    return None


def _detect_has_file(
    media_type: str | None,
    file_mime: str | None,
    raw_msg: dict[str, Any],
) -> bool:
    """Determine whether message contains a file attachment."""

    if file_mime:
        return True

    if media_type and media_type.lower() not in {"", "text", "message"}:
        return True

    files = raw_msg.get("files") or raw_msg.get("media")
    if isinstance(files, list) and files:
        return True

    return bool(raw_msg.get("has_media"))


def _extract_bot_id(raw_msg: dict[str, Any]) -> str | None:
    """Extract bot identifier as string if present."""

    for key in ("via_bot_id", "bot_id", "sender_bot_id"):
        value = raw_msg.get(key)
        if value is None:
            continue
        return str(value)

    bot_block = raw_msg.get("from_bot")
    if isinstance(bot_block, dict):
        bot_id = bot_block.get("id") or bot_block.get("bot_id")
        if bot_id is not None:
            return str(bot_id)

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
    sender_name = raw_msg.get("sender_name")
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

    reactions_map = _normalize_reactions(raw_msg.get("reactions"))
    reactions_count = sum(reactions_map.values())

    reply_reference = _extract_reply_reference(raw_msg)
    reply_to_id = (
        generate_telegram_message_id(channel_id, reply_reference)
        if reply_reference
        else None
    )
    thread_id = raw_msg.get("thread_id") or reply_to_id
    is_reply = reply_to_id is not None

    file_mime = _extract_file_mime(raw_msg)
    has_file = _detect_has_file(media_type, file_mime, raw_msg)

    bot_id = _extract_bot_id(raw_msg)
    is_bot = bool(raw_msg.get("is_bot") or bot_id)

    attachments_count = raw_msg.get("attachments_count", 0)
    attachments_payload = raw_msg.get("attachments")
    if isinstance(attachments_payload, list):
        attachments_count = len(attachments_payload)

    files_count = raw_msg.get("files_count", 0)
    files_payload = raw_msg.get("files")
    if isinstance(files_payload, list):
        files_count = len(files_payload)

    # Generate deterministic message ID
    unique_id = generate_telegram_message_id(channel_id, message_id)

    return TelegramMessage(
        message_id=unique_id,  # Use hash as primary key
        channel=channel_id,
        message_date=message_date,
        sender_id=sender_id,
        sender_name=sender_name,
        bot_id=bot_id,
        is_bot=is_bot,
        reply_to_id=reply_to_id,
        thread_id=thread_id,
        is_reply=is_reply,
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
        reactions=reactions_map,
        reactions_count=reactions_count,
        post_url=post_url,
        attachments_count=attachments_count,
        files_count=files_count,
        has_file=has_file,
        file_mime=file_mime,
        ingested_at=datetime.utcnow().replace(tzinfo=pytz.UTC),
    )


async def ingest_telegram_messages_use_case_async(
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

    datetime.utcnow().replace(tzinfo=pytz.UTC)

    # Default backfill window: 1 day (as per requirements)

    # Get Telegram channels from config
    telegram_channels = getattr(settings, "telegram_channels", [])

    if not telegram_channels:
        logger.warning(
            "telegram_ingestion_no_channels",
            reason="no_channels_configured",
        )
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
            channel_config.get("from_date")
        else:
            # TelegramChannelConfig object
            channel_id = getattr(channel_config, "username", "")
            enabled = getattr(channel_config, "enabled", True)
            getattr(channel_config, "from_date", None)

        if not enabled:
            logger.info(
                "Telegram channel disabled, skipping",
                extra={"channel_id": channel_id, "enabled": enabled},
            )
            continue

        try:
            # Get ingestion state for Telegram
            last_message_id = repository.get_last_processed_message_id(
                channel_id, source_id=MessageSource.TELEGRAM
            )

            # Determine backfill strategy
            if last_message_id is not None:
                # Incremental: fetch from last processed message
                logger.info(
                    "Telegram channel incremental ingestion",
                    extra={
                        "channel_id": channel_id,
                        "last_message_id": last_message_id,
                        "strategy": "incremental",
                    },
                )
                # For Telegram, we'll fetch all and filter by message_id
                # Telethon doesn't support min_id in iter_messages easily
                backfill_date = None
            else:
                # First run: no date filter, accept all messages
                backfill_date = None
                logger.info(
                    "Telegram channel first run (no date filter)",
                    extra={
                        "channel_id": channel_id,
                        "strategy": "first_run_no_filter",
                    },
                )

            min_message_id_int: int | None = None
            if last_message_id is not None:
                try:
                    min_message_id_int = int(last_message_id)
                except ValueError:
                    min_message_id_int = None

            raw_messages = await telegram_client.fetch_messages_async(
                channel_id=channel_id,
                limit=settings.telegram_max_messages_per_run,
                min_message_id=min_message_id_int,
                page_size=settings.telegram_page_size,
            )

            logger.info(
                "Telegram messages fetched",
                extra={
                    "channel_id": channel_id,
                    "raw_count": len(raw_messages),
                    "min_message_id": min_message_id_int,
                },
            )
            total_fetched += len(raw_messages)

            if not raw_messages:
                channels_processed.append(channel_id)
                # Initialize state even if no messages
                if last_message_id is None:
                    repository.update_last_processed_message_id(
                        channel_id, "0", source_id=MessageSource.TELEGRAM
                    )
                    logger.info(
                        "Telegram channel state initialized",
                        extra={"channel_id": channel_id, "reason": "no_messages"},
                    )
                continue

            # Filter messages by date if backfill_date is set
            # Telegram messages are returned newest first, so we need to process them in reverse order
            # to handle backfill_date correctly
            filtered_messages = []
            if raw_messages:
                first_msg_date = raw_messages[0].get("date")
                last_msg_date = raw_messages[-1].get("date")
                logger.info(
                    "Telegram message date range",
                    extra={
                        "channel_id": channel_id,
                        "backfill_date": backfill_date.isoformat()
                        if backfill_date
                        else None,
                        "first_msg_date": first_msg_date.isoformat()
                        if isinstance(first_msg_date, datetime)
                        else str(first_msg_date),
                        "last_msg_date": last_msg_date.isoformat()
                        if isinstance(last_msg_date, datetime)
                        else str(last_msg_date),
                        "total_raw": len(raw_messages),
                    },
                )

            # Process messages in reverse order (oldest first) when backfill_date is set
            # This allows us to find the cutoff point correctly
            if backfill_date:
                messages_to_check = reversed(raw_messages)

                for raw_msg in messages_to_check:
                    msg_date = raw_msg.get("date")
                    if isinstance(msg_date, datetime):
                        if msg_date < backfill_date:
                            # Stop processing older messages
                            logger.info(
                                "Stopping date filter (message too old)",
                                extra={
                                    "channel_id": channel_id,
                                    "msg_date": msg_date.isoformat(),
                                    "backfill_date": backfill_date.isoformat(),
                                },
                            )
                            break
                        filtered_messages.append(raw_msg)

                # Restore original order
                filtered_messages.reverse()
            else:
                # No date filter - accept all messages
                filtered_messages = raw_messages

            logger.info(
                "Telegram messages after date filter",
                extra={
                    "channel_id": channel_id,
                    "filtered_count": len(filtered_messages),
                },
            )

            # Filter by last_message_id if incremental (defensive double-check)
            if min_message_id_int is not None:
                before_filter = len(filtered_messages)
                filtered_messages = [
                    msg
                    for msg in filtered_messages
                    if int(msg.get("message_id", "0")) > min_message_id_int
                ]
                logger.info(
                    "Telegram messages after ID filter",
                    extra={
                        "channel_id": channel_id,
                        "before": before_filter,
                        "after": len(filtered_messages),
                    },
                )

            if not filtered_messages:
                logger.info(
                    "No new messages for Telegram channel",
                    extra={"channel_id": channel_id, "reason": "no_new_messages"},
                )
                channels_processed.append(channel_id)
                continue

            # Process messages
            processed_messages: list[TelegramMessage] = []
            for raw_msg in filtered_messages:
                processed_msg = process_telegram_message(raw_msg, channel_id)
                processed_messages.append(processed_msg)

            logger.info(
                "Telegram messages processed, attempting save",
                extra={
                    "channel_id": channel_id,
                    "processed_count": len(processed_messages),
                },
            )

            # Save to repository
            saved_count = repository.save_telegram_messages(processed_messages)
            logger.info(
                "Telegram messages saved",
                extra={
                    "channel_id": channel_id,
                    "saved_count": saved_count,
                },
            )
            total_saved += saved_count

            # Update ingestion_state to latest message ID
            if processed_messages:
                # Actually, use the real Telegram message ID from raw messages
                telegram_ids = [
                    int(raw_msg.get("message_id", "0")) for raw_msg in filtered_messages
                ]
                max_telegram_id = max(telegram_ids) if telegram_ids else 0

                repository.update_last_processed_message_id(
                    channel_id, str(max_telegram_id), source_id=MessageSource.TELEGRAM
                )
                logger.info(
                    "Telegram channel state updated",
                    extra={
                        "channel_id": channel_id,
                        "new_message_id": str(max_telegram_id),
                        "messages_processed": len(processed_messages),
                    },
                )

            channels_processed.append(channel_id)

        except Exception as e:
            error_msg = f"Channel {channel_id}: {str(e)}"
            errors.append(error_msg)
            logger.error(
                "Telegram ingestion error",
                extra={"channel_id": channel_id, "error": str(e)},
            )

    return IngestResult(
        messages_fetched=total_fetched,
        messages_saved=total_saved,
        channels_processed=channels_processed,
        errors=errors,
    )


def ingest_telegram_messages_use_case(
    telegram_client: TelegramClient,
    repository: RepositoryProtocol,
    settings: Settings,
    backfill_from_date: datetime | None = None,
) -> IngestResult:
    """Synchronous wrapper for Telegram ingestion use case.

    This is a compatibility wrapper that runs the async version using the sync Telegram client.
    For production use in async contexts, use ingest_telegram_messages_use_case_async() directly.

    Args:
        telegram_client: Telegram client instance
        repository: Data repository
        settings: Application settings
        backfill_from_date: Optional backfill start date

    Returns:
        IngestResult with ingestion statistics
    """
    # Use sync client wrapper for backward compatibility with existing tests
    # This avoids asyncio.run() issues in test environments
    import asyncio

    # Create a new event loop for this specific call
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Run the async version in this isolated loop
        result = loop.run_until_complete(
            ingest_telegram_messages_use_case_async(
                telegram_client=telegram_client,
                repository=repository,
                settings=settings,
                backfill_from_date=backfill_from_date,
            )
        )
        return result
    finally:
        loop.close()
