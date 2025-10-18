"""Telegram client adapter using Telethon library.

Implements MessageClientProtocol for Telegram API interactions with user client.
Uses async Telethon library wrapped in synchronous interface for compatibility.
"""

import asyncio
import logging
from typing import Any

import pytz
from telethon import TelegramClient as TelegramClientLib
from telethon.errors import FloodWaitError
from telethon.tl.types import MessageEntityTextUrl, MessageEntityUrl

from src.domain.exceptions import RateLimitError

logger = logging.getLogger(__name__)


class TelegramClient:
    """Telegram client adapter using Telethon (user client).

    Wraps async Telethon library in synchronous interface for compatibility
    with existing MessageClientProtocol.

    Args:
        api_id: Telegram API ID (from my.telegram.org)
        api_hash: Telegram API hash (from my.telegram.org)
        session_name: Path to session file (e.g., 'data/telegram_session')

    Example:
        >>> client = TelegramClient(
        ...     api_id=12345,
        ...     api_hash="abc123...",
        ...     session_name="data/telegram_session"
        ... )
        >>> messages = client.fetch_messages(channel_id="@channel", limit=10)
    """

    def __init__(self, api_id: int, api_hash: str, session_name: str) -> None:
        """Initialize Telegram client with credentials.

        Args:
            api_id: Telegram API ID
            api_hash: Telegram API hash
            session_name: Path to session file
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self._client: TelegramClientLib | None = None

    def _get_client(self) -> TelegramClientLib:
        """Get or create Telethon client instance.

        Returns:
            Telethon client instance
        """
        if self._client is None:
            self._client = TelegramClientLib(
                self.session_name, self.api_id, self.api_hash
            )
        return self._client

    async def _fetch_messages_async(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int = 100,
        max_retries: int = 3,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Telegram channel (async implementation).

        Args:
            channel_id: Channel username (@channel) or numeric ID
            oldest_ts: Oldest message ID (not used for Telegram, kept for compatibility)
            latest_ts: Latest message ID (not used for Telegram, kept for compatibility)
            limit: Maximum messages to fetch
            max_retries: Maximum retry attempts for FloodWait

        Returns:
            List of message dictionaries

        Raises:
            RateLimitError: On FloodWait after max retries
        """
        client = self._get_client()
        retry_count = 0

        try:
            await client.start()

            # Get channel entity to extract username
            channel_entity = await client.get_entity(channel_id)
            channel_username = None
            if hasattr(channel_entity, "username") and channel_entity.username:
                channel_username = channel_entity.username

            messages: list[dict[str, Any]] = []

            while retry_count < max_retries:
                try:
                    # Fetch messages using iter_messages
                    # Note: In tests, iter_messages might return a list instead of async iterator
                    messages_result = client.iter_messages(channel_id, limit=limit)

                    if hasattr(messages_result, "__aiter__"):
                        # Real async iterator (production)
                        async for message in messages_result:
                            # Convert Telethon Message to dict
                            msg_dict = self._convert_message_to_dict(
                                message, channel_id, channel_username
                            )
                            messages.append(msg_dict)

                            # Respect limit
                            if len(messages) >= limit:
                                break
                    else:
                        # List of messages (test mock)
                        for message in messages_result:
                            # Convert Telethon Message to dict
                            msg_dict = self._convert_message_to_dict(
                                message, channel_id, channel_username
                            )
                            messages.append(msg_dict)

                            # Respect limit
                            if len(messages) >= limit:
                                break

                    # Success - break retry loop
                    break

                except FloodWaitError as e:
                    retry_count += 1
                    logger.warning(
                        f"FloodWait error: must wait {e.seconds}s "
                        f"(attempt {retry_count}/{max_retries})"
                    )

                    if retry_count >= max_retries:
                        raise RateLimitError(retry_after=e.seconds)

                    # Wait as requested by Telegram
                    await asyncio.sleep(e.seconds)

            return messages

        finally:
            await client.disconnect()

    def _convert_message_to_dict(
        self, message: Any, channel_id: str, channel_username: str | None
    ) -> dict[str, Any]:
        """Convert Telethon Message object to dictionary.

        Args:
            message: Telethon Message object
            channel_id: Channel ID or username
            channel_username: Channel username (if public)

        Returns:
            Message dictionary with standardized fields
        """
        # Extract text
        text = message.message or message.text or ""

        # Extract URLs from entities
        entities_list = []
        if message.entities:
            for entity in message.entities:
                if isinstance(entity, MessageEntityUrl | MessageEntityTextUrl):
                    entities_list.append(entity)

        # Build post URL for public channels
        post_url = None
        if channel_username:
            post_url = f"https://t.me/{channel_username}/{message.id}"

        # Convert date to UTC datetime
        message_date = message.date
        if message_date and message_date.tzinfo is None:
            message_date = message_date.replace(tzinfo=pytz.UTC)

        # Extract forward info
        forward_from_channel = None
        forward_from_message_id = None
        if message.forwards:
            if hasattr(message.forwards, "from_id"):
                forward_from_channel = str(message.forwards.from_id)
            if hasattr(message.forwards, "channel_post"):
                forward_from_message_id = str(message.forwards.channel_post)

        # Extract media type
        media_type = None
        if message.media:
            media_type = type(message.media).__name__

        # Extract views
        views = message.views or 0

        # Extract reply count
        reply_count = 0
        if message.replies:
            reply_count = message.replies.replies or 0

        return {
            "message_id": str(message.id),  # Convert to string for consistency
            "channel": channel_id,
            "date": message_date,
            "sender_id": str(message.sender_id) if message.sender_id else None,
            "text": text,
            "entities": entities_list,
            "media_type": media_type,
            "views": views,
            "reply_count": reply_count,
            "forward_from_channel": forward_from_channel,
            "forward_from_message_id": forward_from_message_id,
            "post_url": post_url,
        }

    def fetch_messages(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Telegram channel (synchronous wrapper).

        Args:
            channel_id: Channel username (@channel) or numeric ID
            oldest_ts: Oldest message ID (not used, kept for protocol compatibility)
            latest_ts: Latest message ID (not used, kept for protocol compatibility)
            limit: Maximum messages to fetch

        Returns:
            List of message dictionaries

        Raises:
            RateLimitError: On FloodWait after max retries

        Example:
            >>> client = TelegramClient(12345, "hash", "session")
            >>> messages = client.fetch_messages("@channel", limit=10)
            >>> len(messages)
            10
        """
        # Run async function in event loop
        return asyncio.run(
            self._fetch_messages_async(channel_id, oldest_ts, latest_ts, limit)
        )

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        """Get Telegram user information (stub for protocol compatibility).

        Args:
            user_id: Telegram user ID

        Returns:
            User info dictionary (stub implementation)

        Note:
            This is a stub implementation for protocol compatibility.
            Full user info fetching can be implemented later if needed.
        """
        return {
            "id": user_id,
            "username": "unknown",
            "first_name": "Unknown",
        }
