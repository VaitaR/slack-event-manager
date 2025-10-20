"""Telegram client adapter using Telethon library.

Implements MessageClientProtocol for Telegram API interactions with user client.
Uses async Telethon library wrapped in synchronous interface for compatibility.
"""

import asyncio
import logging
import types
from typing import Any

import pytz

try:
    from telethon import (  # type: ignore[import-untyped]
        TelegramClient as TelegramClientLib,
    )
    from telethon.errors import FloodWaitError  # type: ignore[import-untyped]
    from telethon.tl.types import (  # type: ignore[import-untyped]
        MessageEntityTextUrl,
        MessageEntityUrl,
    )
except ImportError:
    # Fallback for when telethon is not available (e.g., in CI)
    TelegramClientLib = None
    FloodWaitError = Exception
    MessageEntityTextUrl = None
    MessageEntityUrl = None

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

        Raises:
            ImportError: If telethon is not available
        """
        # Allow initialization even if telethon is not available (for testing)
        if TelegramClientLib is None:
            # For testing purposes, we'll allow initialization but methods will fail
            pass

        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self._client: TelegramClientLib | None = None
        self._is_connected = False

    def _get_client(self) -> TelegramClientLib:
        """Get or create Telethon client instance.

        Returns:
            Telethon client instance

        Raises:
            ImportError: If telethon is not available
        """
        if TelegramClientLib is None:
            raise ImportError("TelegramClient requires telethon to be installed")

        if self._client is None:
            self._client = TelegramClientLib(
                self.session_name, self.api_id, self.api_hash
            )
        return self._client

    async def connect(self) -> None:
        """Connect to Telegram API.

        Raises:
            ImportError: If telethon is not available
            Exception: If connection fails
        """
        if TelegramClientLib is None:
            raise ImportError("TelegramClient requires telethon to be installed")

        if self._is_connected:
            return

        client = self._get_client()
        await client.start()
        self._is_connected = True
        logger.info("Telegram client connected successfully")

    async def disconnect(self) -> None:
        """Disconnect from Telegram API."""
        if not self._is_connected or self._client is None:
            return

        await self._client.disconnect()
        self._is_connected = False
        logger.info("Telegram client disconnected")

    def is_connected(self) -> bool:
        """Check if client is connected.

        Returns:
            True if connected, False otherwise
        """
        return self._is_connected

    async def close(self) -> None:
        """Close Telegram client connection and cleanup resources.

        This method should be called when the client is no longer needed
        to ensure proper cleanup of connections and resources.
        """
        await self.disconnect()
        self._client = None
        logger.info("Telegram client closed and resources cleaned up")

    def __enter__(self) -> "TelegramClient":
        """Context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit with cleanup."""
        await self.close()

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

        # Ensure we're connected
        await self.connect()

        try:
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
                    wait_seconds = e.seconds

                    logger.warning(
                        "Telegram FloodWait triggered",
                        extra={
                            "wait_seconds": wait_seconds,
                            "retry_count": retry_count,
                            "max_retries": max_retries,
                            "channel_id": channel_id,
                            "attempt": f"{retry_count}/{max_retries}",
                        },
                    )

                    if retry_count >= max_retries:
                        logger.error(
                            "Telegram FloodWait retries exhausted",
                            extra={
                                "wait_seconds": wait_seconds,
                                "final_retry_count": retry_count,
                                "channel_id": channel_id,
                            },
                        )
                        raise RateLimitError(retry_after=wait_seconds)

                    # Wait as requested by Telegram
                    logger.info(
                        f"Waiting {wait_seconds}s due to Telegram FloodWait "
                        f"(retry {retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(wait_seconds)

        except Exception as e:
            # Handle other exceptions that might occur during message fetching
            logger.error(
                "Unexpected error during Telegram message fetching",
                extra={
                    "error": str(e),
                    "channel_id": channel_id,
                    "retry_count": retry_count,
                },
            )
            # Re-raise the exception after logging
            raise

        return messages

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
                # Check if entity is one of the expected types (handle None fallbacks)
                if MessageEntityUrl is not None and MessageEntityTextUrl is not None:
                    # Both types available (Telethon installed)
                    if isinstance(entity, MessageEntityUrl | MessageEntityTextUrl):
                        entities_list.append(entity)
                elif MessageEntityUrl is not None:
                    # Only MessageEntityUrl available
                    if isinstance(entity, MessageEntityUrl):
                        entities_list.append(entity)
                elif MessageEntityTextUrl is not None:
                    # Only MessageEntityTextUrl available
                    if isinstance(entity, MessageEntityTextUrl):
                        entities_list.append(entity)
                # If both are None, skip all entity checks (fallback scenario)

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

    async def fetch_messages_async(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Telegram channel (async version).

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
            >>> messages = await client.fetch_messages_async("@channel", limit=10)
            >>> len(messages)
            10
        """
        return await self._fetch_messages_async(channel_id, oldest_ts, latest_ts, limit)

    def fetch_messages(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Telegram channel (synchronous wrapper).

        IMPORTANT: This method creates an isolated event loop for each call.
        For production use in async contexts (FastAPI, aiojobs), use fetch_messages_async()
        or create a dedicated async client instance.

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

        # Use isolated event loop approach to avoid conflicts with existing event loops
        # This is safe for sync contexts but should be avoided in async server contexts
        def _run_in_isolated_loop() -> list[dict[str, Any]]:
            """Run async function in completely isolated event loop."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                # Create a fresh client instance for this isolated loop
                # This avoids sharing client state between different event loops
                fresh_client = TelegramClient(
                    self.api_id, self.api_hash, self.session_name
                )
                return new_loop.run_until_complete(
                    fresh_client._fetch_messages_async(
                        channel_id, oldest_ts, latest_ts, limit
                    )
                )
            finally:
                # Clean up the client and close the loop
                try:
                    new_loop.run_until_complete(fresh_client.close())
                except Exception:
                    pass  # Ignore cleanup errors
                new_loop.close()

        # Run in thread pool to avoid blocking the main thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_isolated_loop)
            return future.result()

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
