"""Telegram client adapter using Telethon library."""

import asyncio
import threading
import types
from collections.abc import Coroutine
from typing import Any, Final, TypeVar

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

from src.config.logging_config import get_logger
from src.domain.exceptions import RateLimitError

T = TypeVar("T")

logger = get_logger(__name__)

DEFAULT_TELEGRAM_PAGE_SIZE: Final[int] = 200
DEFAULT_TELEGRAM_PAGE_DELAY_SECONDS: Final[float] = 1.0
DEFAULT_TELEGRAM_MAX_RETRIES: Final[int] = 3


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

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str,
        *,
        page_size: int | None = None,
        max_total_messages: int | None = None,
        page_delay_seconds: float = DEFAULT_TELEGRAM_PAGE_DELAY_SECONDS,
        max_retries: int = DEFAULT_TELEGRAM_MAX_RETRIES,
    ) -> None:
        """Initialize Telegram client with credentials.

        Args:
            api_id: Telegram API ID
            api_hash: Telegram API hash
            session_name: Path to session file
            page_size: Optional pagination size override
            max_total_messages: Optional maximum messages per fetch
            page_delay_seconds: Delay between pagination batches
            max_retries: Maximum FloodWait retry attempts

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
        self._page_size = page_size or DEFAULT_TELEGRAM_PAGE_SIZE
        if self._page_size <= 0:
            raise ValueError("Telegram page_size must be positive")
        self._max_total_messages = max_total_messages
        self._page_delay_seconds = max(page_delay_seconds, 0.0)
        self._max_retries = max(max_retries, 1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._loop_lock = threading.Lock()

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

    def _loop_runner(self) -> None:
        """Background thread that owns the asyncio event loop."""

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._loop_lock:
            self._loop = loop
            self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            with self._loop_lock:
                self._loop = None
                self._loop_thread = None
                self._loop_ready.clear()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Ensure a background event loop is running."""

        with self._loop_lock:
            if self._loop and self._loop.is_running():
                return self._loop

            self._loop_ready.clear()
            self._loop_thread = threading.Thread(
                target=self._loop_runner,
                name="TelegramClientLoop",
                daemon=True,
            )
            self._loop_thread.start()

        # Wait with timeout to prevent hanging in tests
        if not self._loop_ready.wait(timeout=10.0):
            raise TimeoutError("Telegram event loop failed to start within 10 seconds")
        assert self._loop is not None
        return self._loop

    def _run_in_loop(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine on background loop and wait for result."""

        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        # Add timeout to prevent hanging (default 60s, can be overridden in tests)
        timeout = getattr(self, "_operation_timeout", 60.0)
        return future.result(timeout=timeout)

    async def _fetch_messages_async(
        self,
        channel_id: str,
        *,
        min_message_id: int | None = None,
        limit: int | None = None,
        page_size: int | None = None,
        max_retries: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Telegram channel using incremental pagination."""

        client = self._get_client()
        await self.connect()

        effective_page_size = page_size or self._page_size
        if effective_page_size <= 0:
            raise ValueError("Telegram page_size must be positive")

        effective_limit = limit if limit is not None else self._max_total_messages
        retry_limit = max_retries or self._max_retries

        try:
            channel_entity = await client.get_entity(channel_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "telegram_get_entity_failed",
                channel_id=channel_id,
                error=str(exc),
            )
            raise

        channel_username = (
            channel_entity.username
            if getattr(channel_entity, "username", None)
            else None
        )

        messages: list[dict[str, Any]] = []
        current_min_id = min_message_id
        retry_count = 0

        while True:
            if effective_limit is not None and len(messages) >= effective_limit:
                break

            remaining = (
                effective_limit - len(messages) if effective_limit is not None else None
            )
            batch_limit = effective_page_size
            if remaining is not None:
                if remaining <= 0:
                    break
                batch_limit = min(batch_limit, remaining)

            iter_kwargs: dict[str, Any] = {
                "limit": batch_limit,
                "reverse": True,
            }
            if current_min_id is not None:
                iter_kwargs["min_id"] = current_min_id

            try:
                messages_iter = client.iter_messages(channel_id, **iter_kwargs)
                batch = await self._collect_messages_batch(
                    messages_iter,
                    channel_id,
                    channel_username,
                    batch_limit,
                )
            except FloodWaitError as error:
                retry_count += 1
                wait_seconds = int(getattr(error, "seconds", 10))
                logger.warning(
                    "telegram_flood_wait",
                    channel_id=channel_id,
                    wait_seconds=wait_seconds,
                    retry_count=retry_count,
                    max_retries=retry_limit,
                )
                if retry_count >= retry_limit:
                    logger.error(
                        "telegram_flood_wait_exhausted",
                        channel_id=channel_id,
                        wait_seconds=wait_seconds,
                    )
                    raise RateLimitError(retry_after=wait_seconds)
                await asyncio.sleep(wait_seconds)
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "telegram_fetch_messages_error",
                    channel_id=channel_id,
                    error=str(exc),
                )
                raise

            retry_count = 0

            if not batch:
                break

            messages.extend(batch)

            last_raw_id = int(batch[-1]["message_id"])
            current_min_id = last_raw_id

            if effective_limit is not None and len(messages) >= effective_limit:
                break

            if self._page_delay_seconds > 0:
                await asyncio.sleep(self._page_delay_seconds)

        return messages

    async def _collect_messages_batch(
        self,
        messages_iter: Any,
        channel_id: str,
        channel_username: str | None,
        batch_limit: int,
    ) -> list[dict[str, Any]]:
        """Collect a single batch of Telegram messages from iterator."""

        batch: list[dict[str, Any]] = []

        if hasattr(messages_iter, "__aiter__"):
            async for message in messages_iter:
                batch.append(
                    self._convert_message_to_dict(message, channel_id, channel_username)
                )
                if len(batch) >= batch_limit:
                    break
        else:
            for message in messages_iter:
                batch.append(
                    self._convert_message_to_dict(message, channel_id, channel_username)
                )
                if len(batch) >= batch_limit:
                    break

        return batch

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
        limit: int | None = None,
        *,
        min_message_id: int | None = None,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Telegram channel (async version).

        Args:
            channel_id: Channel username (@channel) or numeric ID
            oldest_ts: Legacy timestamp input (used to derive min_message_id when provided)
            latest_ts: Unused placeholder for protocol compatibility
            limit: Maximum messages to fetch (None = unlimited)
            min_message_id: Minimum Telegram message ID (exclusive) to start from
            page_size: Optional override for pagination size

        Returns:
            List of message dictionaries
        """
        _ = latest_ts  # Explicitly unused

        derived_min_id = min_message_id
        if derived_min_id is None and oldest_ts:
            try:
                derived_min_id = int(oldest_ts)
            except ValueError:
                try:
                    derived_min_id = int(float(oldest_ts))
                except ValueError:
                    derived_min_id = None

        return await self._fetch_messages_async(
            channel_id,
            min_message_id=derived_min_id,
            limit=limit,
            page_size=page_size,
        )

    def fetch_messages(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int | None = None,
        *,
        min_message_id: int | None = None,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Telegram channel (synchronous wrapper)."""

        return self._run_in_loop(
            self.fetch_messages_async(
                channel_id,
                oldest_ts=oldest_ts,
                latest_ts=latest_ts,
                limit=limit,
                min_message_id=min_message_id,
                page_size=page_size,
            )
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
