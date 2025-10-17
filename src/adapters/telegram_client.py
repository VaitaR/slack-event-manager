"""
Telegram client adapter (stub implementation).

This is a stub implementation that returns empty results.
Full Telegram integration will be implemented in the future.
"""

from typing import Any


class TelegramClient:
    """Telegram client stub for multi-source architecture.

    This is a placeholder implementation that allows the multi-source
    pipeline to be developed and tested without requiring actual Telegram
    API integration.

    Future implementation will use python-telegram-bot or similar library
    to fetch real messages from Telegram channels.

    Args:
        bot_token: Telegram bot token (currently unused in stub)

    Example:
        >>> client = TelegramClient(bot_token="fake_token")
        >>> messages = client.fetch_messages(channel_id="test_channel")
        >>> assert messages == []  # Stub returns empty list
    """

    def __init__(self, bot_token: str) -> None:
        """Initialize Telegram client with bot token.

        Args:
            bot_token: Telegram bot token (stored but not used in stub)
        """
        self.bot_token = bot_token

    def fetch_messages(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Telegram channel (stub returns empty list).

        This is a stub implementation that always returns an empty list.
        Future implementation will fetch real messages from Telegram API.

        Args:
            channel_id: Telegram channel username or ID
            oldest_ts: Oldest message timestamp (optional)
            latest_ts: Latest message timestamp (optional)
            limit: Maximum messages to fetch

        Returns:
            Empty list (stub behavior)

        Example:
            >>> client = TelegramClient(bot_token="fake_token")
            >>> messages = client.fetch_messages(
            ...     channel_id="mychannel",
            ...     limit=50
            ... )
            >>> assert messages == []
        """
        # Stub implementation - always returns empty list
        return []

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        """Get Telegram user information (stub returns empty dict).

        This is a stub implementation that always returns an empty dict.
        Future implementation will fetch real user info from Telegram API.

        Args:
            user_id: Telegram user ID

        Returns:
            Empty dict (stub behavior)

        Example:
            >>> client = TelegramClient(bot_token="fake_token")
            >>> user_info = client.get_user_info(user_id="123456")
            >>> assert user_info == {}
        """
        # Stub implementation - always returns empty dict
        return {}
