"""
Message client factory for multi-source support.

Provides a factory function to instantiate the correct message client
based on the source type (Slack, Telegram, etc.).
"""

from src.adapters.slack_client import SlackClient
from src.adapters.telegram_client import TelegramClient
from src.config.settings import get_settings
from src.domain.models import MessageSource
from src.domain.protocols import MessageClientProtocol


def get_message_client(
    source_id: MessageSource, bot_token: str
) -> MessageClientProtocol:
    """Get appropriate message client for the given source.

    Factory function that instantiates the correct client implementation
    based on the message source type.

    For Telegram, bot_token parameter is ignored (user client requires API_ID/API_HASH).
    Telegram credentials are loaded from settings.

    Args:
        source_id: Message source (SLACK or TELEGRAM)
        bot_token: Bot token for authentication (used for Slack only)

    Returns:
        MessageClientProtocol: Slack or Telegram client instance

    Raises:
        ValueError: If source_id is not supported or Telegram credentials missing

    Example:
        >>> from src.domain.models import MessageSource
        >>> slack_client = get_message_client(
        ...     source_id=MessageSource.SLACK,
        ...     bot_token="xoxb-slack-token"
        ... )
        >>> telegram_client = get_message_client(
        ...     source_id=MessageSource.TELEGRAM,
        ...     bot_token=""  # Not used for Telegram
        ... )
    """
    if source_id == MessageSource.SLACK:
        return SlackClient(bot_token=bot_token)
    elif source_id == MessageSource.TELEGRAM:
        # For Telegram, load credentials from settings
        settings = get_settings()

        if not settings.telegram_api_id or not settings.telegram_api_hash:
            raise ValueError(
                "Telegram API_ID and API_HASH must be configured in .env. "
                "See scripts/telegram_auth.py for setup instructions."
            )

        return TelegramClient(
            api_id=settings.telegram_api_id,
            api_hash=settings.telegram_api_hash.get_secret_value(),
            session_name=settings.telegram_session_path,
        )
    else:
        raise ValueError(
            f"Unsupported message source: {source_id}. "
            f"Supported sources: {[s.value for s in MessageSource]}"
        )
