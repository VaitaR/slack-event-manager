"""
Message client factory for multi-source support.

Provides a factory function to instantiate the correct message client
based on the source type (Slack, Telegram, etc.).
"""

from src.adapters.slack_client import SlackClient
from src.adapters.telegram_client import TelegramClient
from src.domain.models import MessageSource
from src.domain.protocols import MessageClientProtocol


def get_message_client(
    source_id: MessageSource, bot_token: str
) -> MessageClientProtocol:
    """Get appropriate message client for the given source.

    Factory function that instantiates the correct client implementation
    based on the message source type.

    Args:
        source_id: Message source (SLACK or TELEGRAM)
        bot_token: Bot token for authentication

    Returns:
        MessageClientProtocol: Slack or Telegram client instance

    Raises:
        ValueError: If source_id is not supported

    Example:
        >>> from src.domain.models import MessageSource
        >>> slack_client = get_message_client(
        ...     source_id=MessageSource.SLACK,
        ...     bot_token="xoxb-slack-token"
        ... )
        >>> telegram_client = get_message_client(
        ...     source_id=MessageSource.TELEGRAM,
        ...     bot_token="telegram-bot-token"
        ... )
    """
    if source_id == MessageSource.SLACK:
        return SlackClient(bot_token=bot_token)
    elif source_id == MessageSource.TELEGRAM:
        return TelegramClient(bot_token=bot_token)
    else:
        raise ValueError(
            f"Unsupported message source: {source_id}. "
            f"Supported sources: {[s.value for s in MessageSource]}"
        )
