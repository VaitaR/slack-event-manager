"""
Tests for message client factory.

Following TDD: Write tests first, then implement in src/adapters/message_client_factory.py
"""

import os

import pytest

from src.adapters.telegram_client import TelegramClient
from src.domain.models import MessageSource


class TestMessageClientFactory:
    """Test message client factory functionality."""

    def test_factory_exists(self) -> None:
        """Test that get_message_client factory function exists."""
        from src.adapters.message_client_factory import get_message_client

        assert get_message_client is not None
        assert callable(get_message_client)

    def test_get_slack_client(self) -> None:
        """Test factory returns SlackClient for SLACK source."""
        from src.adapters.message_client_factory import get_message_client
        from src.adapters.slack_client import SlackClient

        client = get_message_client(
            source_id=MessageSource.SLACK, bot_token="fake_slack_token"
        )

        assert isinstance(client, SlackClient)

    @pytest.mark.skipif(
        not os.getenv("TELEGRAM_API_ID") or not os.getenv("TELEGRAM_API_HASH"),
        reason="Telegram credentials not configured",
    )
    def test_get_telegram_client(self) -> None:
        """Test factory returns TelegramClient for TELEGRAM source."""
        from src.adapters.message_client_factory import get_message_client
        from src.adapters.telegram_client import TelegramClient

        client = get_message_client(
            source_id=MessageSource.TELEGRAM, bot_token="fake_telegram_token"
        )

        assert isinstance(client, TelegramClient)

    def test_slack_client_has_bot_token(self) -> None:
        """Test that Slack client is initialized correctly."""
        from src.adapters.message_client_factory import get_message_client
        from src.adapters.slack_client import SlackClient

        client = get_message_client(
            source_id=MessageSource.SLACK, bot_token="test_slack_token_123"
        )

        assert isinstance(client, SlackClient)
        # SlackClient stores token in internal WebClient, verify it exists
        assert hasattr(client, "client")

    @pytest.mark.skipif(
        not os.getenv("TELEGRAM_API_ID") or not os.getenv("TELEGRAM_API_HASH"),
        reason="Telegram credentials not configured",
    )
    def test_telegram_client_has_bot_token(self) -> None:
        """Test that Telegram client receives correct bot token."""
        from src.adapters.message_client_factory import get_message_client
        from src.adapters.telegram_client import TelegramClient

        client = get_message_client(
            source_id=MessageSource.TELEGRAM, bot_token="test_telegram_token_456"
        )

        assert isinstance(client, TelegramClient)
        assert client.bot_token == "test_telegram_token_456"

    def test_clients_implement_protocol(self) -> None:
        """Test that all clients from factory implement MessageClientProtocol."""
        from src.adapters.message_client_factory import get_message_client

        for source_id in [MessageSource.SLACK, MessageSource.TELEGRAM]:
            client = get_message_client(source_id=source_id, bot_token="fake_token")

            # Verify protocol methods exist
            assert hasattr(client, "fetch_messages")
            assert hasattr(client, "get_user_info")
            assert callable(client.fetch_messages)
            assert callable(client.get_user_info)

    def test_invalid_source_raises_error(self) -> None:
        """Test that invalid source_id raises ValueError."""
        from src.adapters.message_client_factory import get_message_client

        with pytest.raises(ValueError, match="Unsupported message source"):
            # Cast to bypass type checking for test purposes
            get_message_client(source_id="invalid_source", bot_token="fake_token")  # type: ignore

    @pytest.mark.skipif(
        not os.getenv("TELEGRAM_API_ID") or not os.getenv("TELEGRAM_API_HASH"),
        reason="Telegram credentials not configured",
    )
    def test_factory_with_empty_token(self) -> None:
        """Test factory works with empty token (for testing)."""
        from src.adapters.message_client_factory import get_message_client
        from src.adapters.slack_client import SlackClient
        from src.adapters.telegram_client import TelegramClient

        slack_client = get_message_client(source_id=MessageSource.SLACK, bot_token="")
        telegram_client = get_message_client(
            source_id=MessageSource.TELEGRAM, bot_token=""
        )

        assert isinstance(slack_client, SlackClient)
        assert isinstance(telegram_client, TelegramClient)
        assert telegram_client.bot_token == ""

    @pytest.mark.skipif(
        not os.getenv("TELEGRAM_API_ID") or not os.getenv("TELEGRAM_API_HASH"),
        reason="Telegram credentials not configured",
    )
    def test_factory_creates_independent_instances(self) -> None:
        """Test that factory creates independent client instances."""
        from src.adapters.message_client_factory import get_message_client

        client1 = get_message_client(
            source_id=MessageSource.TELEGRAM, bot_token="token1"
        )
        client2 = get_message_client(
            source_id=MessageSource.TELEGRAM, bot_token="token2"
        )

        assert client1 is not client2
        # Check that clients have different tokens
        if isinstance(client1, TelegramClient):
            # TelegramClient stores token directly
            token1 = client1.bot_token
        else:
            # SlackClient stores token in client.token
            token1 = client1.client.token  # type: ignore
        if isinstance(client2, TelegramClient):
            token2 = client2.bot_token
        else:
            token2 = client2.client.token  # type: ignore
        assert token1 != token2

    @pytest.mark.skipif(
        not os.getenv("TELEGRAM_API_ID") or not os.getenv("TELEGRAM_API_HASH"),
        reason="Telegram credentials not configured",
    )
    def test_factory_returns_different_types_for_different_sources(self) -> None:
        """Test that factory returns different client types for different sources."""
        from src.adapters.message_client_factory import get_message_client

        slack_client = get_message_client(
            source_id=MessageSource.SLACK, bot_token="token1"
        )
        telegram_client = get_message_client(
            source_id=MessageSource.TELEGRAM, bot_token="token2"
        )

        assert not isinstance(slack_client, type(telegram_client))
        assert slack_client.__class__.__name__ == "SlackClient"
        assert telegram_client.__class__.__name__ == "TelegramClient"
