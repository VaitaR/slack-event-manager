"""
Tests for TelegramClient stub.

Following TDD: Write tests first, then implement in src/adapters/telegram_client.py
"""


class TestTelegramClientStub:
    """Test TelegramClient stub implementation."""

    def test_telegram_client_exists(self) -> None:
        """Test that TelegramClient can be imported."""
        from src.adapters.telegram_client import TelegramClient

        assert TelegramClient is not None

    def test_telegram_client_initialization(self) -> None:
        """Test TelegramClient can be initialized with bot token."""
        from src.adapters.telegram_client import TelegramClient

        client = TelegramClient(bot_token="fake_token")

        assert client is not None
        assert hasattr(client, "bot_token")

    def test_fetch_messages_returns_empty_list(self) -> None:
        """Test that fetch_messages returns empty list (stub behavior)."""
        from src.adapters.telegram_client import TelegramClient

        client = TelegramClient(bot_token="fake_token")

        messages = client.fetch_messages(channel_id="test_channel")

        assert isinstance(messages, list)
        assert len(messages) == 0

    def test_fetch_messages_accepts_all_parameters(self) -> None:
        """Test that fetch_messages accepts all protocol parameters."""
        from src.adapters.telegram_client import TelegramClient

        client = TelegramClient(bot_token="fake_token")

        # Should not raise any errors
        messages = client.fetch_messages(
            channel_id="test_channel",
            oldest_ts="1000000",
            latest_ts="2000000",
            limit=50,
        )

        assert isinstance(messages, list)
        assert len(messages) == 0

    def test_get_user_info_returns_empty_dict(self) -> None:
        """Test that get_user_info returns empty dict (stub behavior)."""
        from src.adapters.telegram_client import TelegramClient

        client = TelegramClient(bot_token="fake_token")

        user_info = client.get_user_info(user_id="123456")

        assert isinstance(user_info, dict)
        assert len(user_info) == 0

    def test_telegram_client_implements_protocol(self) -> None:
        """Test that TelegramClient implements MessageClientProtocol."""
        from src.adapters.telegram_client import TelegramClient

        client = TelegramClient(bot_token="fake_token")

        # Check that client has required protocol methods
        assert hasattr(client, "fetch_messages")
        assert hasattr(client, "get_user_info")
        assert callable(client.fetch_messages)
        assert callable(client.get_user_info)

    def test_telegram_client_stores_bot_token(self) -> None:
        """Test that TelegramClient stores the bot token."""
        from src.adapters.telegram_client import TelegramClient

        client = TelegramClient(bot_token="test_token_12345")

        assert client.bot_token == "test_token_12345"

    def test_telegram_client_with_empty_token(self) -> None:
        """Test that TelegramClient accepts empty token (for testing)."""
        from src.adapters.telegram_client import TelegramClient

        client = TelegramClient(bot_token="")

        assert client.bot_token == ""
        assert client.fetch_messages(channel_id="test") == []

    def test_fetch_messages_ignores_parameters(self) -> None:
        """Test that fetch_messages ignores all parameters (stub)."""
        from src.adapters.telegram_client import TelegramClient

        client = TelegramClient(bot_token="fake_token")

        # Different parameters should all return empty list
        result1 = client.fetch_messages(channel_id="channel1", limit=100)
        result2 = client.fetch_messages(channel_id="channel2", limit=10)
        result3 = client.fetch_messages(
            channel_id="channel3", oldest_ts="123", latest_ts="456"
        )

        assert result1 == []
        assert result2 == []
        assert result3 == []

    def test_get_user_info_ignores_user_id(self) -> None:
        """Test that get_user_info ignores user_id (stub)."""
        from src.adapters.telegram_client import TelegramClient

        client = TelegramClient(bot_token="fake_token")

        # Different user IDs should all return empty dict
        result1 = client.get_user_info(user_id="123")
        result2 = client.get_user_info(user_id="456")
        result3 = client.get_user_info(user_id="")

        assert result1 == {}
        assert result2 == {}
        assert result3 == {}
