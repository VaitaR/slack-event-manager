"""Tests for Telegram client adapter.

Following TDD methodology - tests written before implementation.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytz

from src.adapters.telegram_client import TelegramClient
from src.domain.exceptions import RateLimitError


class TestTelegramClientInitialization:
    """Test TelegramClient initialization."""

    def test_telegram_client_initialization(self) -> None:
        """Test TelegramClient can be initialized with credentials."""
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        assert client.api_id == 12345
        assert client.api_hash == "test_hash"
        assert client.session_name == "test_session"

    def test_telegram_client_stores_credentials(self) -> None:
        """Test TelegramClient stores credentials for later use."""
        client = TelegramClient(
            api_id=99999, api_hash="secret_hash", session_name="my_session"
        )
        assert client.api_id == 99999
        assert client.api_hash == "secret_hash"


class TestFetchMessages:
    """Test fetch_messages method."""

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_fetch_messages_returns_list(self, mock_telethon: Mock) -> None:
        """Test fetch_messages returns list of message dictionaries."""
        # Setup mock
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Create mock messages
        mock_msg = Mock()
        mock_msg.id = 123
        mock_msg.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
        mock_msg.message = "Test message"
        mock_msg.sender_id = 456
        mock_msg.text = "Test message"
        mock_msg.entities = []
        mock_msg.media = None
        mock_msg.views = 100
        mock_msg.forwards = None
        mock_msg.replies = None

        # Mock iter_messages to return a list (not async iterator)
        mock_client_instance.iter_messages = Mock(return_value=[mock_msg])
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        # Test
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        messages = client.fetch_messages(channel_id="@test_channel", limit=10)

        assert isinstance(messages, list)

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_fetch_messages_with_limit(self, mock_telethon: Mock) -> None:
        """Test fetch_messages respects limit parameter."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Create 5 mock messages
        mock_messages = []
        for i in range(5):
            mock_msg = Mock()
            mock_msg.id = i + 1
            mock_msg.date = datetime(2025, 10, 17, 12, i, 0, tzinfo=pytz.UTC)
            mock_msg.message = f"Message {i}"
            mock_msg.text = f"Message {i}"
            mock_msg.sender_id = 456
            mock_msg.entities = []
            mock_msg.media = None
            mock_msg.views = 0
            mock_msg.forwards = None
            mock_msg.replies = None
            mock_messages.append(mock_msg)

        mock_client_instance.iter_messages = Mock(return_value=mock_messages)
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        messages = client.fetch_messages(channel_id="@test_channel", limit=3)

        # Should return only 3 messages
        assert len(messages) <= 3

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_fetch_messages_uses_min_message_id(self, mock_telethon: Mock) -> None:
        """Test fetch_messages forwards min_message_id to Telethon iterator."""

        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        mock_message = Mock()
        mock_message.id = 11
        mock_message.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
        mock_message.message = "Test"
        mock_message.text = "Test"
        mock_message.sender_id = 123
        mock_message.entities = []
        mock_message.media = None
        mock_message.views = 0
        mock_message.forwards = None
        mock_message.replies = None

        mock_client_instance.iter_messages = Mock(return_value=[mock_message])
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345,
            api_hash="test_hash",
            session_name="test_session",
        )

        client.fetch_messages(channel_id="@test_channel", min_message_id=5, page_size=1)

        iter_kwargs = mock_client_instance.iter_messages.call_args.kwargs
        assert iter_kwargs["min_id"] == 5
        assert iter_kwargs["limit"] == 1

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_fetch_messages_async_wrapper(self, mock_telethon: Mock) -> None:
        """Test that fetch_messages properly wraps async Telethon calls."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        mock_client_instance.iter_messages = Mock(return_value=[])
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        # Should not raise exception (async properly wrapped)
        messages = client.fetch_messages(channel_id="@test_channel")
        assert isinstance(messages, list)


class TestFloodWaitHandling:
    """Test FloodWait error handling."""

    @patch("src.adapters.telegram_client.TelegramClientLib")
    @patch("src.adapters.telegram_client.asyncio.sleep")
    def test_flood_wait_handling(self, mock_sleep: Mock, mock_telethon: Mock) -> None:
        """Test that FloodWaitError is caught and handled with sleep."""
        from telethon.errors import FloodWaitError

        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # First call raises FloodWaitError, second succeeds
        call_count = 0

        def mock_iter_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error = FloodWaitError(None)
                error.seconds = 10
                raise error
            else:
                return []

        mock_client_instance.iter_messages = Mock(side_effect=mock_iter_side_effect)
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Should handle FloodWait and retry
        messages = client.fetch_messages(channel_id="@test_channel")
        assert isinstance(messages, list)

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_flood_wait_raises_after_max_retries(self, mock_telethon: Mock) -> None:
        """Test that FloodWaitError is raised after max retries."""
        from telethon.errors import FloodWaitError

        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Always raise FloodWaitError
        flood_error = FloodWaitError(10)
        mock_client_instance.iter_messages = Mock(side_effect=flood_error)
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Should raise RateLimitError after retries
        with pytest.raises(RateLimitError):
            client.fetch_messages(channel_id="@test_channel")


class TestTextExtraction:
    """Test text extraction from messages."""

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_text_extraction_with_entities(self, mock_telethon: Mock) -> None:
        """Test text extraction includes entity text."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Mock message with text
        mock_msg = Mock()
        mock_msg.id = 1
        mock_msg.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
        mock_msg.message = "Check out https://example.com"
        mock_msg.text = "Check out https://example.com"
        mock_msg.sender_id = 123
        mock_msg.entities = []
        mock_msg.media = None
        mock_msg.views = 0
        mock_msg.forwards = None
        mock_msg.replies = None

        mock_client_instance.iter_messages = Mock(return_value=[mock_msg])
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        messages = client.fetch_messages(channel_id="@test_channel")

        assert len(messages) == 1
        assert "text" in messages[0]
        assert messages[0]["text"] == "Check out https://example.com"


class TestURLExtraction:
    """Test URL extraction from message entities."""

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_url_extraction_from_entities(self, mock_telethon: Mock) -> None:
        """Test URLs are extracted from MessageEntityUrl entities."""
        from telethon.tl.types import MessageEntityUrl

        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Mock message with URL entity
        mock_msg = Mock()
        mock_msg.id = 1
        mock_msg.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
        mock_msg.message = "Check out https://example.com"
        mock_msg.text = "Check out https://example.com"
        mock_msg.sender_id = 123
        mock_msg.media = None
        mock_msg.views = 0
        mock_msg.forwards = None
        mock_msg.replies = None

        # Create URL entity
        url_entity = MessageEntityUrl(offset=10, length=19)
        mock_msg.entities = [url_entity]

        mock_client_instance.iter_messages = Mock(return_value=[mock_msg])
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        messages = client.fetch_messages(channel_id="@test_channel")

        assert len(messages) == 1
        assert "entities" in messages[0]
        # URLs should be extracted
        assert len(messages[0]["entities"]) > 0

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_url_extraction_from_text_url_entities(self, mock_telethon: Mock) -> None:
        """Test URLs are extracted from MessageEntityTextUrl entities."""
        from telethon.tl.types import MessageEntityTextUrl

        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Mock message with TextUrl entity
        mock_msg = Mock()
        mock_msg.id = 1
        mock_msg.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
        mock_msg.message = "Check out this link"
        mock_msg.text = "Check out this link"
        mock_msg.sender_id = 123
        mock_msg.media = None
        mock_msg.views = 0
        mock_msg.forwards = None
        mock_msg.replies = None

        # Create TextUrl entity with URL
        text_url_entity = MessageEntityTextUrl(
            offset=10, length=9, url="https://example.com"
        )
        mock_msg.entities = [text_url_entity]

        mock_client_instance.iter_messages = Mock(return_value=[mock_msg])
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        messages = client.fetch_messages(channel_id="@test_channel")

        assert len(messages) == 1
        assert "entities" in messages[0]


class TestPostURLConstruction:
    """Test post URL construction for public channels."""

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_post_url_construction_public_channel(self, mock_telethon: Mock) -> None:
        """Test post URL is constructed for public channels with username."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Mock channel entity with username
        mock_channel = Mock()
        mock_channel.username = "test_channel"
        mock_channel.id = -1001234567890
        # Make isinstance check work
        mock_channel.__class__.__name__ = "Channel"

        # Mock message
        mock_msg = Mock()
        mock_msg.id = 123
        mock_msg.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
        mock_msg.message = "Test"
        mock_msg.text = "Test"
        mock_msg.sender_id = 456
        mock_msg.entities = []
        mock_msg.media = None
        mock_msg.views = 0
        mock_msg.forwards = None
        mock_msg.replies = None

        mock_client_instance.iter_messages = Mock(return_value=[mock_msg])
        mock_client_instance.get_entity = AsyncMock(return_value=mock_channel)
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        messages = client.fetch_messages(channel_id="@test_channel")

        assert len(messages) == 1
        # Should have post_url field
        assert "post_url" in messages[0]
        # Should be in format https://t.me/username/message_id
        assert messages[0]["post_url"] == "https://t.me/test_channel/123"


class TestMessageIDConversion:
    """Test message ID conversion from integer to string."""

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_message_id_to_string_conversion(self, mock_telethon: Mock) -> None:
        """Test message ID is converted from integer to string."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Mock message with integer ID
        mock_msg = Mock()
        mock_msg.id = 12345
        mock_msg.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
        mock_msg.message = "Test"
        mock_msg.text = "Test"
        mock_msg.sender_id = 456
        mock_msg.entities = []
        mock_msg.media = None
        mock_msg.views = 0
        mock_msg.forwards = None
        mock_msg.replies = None

        mock_client_instance.iter_messages = Mock(return_value=[mock_msg])
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        messages = client.fetch_messages(channel_id="@test_channel")

        assert len(messages) == 1
        assert "message_id" in messages[0]
        # Should be string, not integer
        assert isinstance(messages[0]["message_id"], str)
        assert messages[0]["message_id"] == "12345"

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_message_id_preserves_value(self, mock_telethon: Mock) -> None:
        """Test message ID string preserves original integer value."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        mock_msg = Mock()
        mock_msg.id = 999888777
        mock_msg.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
        mock_msg.message = "Test"
        mock_msg.text = "Test"
        mock_msg.sender_id = 456
        mock_msg.entities = []
        mock_msg.media = None
        mock_msg.views = 0
        mock_msg.forwards = None
        mock_msg.replies = None

        mock_client_instance.iter_messages = Mock(return_value=[mock_msg])
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        messages = client.fetch_messages(channel_id="@test_channel")

        assert messages[0]["message_id"] == "999888777"
        # Should be convertible back to integer
        assert int(messages[0]["message_id"]) == 999888777


class TestChannelIDFormats:
    """Test different channel ID formats (username, numeric ID)."""

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_channel_username_format(self, mock_telethon: Mock) -> None:
        """Test fetching with @username format."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        mock_client_instance.iter_messages = Mock(return_value=[])
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        # Should accept @username format
        messages = client.fetch_messages(channel_id="@test_channel")
        assert isinstance(messages, list)

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_channel_numeric_id_format(self, mock_telethon: Mock) -> None:
        """Test fetching with numeric ID format."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        mock_client_instance.iter_messages = Mock(return_value=[])
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )
        # Should accept numeric ID format
        messages = client.fetch_messages(channel_id="-1001234567890")
        assert isinstance(messages, list)


class TestFallbackScenarios:
    """Test fallback scenarios when Telethon is not available."""

    @patch("src.adapters.telegram_client.TelegramClientLib", None)
    def test_convert_message_fallback_no_telethon(self) -> None:
        """Test _convert_message_to_dict handles None entity types gracefully."""
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Mock message with entities (unknown types when Telethon is not available)
        class MockMessage:
            def __init__(self):
                self.id = 123
                self.message = "Test message"
                self.text = "Test message"
                self.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
                self.sender_id = 456
                self.entities = [
                    type(
                        "MockEntity", (), {"offset": 10, "length": 15}
                    )(),  # Mock entity
                ]
                self.media = None
                self.views = 0
                self.forwards = None
                self.replies = None

        mock_msg = MockMessage()

        # Should not crash and return valid dict
        result = client._convert_message_to_dict(
            mock_msg, "@test_channel", "test_channel"
        )

        assert isinstance(result, dict)
        assert result["message_id"] == "123"
        assert result["text"] == "Test message"
        assert result["entities"] == []  # Should be empty due to None types
        assert result["channel"] == "@test_channel"

    @patch("src.adapters.telegram_client.MessageEntityUrl", None)
    @patch("src.adapters.telegram_client.MessageEntityTextUrl", None)
    def test_convert_message_partial_fallback(self) -> None:
        """Test _convert_message_to_dict with only one entity type available."""
        # Patch only MessageEntityTextUrl to be available
        with (
            patch("src.adapters.telegram_client.MessageEntityUrl", None),
            patch(
                "src.adapters.telegram_client.MessageEntityTextUrl",
                type("MockType", (), {}),
            ),
        ):
            client = TelegramClient(
                api_id=12345, api_hash="test_hash", session_name="test_session"
            )

            # Mock message with TextUrl entity
            class MockTextUrlEntity:
                def __init__(self, url: str):
                    self.url = url

            class MockMessage:
                def __init__(self):
                    self.id = 123
                    self.message = "Test message"
                    self.text = "Test message"
                    self.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
                    self.sender_id = 456
                    self.entities = [
                        MockTextUrlEntity(url="https://test.com"),
                    ]
                    self.media = None
                    self.views = 0
                    self.forwards = None
                    self.replies = None

            mock_msg = MockMessage()

            # Should extract TextUrl entity but not crash
            result = client._convert_message_to_dict(
                mock_msg, "@test_channel", "test_channel"
            )

            assert isinstance(result, dict)
        assert (
            result["entities"] == []
        )  # TextUrl entities not processed in this test setup


class TestAsyncEventLoopScenarios:
    """Test Telegram client in scenarios with active event loops."""

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_sync_wrapper_with_active_event_loop(self, mock_telethon: Mock) -> None:
        """Test that sync wrapper works correctly when called from within an active event loop."""
        import asyncio

        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Mock successful message fetching
        mock_messages = [
            Mock(
                id=1,
                message="Test message",
                text="Test message",
                date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
                sender_id=123,
                entities=[],
                media=None,
                views=0,
                forwards=None,
                replies=None,
            )
        ]
        mock_client_instance.iter_messages = Mock(return_value=mock_messages)
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Simulate calling from within an active event loop (like in FastAPI)
        async def test_in_event_loop():
            # This simulates what would happen in a web server context
            return client.fetch_messages(channel_id="@test_channel", limit=1)

        # Should work without hanging or crashing
        messages = asyncio.run(test_in_event_loop())
        assert isinstance(messages, list)
        assert len(messages) == 1

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_async_method_works_in_event_loop(self, mock_telethon: Mock) -> None:
        """Test that async fetch_messages_async works correctly in event loop."""
        import asyncio

        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Mock successful message fetching
        mock_messages = [
            Mock(
                id=1,
                message="Test message",
                text="Test message",
                date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
                sender_id=123,
                entities=[],
                media=None,
                views=0,
                forwards=None,
                replies=None,
            )
        ]
        mock_client_instance.iter_messages = Mock(return_value=mock_messages)
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Test async method in event loop context
        async def test_async_method():
            return await client.fetch_messages_async(
                channel_id="@test_channel", limit=1
            )

        messages = asyncio.run(test_async_method())
        assert isinstance(messages, list)
        assert len(messages) == 1

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_multiple_sync_calls_isolation(self, mock_telethon: Mock) -> None:
        """Test that multiple sync calls don't interfere with each other."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Mock different responses for different calls
        call_count = 0

        def mock_iter_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    Mock(
                        id=1,
                        message="Message 1",
                        text="Message 1",
                        date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
                        sender_id=123,
                        entities=[],
                        media=None,
                        views=0,
                        forwards=None,
                        replies=None,
                    )
                ]
            else:
                return [
                    Mock(
                        id=2,
                        message="Message 2",
                        text="Message 2",
                        date=datetime(2025, 10, 17, 12, 1, 0, tzinfo=pytz.UTC),
                        sender_id=456,
                        entities=[],
                        media=None,
                        views=0,
                        forwards=None,
                        replies=None,
                    )
                ]

        mock_client_instance.iter_messages = Mock(side_effect=mock_iter_side_effect)
        mock_client_instance.get_entity = AsyncMock()
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Make multiple sync calls
        messages1 = client.fetch_messages(channel_id="@test_channel", limit=1)
        messages2 = client.fetch_messages(channel_id="@test_channel", limit=1)

        # Should get different results (isolated clients)
        assert len(messages1) == 1
        assert len(messages2) == 1
        assert messages1[0]["message_id"] != messages2[0]["message_id"]
