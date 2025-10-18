"""End-to-end tests for Telegram integration.

Tests the complete Telegram pipeline with mocked Telethon client.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytz

from src.adapters.sqlite_repository import SQLiteRepository
from src.adapters.telegram_client import TelegramClient
from src.config.settings import get_settings
from src.domain.models import MessageSource
from src.use_cases.ingest_telegram_messages import ingest_telegram_messages_use_case


class TestTelegramE2E:
    """End-to-end tests for Telegram integration."""

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_telegram_ingestion_e2e(self, mock_telethon: Mock) -> None:
        """Test complete Telegram ingestion flow."""
        # Setup mock Telethon client
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Mock channel entity
        mock_channel = Mock()
        mock_channel.username = "test_channel"
        mock_channel.id = -1001234567890

        # Create mock messages
        mock_messages = []
        for i in range(5):
            mock_msg = Mock()
            mock_msg.id = i + 1
            mock_msg.date = datetime(2025, 10, 17, 12, i, 0, tzinfo=pytz.UTC)
            mock_msg.message = f"Test message {i}"
            mock_msg.text = f"Test message {i}"
            mock_msg.sender_id = 123456
            mock_msg.entities = []
            mock_msg.media = None
            mock_msg.views = i * 10
            mock_msg.forwards = None
            mock_msg.replies = None
            mock_messages.append(mock_msg)

        mock_client_instance.iter_messages = AsyncMock(return_value=mock_messages)
        mock_client_instance.get_entity = AsyncMock(return_value=mock_channel)
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        # Create test database with unique file name
        import os
        import uuid

        db_path = f"/tmp/test_telegram_{uuid.uuid4()}.db"
        repository = SQLiteRepository(db_path=db_path)
        # Clean up after test
        try:
            # Repository is created and will be used by the test
            pass
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

        # Create Telegram client
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Create mock settings with telegram_channels
        settings = get_settings()
        settings.telegram_channels = [
            {
                "channel_id": "@test_channel",
                "channel_name": "Test Channel",
                "from_date": "2025-10-16T00:00:00Z",
                "enabled": True,
            }
        ]

        # Run ingestion
        result = ingest_telegram_messages_use_case(
            telegram_client=client,
            repository=repository,
            settings=settings,
            backfill_from_date=datetime(2025, 10, 16, 0, 0, 0, tzinfo=pytz.UTC),
        )

        # Verify results
        assert result.messages_fetched == 5
        assert result.messages_saved == 5
        assert "@test_channel" in result.channels_processed
        assert len(result.errors) == 0

        # Verify messages in database
        messages = repository.get_telegram_messages(channel="@test_channel", limit=10)
        assert len(messages) == 5

        # Verify message content
        first_msg = messages[0]
        assert first_msg.channel == "@test_channel"
        assert "Test message" in first_msg.text
        assert first_msg.source_id == MessageSource.TELEGRAM

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_telegram_ingestion_with_urls(self, mock_telethon: Mock) -> None:
        """Test Telegram ingestion extracts URLs correctly."""
        from telethon.tl.types import MessageEntityUrl

        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        mock_channel = Mock()
        mock_channel.username = "test_channel"

        # Message with URL entity
        mock_msg = Mock()
        mock_msg.id = 1
        mock_msg.date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
        mock_msg.message = "Check https://example.com"
        mock_msg.text = "Check https://example.com"
        mock_msg.sender_id = 123
        mock_msg.media = None
        mock_msg.views = 0
        mock_msg.forwards = None
        mock_msg.replies = None

        # Add URL entity
        url_entity = MessageEntityUrl(offset=6, length=19)
        mock_msg.entities = [url_entity]

        mock_client_instance.iter_messages = AsyncMock(return_value=[mock_msg])
        mock_client_instance.get_entity = AsyncMock(return_value=mock_channel)
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        # Create test database with unique file name
        import os
        import uuid

        db_path = f"/tmp/test_telegram_{uuid.uuid4()}.db"
        repository = SQLiteRepository(db_path=db_path)
        # Clean up after test
        try:
            # Repository is created and will be used by the test
            pass
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        settings = get_settings()
        settings.telegram_channels = [
            {
                "channel_id": "@test_channel",
                "enabled": True,
            }
        ]

        result = ingest_telegram_messages_use_case(
            telegram_client=client,
            repository=repository,
            settings=settings,
        )

        assert result.messages_saved == 1

        # Verify URL extraction
        messages = repository.get_telegram_messages(channel="@test_channel", limit=1)
        assert len(messages) == 1
        assert len(messages[0].links_raw) > 0
        assert "example.com" in messages[0].links_raw[0]

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_telegram_ingestion_disabled_channel(self, mock_telethon: Mock) -> None:
        """Test disabled channels are skipped."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        # Create test database with unique file name
        import os
        import uuid

        db_path = f"/tmp/test_telegram_{uuid.uuid4()}.db"
        repository = SQLiteRepository(db_path=db_path)
        # Clean up after test
        try:
            # Repository is created and will be used by the test
            pass
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        settings = get_settings()
        settings.telegram_channels = [
            {
                "channel_id": "@disabled_channel",
                "enabled": False,  # Disabled
            }
        ]

        result = ingest_telegram_messages_use_case(
            telegram_client=client,
            repository=repository,
            settings=settings,
        )

        # Should skip disabled channel
        assert result.messages_fetched == 0
        assert result.messages_saved == 0
        assert len(result.channels_processed) == 0

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_telegram_ingestion_no_channels(self, mock_telethon: Mock) -> None:
        """Test behavior when no channels configured."""
        # Create test database with unique file name
        import os
        import uuid

        db_path = f"/tmp/test_telegram_{uuid.uuid4()}.db"
        repository = SQLiteRepository(db_path=db_path)
        # Clean up after test
        try:
            # Repository is created and will be used by the test
            pass
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        settings = get_settings()
        settings.telegram_channels = []  # No channels

        result = ingest_telegram_messages_use_case(
            telegram_client=client,
            repository=repository,
            settings=settings,
        )

        # Should return empty result with error
        assert result.messages_fetched == 0
        assert result.messages_saved == 0
        assert len(result.errors) > 0
        assert "No Telegram channels configured" in result.errors[0]

    @patch("src.adapters.telegram_client.TelegramClientLib")
    def test_telegram_ingestion_incremental(self, mock_telethon: Mock) -> None:
        """Test incremental ingestion (only new messages)."""
        mock_client_instance = AsyncMock()
        mock_telethon.return_value = mock_client_instance

        mock_channel = Mock()
        mock_channel.username = "test_channel"

        # First run: 3 messages
        mock_messages_first = []
        for i in range(3):
            mock_msg = Mock()
            mock_msg.id = i + 1
            mock_msg.date = datetime(2025, 10, 17, 12, i, 0, tzinfo=pytz.UTC)
            mock_msg.message = f"Message {i}"
            mock_msg.text = f"Message {i}"
            mock_msg.sender_id = 123
            mock_msg.entities = []
            mock_msg.media = None
            mock_msg.views = 0
            mock_msg.forwards = None
            mock_msg.replies = None
            mock_messages_first.append(mock_msg)

        mock_client_instance.iter_messages = AsyncMock(return_value=mock_messages_first)
        mock_client_instance.get_entity = AsyncMock(return_value=mock_channel)
        mock_client_instance.start = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        # Create test database with unique file name
        import os
        import uuid

        db_path = f"/tmp/test_telegram_{uuid.uuid4()}.db"
        repository = SQLiteRepository(db_path=db_path)
        # Clean up after test
        try:
            # Repository is created and will be used by the test
            pass
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        settings = get_settings()
        settings.telegram_channels = [
            {
                "channel_id": "@test_channel",
                "enabled": True,
            }
        ]

        # First ingestion
        result1 = ingest_telegram_messages_use_case(
            telegram_client=client,
            repository=repository,
            settings=settings,
        )

        assert result1.messages_saved == 3

        # Second run: 2 new messages (IDs 4, 5)
        mock_messages_second = []
        for i in range(3, 5):
            mock_msg = Mock()
            mock_msg.id = i + 1
            mock_msg.date = datetime(2025, 10, 17, 12, i, 0, tzinfo=pytz.UTC)
            mock_msg.message = f"Message {i}"
            mock_msg.text = f"Message {i}"
            mock_msg.sender_id = 123
            mock_msg.entities = []
            mock_msg.media = None
            mock_msg.views = 0
            mock_msg.forwards = None
            mock_msg.replies = None
            mock_messages_second.append(mock_msg)

        mock_client_instance.iter_messages = AsyncMock(
            return_value=mock_messages_second
        )

        # Second ingestion (incremental)
        result2 = ingest_telegram_messages_use_case(
            telegram_client=client,
            repository=repository,
            settings=settings,
        )

        # Should save only new messages
        assert result2.messages_saved == 2

        # Total messages in DB
        all_messages = repository.get_telegram_messages(
            channel="@test_channel", limit=10
        )
        assert len(all_messages) == 5
