"""End-to-end tests for Telegram integration.

Tests the complete Telegram pipeline with mocked Telethon client.
"""

from datetime import datetime
from unittest.mock import Mock

import pytz

from src.adapters.sqlite_repository import SQLiteRepository
from src.adapters.telegram_client import TelegramClient
from src.config.settings import get_settings
from src.domain.models import MessageSource, TelegramChannelConfig
from src.use_cases.ingest_telegram_messages import ingest_telegram_messages_use_case


class TestTelegramE2E:
    """End-to-end tests for Telegram integration."""

    def test_telegram_ingestion_e2e(self) -> None:
        """Test complete Telegram ingestion flow."""
        # Create Telegram client
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Mock the fetch_messages method directly
        mock_messages = [
            {
                "id": i + 1,  # Numeric message ID for each message
                "message_id": i + 1,  # Add message_id field for compatibility
                "date": datetime(2025, 10, 17, 12, i, 0, tzinfo=pytz.UTC),
                "text": f"Test message {i}",
                "sender_id": f"12345{i}",
                "channel": "@test_channel",
            }
            for i in range(5)
        ]

        print(f"Mock messages: {[msg['id'] for msg in mock_messages]}")
        client.fetch_messages = Mock(return_value=mock_messages)

        # Create test database with unique file name
        import os
        import uuid

        db_path = f"/tmp/test_telegram_{uuid.uuid4()}.db"
        # Clean up after test
        try:
            repository = SQLiteRepository(db_path=db_path)

            # Ensure ingestion_state_telegram table exists (debug)
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_state_telegram (
                    channel_id TEXT PRIMARY KEY,
                    last_processed_ts REAL NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            conn.commit()
            conn.close()

            # Create mock settings with telegram_channels
            settings = get_settings()

            settings.telegram_channels = [
                TelegramChannelConfig(
                    username="@test_channel",
                    channel_name="Test Channel",
                )
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
            messages = repository.get_telegram_messages(
                channel="@test_channel", limit=10
            )
            print(f"Found {len(messages)} messages in database")
            for i, msg in enumerate(messages):
                print(f"Message {i + 1}: id='{msg.message_id}', text='{msg.text}'")
            assert len(messages) == 5

            # Verify message content
            first_msg = messages[0]
            assert first_msg.channel == "@test_channel"
            assert "Test message" in first_msg.text
            assert first_msg.source_id == MessageSource.TELEGRAM

            # Clean up database after test
            import os

            if os.path.exists(db_path):
                os.unlink(db_path)

        except Exception:
            # Clean up database on error
            import os

            if os.path.exists(db_path):
                os.unlink(db_path)
            raise

    def test_telegram_ingestion_with_urls(self) -> None:
        """Test Telegram ingestion extracts URLs correctly."""
        # Create Telegram client
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Mock the fetch_messages method directly with URL data
        client.fetch_messages = Mock(
            return_value=[
                {
                    "id": 1,
                    "message_id": 1,
                    "date": datetime.now(pytz.UTC).replace(
                        hour=12, minute=0, second=0, microsecond=0
                    ),  # Use current date
                    "text": "Check https://example.com",
                    "sender_id": "123456",
                    "channel": "@test_channel",
                    "entities": [],  # Simplified for this test
                }
            ]
        )

        # Create test database with unique file name
        import os
        import uuid

        db_path = f"/tmp/test_telegram_{uuid.uuid4()}.db"
        # Clean up after test
        try:
            repository = SQLiteRepository(db_path=db_path)

            # Ensure ingestion_state_telegram table exists (debug)
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_state_telegram (
                    channel_id TEXT PRIMARY KEY,
                    last_processed_ts REAL NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        finally:
            pass  # Don't delete DB here, we'll delete it at the end

        settings = get_settings()
        settings.telegram_channels = [
            TelegramChannelConfig(
                username="@test_channel",
                channel_name="Test Channel",
            )
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

        # Clean up database after test

        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_telegram_ingestion_disabled_channel(self) -> None:
        """Test disabled channels are skipped."""
        # Create Telegram client (no need to mock since disabled channels don't fetch)
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Mock empty fetch_messages since disabled channels shouldn't fetch
        client.fetch_messages = Mock(return_value=[])

        # Create test database with unique file name
        import os
        import uuid

        db_path = f"/tmp/test_telegram_{uuid.uuid4()}.db"
        # Clean up after test
        try:
            repository = SQLiteRepository(db_path=db_path)

            # Ensure ingestion_state_telegram table exists (debug)
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_state_telegram (
                    channel_id TEXT PRIMARY KEY,
                    last_processed_ts REAL NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        finally:
            pass  # Don't delete DB here, we'll delete it at the end

        settings = get_settings()
        settings.telegram_channels = [
            TelegramChannelConfig(
                username="@disabled_channel",
                channel_name="Disabled Channel",
                enabled=False,  # Explicitly disable this channel
            )
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

        # Clean up database after test

        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_telegram_ingestion_no_channels(self) -> None:
        """Test behavior when no channels configured."""
        # Create Telegram client
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Create test database with unique file name
        import os
        import uuid

        db_path = f"/tmp/test_telegram_{uuid.uuid4()}.db"
        # Clean up after test
        try:
            repository = SQLiteRepository(db_path=db_path)

            # Ensure ingestion_state_telegram table exists (debug)
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_state_telegram (
                    channel_id TEXT PRIMARY KEY,
                    last_processed_ts REAL NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        finally:
            pass  # Don't delete DB here, we'll delete it at the end

        settings = get_settings()
        settings.telegram_channels = []  # No channels - this is fine as TelegramChannelConfig list

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

        # Clean up database after test

        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_telegram_ingestion_incremental(self) -> None:
        """Test incremental ingestion (only new messages)."""
        # Create Telegram client
        client = TelegramClient(
            api_id=12345, api_hash="test_hash", session_name="test_session"
        )

        # Mock the fetch_messages method directly with 3 messages
        client.fetch_messages = Mock(
            return_value=[
                {
                    "id": i + 1,
                    "message_id": i + 1,
                    "date": datetime.now(pytz.UTC).replace(
                        hour=12, minute=i, second=0, microsecond=0
                    ),  # Use current date with different minutes
                    "text": f"Message {i}",
                    "sender_id": f"12345{i}",
                    "channel": "@test_channel",
                }
                for i in range(3)
            ]
        )

        # Create test database with unique file name
        import os
        import uuid

        db_path = f"/tmp/test_telegram_{uuid.uuid4()}.db"
        # Clean up after test
        try:
            repository = SQLiteRepository(db_path=db_path)

            # Ensure ingestion_state_telegram table exists (debug)
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_state_telegram (
                    channel_id TEXT PRIMARY KEY,
                    last_processed_ts REAL NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        finally:
            pass  # Don't delete DB here, we'll delete it at the end

        settings = get_settings()
        settings.telegram_channels = [
            TelegramChannelConfig(
                username="@test_channel",
                channel_name="Test Channel",
            )
        ]

        # First ingestion
        result1 = ingest_telegram_messages_use_case(
            telegram_client=client,
            repository=repository,
            settings=settings,
        )

        assert result1.messages_saved == 3

        # Second run: 2 new messages (IDs 4, 5)
        client.fetch_messages = Mock(
            return_value=[
                {
                    "id": i + 1,
                    "message_id": i + 1,
                    "date": datetime.now(pytz.UTC).replace(
                        hour=12, minute=i, second=0, microsecond=0
                    ),
                    "text": f"Message {i}",
                    "sender_id": f"12345{i}",
                    "channel": "@test_channel",
                }
                for i in range(3, 5)  # Messages 3, 4 (IDs 4, 5)
            ]
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

        # Clean up database after test

        if os.path.exists(db_path):
            os.unlink(db_path)
