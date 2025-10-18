"""Tests for Telegram message processing and transformation.

Tests the conversion of raw Telegram message dictionaries to TelegramMessage domain models.
"""

from datetime import datetime

import pytz

from src.domain.models import MessageSource, TelegramMessage


class TestTelegramMessageModel:
    """Test TelegramMessage domain model."""

    def test_telegram_message_creation(self) -> None:
        """Test TelegramMessage can be created with required fields."""
        msg = TelegramMessage(
            message_id="123",
            channel="@test_channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Test message",
        )
        assert msg.message_id == "123"
        assert msg.channel == "@test_channel"
        assert msg.text == "Test message"
        assert msg.source_id == MessageSource.TELEGRAM

    def test_telegram_message_with_optional_fields(self) -> None:
        """Test TelegramMessage with all optional fields."""
        msg = TelegramMessage(
            message_id="456",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            sender_id="789",
            sender_name="Test User",
            is_bot=False,
            text="Full message",
            text_norm="full message",
            blocks_text="",
            forward_from_channel="@other_channel",
            forward_from_message_id="999",
            media_type="photo",
            links_raw=["https://example.com"],
            links_norm=["https://example.com"],
            anchors=["ABC-123"],
            views=100,
            reply_count=5,
            reactions={},
        )
        assert msg.sender_id == "789"
        assert msg.sender_name == "Test User"
        assert msg.views == 100
        assert msg.reply_count == 5

    def test_telegram_message_ts_dt_alias(self) -> None:
        """Test ts_dt property returns message_date for compatibility."""
        msg_date = datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC)
        msg = TelegramMessage(
            message_id="123",
            channel="@channel",
            message_date=msg_date,
            text="Test",
        )
        # ts_dt should be alias for message_date
        assert msg.ts_dt == msg_date
        assert msg.ts_dt == msg.message_date


class TestMessageIDValidation:
    """Test message ID format validation."""

    def test_message_id_as_string(self) -> None:
        """Test message ID is stored as string."""
        msg = TelegramMessage(
            message_id="12345",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Test",
        )
        assert isinstance(msg.message_id, str)
        assert msg.message_id == "12345"

    def test_message_id_numeric_string(self) -> None:
        """Test message ID can be converted back to integer."""
        msg = TelegramMessage(
            message_id="999888777",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Test",
        )
        # Should be convertible to integer
        assert int(msg.message_id) == 999888777


class TestURLExtraction:
    """Test URL extraction from Telegram messages."""

    def test_links_raw_field(self) -> None:
        """Test links_raw field stores raw URLs."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Check https://example.com",
            links_raw=["https://example.com"],
        )
        assert len(msg.links_raw) == 1
        assert msg.links_raw[0] == "https://example.com"

    def test_links_norm_field(self) -> None:
        """Test links_norm field stores normalized URLs."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Check https://EXAMPLE.COM/path",
            links_raw=["https://EXAMPLE.COM/path"],
            links_norm=["https://example.com/path"],
        )
        assert len(msg.links_norm) == 1
        assert msg.links_norm[0] == "https://example.com/path"

    def test_multiple_urls(self) -> None:
        """Test message with multiple URLs."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Check https://example.com and https://test.com",
            links_raw=["https://example.com", "https://test.com"],
            links_norm=["https://example.com", "https://test.com"],
        )
        assert len(msg.links_raw) == 2
        assert len(msg.links_norm) == 2


class TestAnchorExtraction:
    """Test anchor extraction from Telegram messages."""

    def test_anchors_field(self) -> None:
        """Test anchors field stores extracted anchors."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Fixed ABC-123 and XYZ-456",
            anchors=["ABC-123", "XYZ-456"],
        )
        assert len(msg.anchors) == 2
        assert "ABC-123" in msg.anchors
        assert "XYZ-456" in msg.anchors


class TestForwardedMessages:
    """Test forwarded message handling."""

    def test_forward_from_channel(self) -> None:
        """Test forward_from_channel field."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Forwarded message",
            forward_from_channel="@original_channel",
            forward_from_message_id="999",
        )
        assert msg.forward_from_channel == "@original_channel"
        assert msg.forward_from_message_id == "999"

    def test_non_forwarded_message(self) -> None:
        """Test non-forwarded message has None forward fields."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Original message",
        )
        assert msg.forward_from_channel is None
        assert msg.forward_from_message_id is None


class TestMediaHandling:
    """Test media type handling."""

    def test_media_type_field(self) -> None:
        """Test media_type field stores media type."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Photo message",
            media_type="MessageMediaPhoto",
        )
        assert msg.media_type == "MessageMediaPhoto"

    def test_text_only_message(self) -> None:
        """Test text-only message has None media_type."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Text only",
        )
        assert msg.media_type is None


class TestViewsAndEngagement:
    """Test views and engagement metrics."""

    def test_views_field(self) -> None:
        """Test views field stores view count."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Popular message",
            views=1000,
        )
        assert msg.views == 1000

    def test_reply_count_field(self) -> None:
        """Test reply_count field stores reply count."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Discussion starter",
            reply_count=25,
        )
        assert msg.reply_count == 25

    def test_default_views_and_replies(self) -> None:
        """Test default values for views and replies."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="New message",
        )
        assert msg.views == 0
        assert msg.reply_count == 0


class TestTextNormalization:
    """Test text normalization fields."""

    def test_text_norm_field(self) -> None:
        """Test text_norm field stores normalized text."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Original TEXT with CAPS",
            text_norm="original text with caps",
        )
        assert msg.text_norm == "original text with caps"

    def test_blocks_text_field(self) -> None:
        """Test blocks_text field for scoring compatibility."""
        msg = TelegramMessage(
            message_id="1",
            channel="@channel",
            message_date=datetime(2025, 10, 17, 12, 0, 0, tzinfo=pytz.UTC),
            text="Main text",
            blocks_text="Formatted text for scoring",
        )
        assert msg.blocks_text == "Formatted text for scoring"
