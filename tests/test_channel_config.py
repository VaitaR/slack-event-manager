"""Tests for enhanced channel configuration functionality."""

import pytest
from pydantic import ValidationError

from src.domain.models import ChannelConfig, TelegramChannelConfig


class TestChannelConfig:
    """Test enhanced ChannelConfig with prompt_file support."""

    def test_channel_config_with_prompt_file(self) -> None:
        """Test ChannelConfig with custom prompt file."""
        config = ChannelConfig(
            channel_id="C123456",
            channel_name="releases",
            prompt_file="config/prompts/releases.yaml",
        )

        assert config.prompt_file == "config/prompts/releases.yaml"
        assert config.channel_id == "C123456"
        assert config.channel_name == "releases"

    def test_channel_config_default_prompt_file(self) -> None:
        """Test ChannelConfig with default (empty) prompt file."""
        config = ChannelConfig(
            channel_id="C123456",
            channel_name="releases",
        )

        assert config.prompt_file == ""

    def test_channel_config_all_fields(self) -> None:
        """Test ChannelConfig with all fields including new ones."""
        config = ChannelConfig(
            channel_id="C123456",
            channel_name="releases",
            threshold_score=15.0,
            keyword_weight=12.0,
            mention_weight=10.0,
            reply_weight=7.0,
            reaction_weight=5.0,
            anchor_weight=6.0,
            link_weight=4.0,
            file_weight=8.0,
            bot_penalty=-20.0,
            whitelist_keywords=["release", "deploy"],
            trusted_bots=["B123"],
            prompt_file="config/prompts/custom.yaml",
        )

        assert config.prompt_file == "config/prompts/custom.yaml"
        assert config.keyword_weight == 12.0
        assert config.mention_weight == 10.0
        assert config.bot_penalty == -20.0


class TestTelegramChannelConfig:
    """Test TelegramChannelConfig with username validation."""

    def test_valid_telegram_username(self) -> None:
        """Test TelegramChannelConfig with valid username."""
        config = TelegramChannelConfig(
            username="@productreleases",
            channel_name="product-releases",
            prompt_file="config/prompts/telegram_product.yaml",
        )

        assert config.username == "@productreleases"
        assert config.channel_name == "product-releases"
        assert config.prompt_file == "config/prompts/telegram_product.yaml"

    def test_telegram_username_validation_starts_with_at(self) -> None:
        """Test that username must start with @."""
        with pytest.raises(ValidationError) as exc_info:
            TelegramChannelConfig(
                username="productreleases",  # Missing @
                channel_name="product-releases",
            )

        assert "String should match pattern" in str(exc_info.value)

    def test_telegram_username_validation_length(self) -> None:
        """Test that username has valid length."""
        # Too short
        with pytest.raises(ValidationError) as exc_info:
            TelegramChannelConfig(
                username="@abc",  # Too short
                channel_name="product-releases",
            )

        assert "String should match pattern" in str(exc_info.value)

        # Too long
        with pytest.raises(ValidationError) as exc_info:
            TelegramChannelConfig(
                username="@verylongusernamethatexceeds32characters",
                channel_name="product-releases",
            )

        assert "String should match pattern" in str(exc_info.value)

    def test_telegram_username_validation_format(self) -> None:
        """Test that username follows proper format."""
        # Valid format
        config = TelegramChannelConfig(
            username="@test_channel_123",
            channel_name="test-channel",
        )
        assert config.username == "@test_channel_123"

        # Invalid format (no letters)
        with pytest.raises(ValidationError):
            TelegramChannelConfig(
                username="@123456",  # No letters
                channel_name="test-channel",
            )

    def test_telegram_channel_config_all_fields(self) -> None:
        """Test TelegramChannelConfig with all fields."""
        config = TelegramChannelConfig(
            username="@engineeringteam",
            channel_name="engineering-team",
            threshold_score=20.0,
            keyword_weight=15.0,
            mention_weight=12.0,
            reply_weight=8.0,
            reaction_weight=6.0,
            anchor_weight=7.0,
            link_weight=5.0,
            file_weight=9.0,
            bot_penalty=-25.0,
            whitelist_keywords=["deployment", "release", "incident"],
            prompt_file="config/prompts/telegram_engineering.yaml",
        )

        assert config.username == "@engineeringteam"
        assert config.prompt_file == "config/prompts/telegram_engineering.yaml"
        assert config.keyword_weight == 15.0
        assert config.mention_weight == 12.0
        assert config.bot_penalty == -25.0
        assert len(config.whitelist_keywords) == 3

    def test_telegram_channel_config_defaults(self) -> None:
        """Test TelegramChannelConfig with default values."""
        config = TelegramChannelConfig(
            username="@minimal",
            channel_name="minimal-channel",
        )

        assert config.username == "@minimal"
        assert config.channel_name == "minimal-channel"
        assert config.prompt_file == ""
        assert config.threshold_score == 0.0
        assert config.keyword_weight == 10.0
        assert config.mention_weight == 8.0
        assert config.reply_weight == 5.0
        assert config.reaction_weight == 3.0
        assert config.anchor_weight == 4.0
        assert config.link_weight == 2.0
        assert config.file_weight == 3.0
        assert config.bot_penalty == -15.0
        assert config.whitelist_keywords == []
