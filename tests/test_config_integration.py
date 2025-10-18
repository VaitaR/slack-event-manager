"""
Integration tests for multi-source configuration with auto-migration.

Tests the complete flow of loading config, auto-migrating legacy settings,
and accessing message source configurations.
"""

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from src.config.settings import Settings
from src.domain.models import MessageSource


class TestMessageSourceConfigIntegration:
    """Test message_sources configuration integration."""

    def test_settings_loads_message_sources(self, tmp_path: Path) -> None:
        """Test Settings loads message_sources from config."""
        # Create temporary config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Create main.yaml with message_sources
        config_content = {
            "message_sources": [
                {
                    "source_id": "slack",
                    "enabled": True,
                    "bot_token_env": "SLACK_BOT_TOKEN",
                    "raw_table": "raw_slack_messages",
                    "state_table": "ingestion_state_slack",
                    "prompt_file": "config/prompts/slack.yaml",
                    "llm_settings": {"temperature": 1.0, "timeout_seconds": 30},
                    "channels": ["C123", "C456"],
                }
            ]
        }

        main_yaml = config_dir / "main.yaml"
        with open(main_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        # Change to temp directory
        with patch.object(Path, "cwd", return_value=tmp_path):
            os.chdir(tmp_path)
            settings = Settings()

            # Should have message_sources attribute
            assert hasattr(settings, "message_sources")
            assert len(settings.message_sources) == 1
            assert settings.message_sources[0].source_id == MessageSource.SLACK

    def test_auto_migration_from_legacy_channels(self, tmp_path: Path) -> None:
        """Test auto-migration creates message_sources from legacy slack_channels."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Create config with only legacy channels (no message_sources)
        config_content = {
            "channels": [
                {"channel_id": "C123", "channel_name": "releases"},
                {"channel_id": "C456", "channel_name": "updates"},
            ]
        }

        channels_yaml = config_dir / "channels.yaml"
        with open(channels_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        with patch.object(Path, "cwd", return_value=tmp_path):
            os.chdir(tmp_path)
            settings = Settings()

            # Should auto-create message_sources from legacy channels
            assert hasattr(settings, "message_sources")
            assert len(settings.message_sources) >= 1

            # First source should be Slack
            slack_source = settings.message_sources[0]
            assert slack_source.source_id == MessageSource.SLACK
            assert slack_source.enabled is True

            # Should include channel IDs from legacy config
            assert "C123" in slack_source.channels
            assert "C456" in slack_source.channels

    def test_get_source_config_by_id(self, tmp_path: Path) -> None:
        """Test retrieving specific source config by ID."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        config_content = {
            "message_sources": [
                {
                    "source_id": "slack",
                    "enabled": True,
                    "bot_token_env": "SLACK_BOT_TOKEN",
                    "raw_table": "raw_slack_messages",
                    "state_table": "ingestion_state_slack",
                    "channels": ["C123"],
                },
                {
                    "source_id": "telegram",
                    "enabled": False,
                    "bot_token_env": "TELEGRAM_BOT_TOKEN",
                    "raw_table": "raw_telegram_messages",
                    "state_table": "ingestion_state_telegram",
                    "channels": [],
                },
            ]
        }

        main_yaml = config_dir / "main.yaml"
        with open(main_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        with patch.object(Path, "cwd", return_value=tmp_path):
            os.chdir(tmp_path)
            settings = Settings()

            # Should be able to get source config by ID
            slack_config = settings.get_source_config(MessageSource.SLACK)
            assert slack_config is not None
            assert slack_config.source_id == MessageSource.SLACK
            assert slack_config.enabled is True

            telegram_config = settings.get_source_config(MessageSource.TELEGRAM)
            assert telegram_config is not None
            assert telegram_config.source_id == MessageSource.TELEGRAM
            assert telegram_config.enabled is False

    def test_get_enabled_sources(self, tmp_path: Path) -> None:
        """Test filtering only enabled message sources."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        config_content = {
            "message_sources": [
                {
                    "source_id": "slack",
                    "enabled": True,
                    "channels": ["C123"],
                    "raw_table": "raw_slack_messages",
                    "state_table": "ingestion_state_slack",
                },
                {
                    "source_id": "telegram",
                    "enabled": False,
                    "channels": [],
                    "raw_table": "raw_telegram_messages",
                    "state_table": "ingestion_state_telegram",
                },
            ]
        }

        main_yaml = config_dir / "main.yaml"
        with open(main_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        with patch.object(Path, "cwd", return_value=tmp_path):
            os.chdir(tmp_path)
            settings = Settings()

            # Should return only enabled sources
            enabled = settings.get_enabled_sources()
            assert len(enabled) == 1
            assert enabled[0].source_id == MessageSource.SLACK


class TestLLMSettingsPerSource:
    """Test per-source LLM settings."""

    def test_source_specific_temperature(self, tmp_path: Path) -> None:
        """Test each source can have its own temperature setting."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        config_content = {
            "message_sources": [
                {
                    "source_id": "slack",
                    "enabled": True,
                    "raw_table": "raw_slack_messages",
                    "state_table": "ingestion_state_slack",
                    "llm_settings": {"temperature": 1.0},
                },
                {
                    "source_id": "telegram",
                    "enabled": True,
                    "raw_table": "raw_telegram_messages",
                    "state_table": "ingestion_state_telegram",
                    "llm_settings": {"temperature": 0.7},
                },
            ]
        }

        main_yaml = config_dir / "main.yaml"
        with open(main_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        with patch.object(Path, "cwd", return_value=tmp_path):
            os.chdir(tmp_path)
            settings = Settings()

            slack_config = settings.get_source_config(MessageSource.SLACK)
            telegram_config = settings.get_source_config(MessageSource.TELEGRAM)

            assert slack_config.llm_settings["temperature"] == 1.0
            assert telegram_config.llm_settings["temperature"] == 0.7

    def test_source_specific_prompt_file(self, tmp_path: Path) -> None:
        """Test each source can have its own prompt file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        config_content = {
            "message_sources": [
                {
                    "source_id": "slack",
                    "enabled": True,
                    "raw_table": "raw_slack_messages",
                    "state_table": "ingestion_state_slack",
                    "prompt_file": "config/prompts/slack.yaml",
                },
                {
                    "source_id": "telegram",
                    "enabled": True,
                    "raw_table": "raw_telegram_messages",
                    "state_table": "ingestion_state_telegram",
                    "prompt_file": "config/prompts/telegram.yaml",
                },
            ]
        }

        main_yaml = config_dir / "main.yaml"
        with open(main_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        with patch.object(Path, "cwd", return_value=tmp_path):
            os.chdir(tmp_path)
            settings = Settings()

            slack_config = settings.get_source_config(MessageSource.SLACK)
            telegram_config = settings.get_source_config(MessageSource.TELEGRAM)

            assert "slack.yaml" in slack_config.prompt_file
            assert "telegram.yaml" in telegram_config.prompt_file


class TestBackwardCompatibilityIntegration:
    """Test backward compatibility with existing deployments."""

    def test_existing_deployment_without_message_sources(self, tmp_path: Path) -> None:
        """Test existing config without message_sources still works."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Typical existing config (no message_sources)
        config_content = {
            "llm": {"model": "gpt-5-nano", "temperature": 1.0},
            "database": {"path": "data/slack_events.db"},
            "slack": {"digest_channel_id": "D123"},
            "channels": [{"channel_id": "C123", "channel_name": "releases"}],
        }

        main_yaml = config_dir / "main.yaml"
        with open(main_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        channels_yaml = config_dir / "channels.yaml"
        with open(channels_yaml, "w", encoding="utf-8") as f:
            yaml.dump({"channels": config_content["channels"]}, f)

        with patch.object(Path, "cwd", return_value=tmp_path):
            os.chdir(tmp_path)
            settings = Settings()

            # Should still work with all legacy settings
            assert settings.llm_model == "gpt-5-nano"
            assert settings.db_path == "data/slack_events.db"
            assert len(settings.slack_channels) == 1

            # Should auto-create message_sources
            assert hasattr(settings, "message_sources")
            assert len(settings.message_sources) > 0

    def test_no_breaking_changes_for_existing_code(self, tmp_path: Path) -> None:
        """Test existing code using Settings still works."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        config_content = {
            "channels": [{"channel_id": "C123", "channel_name": "releases"}]
        }

        channels_yaml = config_dir / "channels.yaml"
        with open(channels_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        with patch.object(Path, "cwd", return_value=tmp_path):
            os.chdir(tmp_path)
            settings = Settings()

            # All existing attributes should still be accessible
            assert hasattr(settings, "slack_bot_token")
            assert hasattr(settings, "openai_api_key")
            assert hasattr(settings, "slack_channels")
            assert hasattr(settings, "llm_model")
            assert hasattr(settings, "db_path")
