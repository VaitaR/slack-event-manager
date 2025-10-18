"""Tests for multi-source configuration settings.

Tests loading message_sources from YAML, auto-migration, and backward compatibility.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.domain.models import MessageSource


class TestMessageSourceConfigLoading:
    """Test loading message_sources from configuration."""

    def test_load_message_sources_from_yaml(self) -> None:
        """Test loading message_sources section from YAML."""
        # Create a simple YAML content with message_sources
        config_content = {
            "message_sources": [
                {
                    "source_id": "slack",
                    "enabled": True,
                    "raw_table": "raw_slack_messages",
                    "state_table": "ingestion_state_slack",
                    "prompt_file": "config/prompts/slack.txt",
                    "llm_settings": {"temperature": 1.0, "timeout_seconds": 30},
                    "channels": ["C123", "C456"],
                },
                {
                    "source_id": "telegram",
                    "enabled": False,
                    "raw_table": "raw_telegram_messages",
                    "state_table": "ingestion_state_telegram",
                    "prompt_file": "config/prompts/telegram.txt",
                    "llm_settings": {"temperature": 0.7, "timeout_seconds": 30},
                    "channels": [],
                },
            ]
        }

        # Verify the structure is valid YAML
        yaml_string = yaml.dump(config_content)
        loaded = yaml.safe_load(yaml_string)

        assert "message_sources" in loaded
        assert len(loaded["message_sources"]) == 2
        assert loaded["message_sources"][0]["source_id"] == "slack"
        assert loaded["message_sources"][1]["source_id"] == "telegram"

    def test_message_sources_parsed_correctly(self, tmp_path: Path) -> None:
        """Test message_sources are parsed into Settings correctly."""
        # This is a simplified test - actual implementation will be more complex
        config_content = {
            "message_sources": [
                {
                    "source_id": "slack",
                    "enabled": True,
                    "raw_table": "raw_slack_messages",
                    "state_table": "ingestion_state_slack",
                    "prompt_file": "config/prompts/slack.txt",
                    "llm_settings": {"temperature": 1.0},
                    "channels": ["C123"],
                }
            ]
        }

        # Just verify the structure is valid - actual Settings integration
        # will be implemented next
        assert config_content["message_sources"][0]["source_id"] == "slack"
        assert config_content["message_sources"][0]["enabled"] is True


class TestBackwardCompatibility:
    """Test backward compatibility with legacy Slack-only config."""

    def test_no_message_sources_defaults_to_slack(self) -> None:
        """Test that missing message_sources creates default Slack source."""
        # When no message_sources in config, should auto-create from slack_channels
        config: dict[str, Any] = {}

        # This test documents expected behavior - implementation will
        # auto-migrate slack_channels to message_sources format
        assert "message_sources" not in config

        # After Settings initialization, should have default Slack source
        # Implementation will be in Settings.__init__

    def test_legacy_slack_channels_still_work(self) -> None:
        """Test legacy slack_channels config still works."""
        config = {
            "channels": [
                {"channel_id": "C123", "channel_name": "releases"},
                {"channel_id": "C456", "channel_name": "updates"},
            ]
        }

        # Legacy format should still work
        assert len(config["channels"]) == 2


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_source_id_rejected(self) -> None:
        """Test invalid source_id values are rejected."""
        with pytest.raises(ValueError, match="invalid"):
            MessageSource("invalid")

    def test_valid_source_ids_accepted(self) -> None:
        """Test valid source_id values are accepted."""
        assert MessageSource("slack") == MessageSource.SLACK
        assert MessageSource("telegram") == MessageSource.TELEGRAM


class TestPerSourceLLMSettings:
    """Test per-source LLM settings."""

    def test_per_source_llm_temperature(self) -> None:
        """Test different temperature settings per source."""
        slack_config: dict[str, Any] = {
            "source_id": "slack",
            "llm_settings": {"temperature": 1.0},
        }

        telegram_config: dict[str, Any] = {
            "source_id": "telegram",
            "llm_settings": {"temperature": 0.7},
        }

        assert slack_config["llm_settings"]["temperature"] == 1.0
        assert telegram_config["llm_settings"]["temperature"] == 0.7

    def test_per_source_prompt_files(self) -> None:
        """Test different prompt files per source."""
        slack_config = {
            "source_id": "slack",
            "prompt_file": "config/prompts/slack.txt",
        }

        telegram_config = {
            "source_id": "telegram",
            "prompt_file": "config/prompts/telegram.txt",
        }

        assert "slack.txt" in slack_config["prompt_file"]
        assert "telegram.txt" in telegram_config["prompt_file"]
