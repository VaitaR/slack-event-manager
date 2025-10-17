"""
Tests for LLM prompt file loading functionality.

Tests that LLMClient can load prompts from files and use them
for event extraction with source-specific configurations.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.adapters.llm_client import LLMClient


class TestLLMPromptLoading:
    """Test prompt file loading in LLMClient."""

    def test_load_prompt_from_file(self, tmp_path: Path) -> None:
        """Test loading prompt from a file."""
        # Create a temporary prompt file
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_content = "Test prompt for {source_id}\nExtract events from messages."
        prompt_file.write_text(prompt_content)

        # Load prompt
        from src.adapters.llm_client import load_prompt_from_file

        loaded_prompt = load_prompt_from_file(str(prompt_file))
        assert loaded_prompt == prompt_content

    def test_load_prompt_file_not_found(self) -> None:
        """Test loading prompt from non-existent file raises error."""
        from src.adapters.llm_client import load_prompt_from_file

        with pytest.raises(FileNotFoundError):
            load_prompt_from_file("nonexistent_prompt.txt")

    def test_llm_client_accepts_custom_prompt(self) -> None:
        """Test LLMClient accepts custom prompt template."""
        custom_prompt = "Custom prompt for testing"

        with patch("openai.OpenAI"):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_template=custom_prompt,
            )

            assert client.system_prompt == custom_prompt

    def test_llm_client_uses_default_prompt_if_none_provided(self) -> None:
        """Test LLMClient uses default prompt if custom one not provided."""
        with patch("openai.OpenAI"):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
            )

            # Should have a non-empty default prompt
            assert client.system_prompt
            assert len(client.system_prompt) > 0

    def test_llm_client_with_prompt_file_path(self, tmp_path: Path) -> None:
        """Test LLMClient loading prompt from file path."""
        # Create a temporary prompt file
        prompt_file = tmp_path / "slack_prompt.txt"
        prompt_content = "Slack-specific extraction prompt"
        prompt_file.write_text(prompt_content)

        with patch("openai.OpenAI"):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_file=str(prompt_file),
            )

            assert client.system_prompt == prompt_content

    def test_slack_prompt_file_exists(self) -> None:
        """Test that slack.txt prompt file exists."""
        # Use absolute path from project root
        project_root = Path(__file__).parent.parent
        prompt_path = project_root / "config" / "prompts" / "slack.txt"
        assert prompt_path.exists(), f"Slack prompt file should exist at {prompt_path}"

        content = prompt_path.read_text()
        assert len(content) > 0, "Slack prompt should not be empty"
        assert "Slack" in content, "Slack prompt should mention Slack"

    def test_telegram_prompt_file_exists(self) -> None:
        """Test that telegram.txt prompt file exists."""
        # Use absolute path from project root
        project_root = Path(__file__).parent.parent
        prompt_path = project_root / "config" / "prompts" / "telegram.txt"
        assert (
            prompt_path.exists()
        ), f"Telegram prompt file should exist at {prompt_path}"

        content = prompt_path.read_text()
        assert len(content) > 0, "Telegram prompt should not be empty"
        assert "Telegram" in content, "Telegram prompt should mention Telegram"

    def test_prompt_files_contain_required_sections(self) -> None:
        """Test that prompt files contain all required sections."""
        required_keywords = [
            "event extraction",
            "category",
            "product",
            "risk",
            "JSON",
        ]

        # Use absolute path from project root
        project_root = Path(__file__).parent.parent

        for source in ["slack", "telegram"]:
            prompt_path = project_root / "config" / "prompts" / f"{source}.txt"
            content = prompt_path.read_text().lower()

            for keyword in required_keywords:
                assert (
                    keyword.lower() in content
                ), f"{source} prompt should contain '{keyword}'"

    def test_llm_client_prompt_file_overrides_template(self, tmp_path: Path) -> None:
        """Test that prompt_file parameter takes precedence over prompt_template."""
        prompt_file = tmp_path / "test.txt"
        prompt_file.write_text("File prompt")

        with patch("openai.OpenAI"):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_template="Template prompt",
                prompt_file=str(prompt_file),
            )

            # File should take precedence
            assert client.system_prompt == "File prompt"

    def test_llm_client_handles_empty_prompt_file(self, tmp_path: Path) -> None:
        """Test LLMClient handles empty prompt file gracefully."""
        prompt_file = tmp_path / "empty.txt"
        prompt_file.write_text("")

        with patch("openai.OpenAI"):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_file=str(prompt_file),
            )

            # Should fall back to default or handle empty
            assert client.system_prompt is not None
