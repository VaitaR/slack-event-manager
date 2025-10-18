"""Tests for prompt loading and caching in the LLM client."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.adapters import llm_client
from src.adapters.llm_client import LLMClient, PromptFileData, load_prompt_from_file


@pytest.fixture(autouse=True)
def clear_prompt_cache() -> None:
    """Ensure prompt cache is cleared between tests."""

    llm_client._PROMPT_CACHE.clear()


class TestPromptLoading:
    """Validate prompt loading from YAML and legacy text files."""

    def test_loads_yaml_with_version(self) -> None:
        """The loader returns versioned prompts for YAML files."""

        prompt = load_prompt_from_file("config/prompts/slack.yaml")
        assert isinstance(prompt, PromptFileData)
        assert prompt.version is not None
        assert "Slack" in prompt.content
        assert (
            prompt.checksum
            == hashlib.sha256(prompt.content.encode("utf-8")).hexdigest()
        )

    def test_loads_telegram_yaml(self) -> None:
        """Telegram YAML prompt is parsed correctly."""

        prompt = load_prompt_from_file("config/prompts/telegram.yaml")
        assert prompt.version is not None
        assert "Telegram" in prompt.content

    def test_supports_legacy_text_prompts(self, tmp_path: Path) -> None:
        """Legacy .txt prompts still load without a version."""

        prompt_file = tmp_path / "legacy.txt"
        prompt_file.write_text("Legacy prompt content")

        prompt = load_prompt_from_file(str(prompt_file))
        assert prompt.version is None
        assert prompt.content == "Legacy prompt content"

    def test_cache_hit_without_mtime_change(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The loader reuses cached prompts when mtime is unchanged."""

        prompt_file = tmp_path / "cached.yaml"
        prompt_file.write_text(
            "\n".join(
                [
                    'version: "20250101.1"',
                    'description: "test"',
                    "system: |",
                    "  cached prompt",
                ]
            )
        )

        read_count = 0
        original_read_text = Path.read_text

        def tracked_read_text(self: Path, *args: object, **kwargs: object) -> str:
            nonlocal read_count
            if self == prompt_file:
                read_count += 1
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", tracked_read_text)

        load_prompt_from_file(str(prompt_file))
        load_prompt_from_file(str(prompt_file))

        assert read_count == 1

    def test_cache_miss_on_mtime_change(self, tmp_path: Path) -> None:
        """Cache invalidates when prompt mtime changes."""

        prompt_file = tmp_path / "changing.yaml"
        prompt_file.write_text(
            "\n".join(
                [
                    'version: "20250101.1"',
                    'description: "test"',
                    "system: |",
                    "  first version",
                ]
            )
        )

        first = load_prompt_from_file(str(prompt_file))

        previous_stat = prompt_file.stat()

        prompt_file.write_text(
            "\n".join(
                [
                    'version: "20250101.2"',
                    'description: "test"',
                    "system: |",
                    "  second version",
                ]
            )
        )
        os.utime(prompt_file, (previous_stat.st_atime, previous_stat.st_mtime + 1))

        second = load_prompt_from_file(str(prompt_file))

        assert first.content != second.content
        assert first.checksum != second.checksum
        assert second.version == "20250101.2"

    def test_checksum_consistency(self, tmp_path: Path) -> None:
        """Checksum is derived solely from the system prompt text."""

        prompt_file = tmp_path / "checksum.yaml"
        prompt_file.write_text(
            "\n".join(
                [
                    'version: "20250101.1"',
                    'description: "test"',
                    "system: |",
                    "  checksum body",
                ]
            )
        )

        prompt = load_prompt_from_file(str(prompt_file))
        expected = hashlib.sha256(b"checksum body").hexdigest()
        assert prompt.checksum == expected


class TestLLMClientPromptSelection:
    """Ensure LLMClient selects prompts with correct precedence."""

    def test_default_prompt_uses_slack_yaml(self) -> None:
        """Default prompt loads from bundled Slack YAML."""

        with patch("src.adapters.llm_client.OpenAI"):
            client = LLMClient(api_key="test", model="gpt-4o-mini", temperature=0.7)

        bundled = load_prompt_from_file("config/prompts/slack.yaml")
        assert client.system_prompt == bundled.content
        assert client.prompt_version == bundled.version
        assert (
            client._system_prompt_hash
            == hashlib.sha256(client.system_prompt.encode("utf-8")).hexdigest()
        )

    def test_prompt_file_overrides_template(self, tmp_path: Path) -> None:
        """prompt_file parameter takes precedence over template."""

        prompt_file = tmp_path / "custom.yaml"
        prompt_file.write_text(
            "\n".join(
                [
                    'version: "20250101.1"',
                    'description: "test"',
                    "system: |",
                    "  file prompt",
                ]
            )
        )

        with patch("src.adapters.llm_client.OpenAI"):
            client = LLMClient(
                api_key="test",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_template="Template prompt",
                prompt_file=str(prompt_file),
            )

        assert client.system_prompt.strip() == "file prompt"
        assert client.prompt_version == "20250101.1"

    def test_prompt_template_has_no_version(self) -> None:
        """Inline prompt templates have no version metadata."""

        with patch("src.adapters.llm_client.OpenAI"):
            client = LLMClient(
                api_key="test",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_template="Inline prompt",
            )

        assert client.prompt_version is None

    def test_metadata_prompt_hash_matches_system_prompt(self, tmp_path: Path) -> None:
        """LLM call metadata stores the system prompt hash."""

        prompt_file = tmp_path / "meta.yaml"
        prompt_file.write_text(
            "\n".join(
                [
                    'version: "20250101.1"',
                    'description: "test"',
                    "system: |",
                    "  metadata prompt",
                ]
            )
        )

        with patch("src.adapters.llm_client.OpenAI") as mock_openai:
            client_instance = MagicMock()
            response_mock = MagicMock()
            response_mock.choices = [
                MagicMock(
                    message=MagicMock(content='{ "is_event": false, "events": [] }')
                )
            ]
            response_mock.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            client_instance.chat.completions.create.return_value = response_mock
            mock_openai.return_value = client_instance

            client = LLMClient(
                api_key="test",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_file=str(prompt_file),
            )

            response = client.extract_events(
                text="message",
                links=[],
                message_ts_dt=datetime.utcnow(),
                channel_name="",
            )

            assert response.is_event is False
            metadata = client.get_call_metadata()
            expected_hash = hashlib.sha256(
                client.system_prompt.encode("utf-8")
            ).hexdigest()
            assert metadata.prompt_hash == expected_hash
