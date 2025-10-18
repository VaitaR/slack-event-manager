"""Tests for prompt file loading, caching, and LLM client metadata."""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.llm_client import (
    LLMClient,
    clear_prompt_cache,
    load_prompt_from_file,
)


class TestPromptLoader:
    """Tests for the prompt loading helper."""

    def test_load_yaml_prompt_returns_metadata(self, tmp_path: Path) -> None:
        """YAML prompt files provide versioned system text."""

        clear_prompt_cache()
        prompt_file = tmp_path / "test_prompt.yaml"
        prompt_file.write_text(
            """
version: "20250215.2"
description: "Test prompt"
system: |
  Hello world
  Second line
""".strip()
        )

        entry, cache_hit = load_prompt_from_file(str(prompt_file))

        assert cache_hit is False
        assert entry.version == "20250215.2"
        assert entry.description == "Test prompt"
        assert entry.content == "Hello world\nSecond line"
        assert entry.size_bytes == len(entry.content.encode("utf-8"))

    def test_load_prompt_file_not_found(self) -> None:
        """Missing prompt files raise FileNotFoundError."""

        clear_prompt_cache()
        with pytest.raises(FileNotFoundError):
            load_prompt_from_file("/tmp/does-not-exist.yaml")

    def test_prompt_cache_hit_and_miss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cache should prevent duplicate reads until mtime changes."""

        clear_prompt_cache()
        prompt_file = tmp_path / "cached.yaml"
        prompt_file.write_text(
            """
version: "20250215.3"
system: |
  Cached prompt
""".strip()
        )

        read_calls = {"count": 0}
        original_read_text = Path.read_text

        def _wrapped_read_text(self: Path, *args, **kwargs):
            read_calls["count"] += 1
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _wrapped_read_text)

        first_entry, first_hit = load_prompt_from_file(str(prompt_file))
        second_entry, second_hit = load_prompt_from_file(str(prompt_file))

        assert first_hit is False
        assert second_hit is True
        assert read_calls["count"] == 1
        assert first_entry.content == second_entry.content

        # Modify file to bump mtime and content
        time.sleep(0.05)
        prompt_file.write_text(
            """
version: "20250215.4"
system: |
  Updated prompt text
""".strip()
        )
        os.utime(prompt_file, None)

        third_entry, third_hit = load_prompt_from_file(str(prompt_file))
        assert third_hit is False
        assert read_calls["count"] == 2
        assert third_entry.content.startswith("Updated prompt text")

    def test_checksum_matches_expected(self, tmp_path: Path) -> None:
        """SHA256 checksum should match manual calculation."""

        clear_prompt_cache()
        prompt_file = tmp_path / "checksum.txt"
        prompt_content = "Checksum source text"
        prompt_file.write_text(prompt_content)

        entry, _ = load_prompt_from_file(str(prompt_file))
        expected = hashlib.sha256(prompt_content.encode("utf-8")).hexdigest()
        assert entry.checksum == expected

    def test_legacy_txt_prompt_supported(self) -> None:
        """Legacy .txt prompts remain supported for backward compatibility."""

        clear_prompt_cache()
        project_root = Path(__file__).parent.parent
        prompt_path = project_root / "config" / "prompts" / "slack.txt"

        entry, cache_hit = load_prompt_from_file(str(prompt_path))
        assert cache_hit is False
        assert entry.version is None
        assert "Slack" in entry.content


class TestLLMClientPrompts:
    """Tests for how LLMClient consumes prompts."""

    def test_llm_client_accepts_custom_prompt_template(self) -> None:
        """Custom template strings should be used directly."""

        custom_prompt = "Custom template"
        with patch("src.adapters.llm_client.OpenAI"):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_template=custom_prompt,
            )

        assert client.system_prompt == custom_prompt
        assert client.prompt_version is None
        assert (
            client.prompt_checksum
            == hashlib.sha256(custom_prompt.encode("utf-8")).hexdigest()
        )

    def test_llm_client_loads_yaml_prompt_metadata(self, tmp_path: Path) -> None:
        """Prompt metadata from YAML should populate client fields."""

        clear_prompt_cache()
        prompt_file = tmp_path / "client_prompt.yaml"
        prompt_file.write_text(
            """
version: "20250216.1"
description: "Client prompt"
system: |
  Hello client
""".strip()
        )

        with patch("src.adapters.llm_client.OpenAI"):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_file=str(prompt_file),
            )

        assert client.system_prompt == "Hello client"
        assert client.prompt_version == "20250216.1"
        assert client.prompt_description == "Client prompt"
        assert client.prompt_size_bytes == len(client.system_prompt.encode("utf-8"))

    def test_llm_client_uses_default_yaml_prompt(self) -> None:
        """Default prompt should load from config/prompts/slack.yaml."""

        clear_prompt_cache()
        with patch("src.adapters.llm_client.OpenAI"):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
            )

        assert client.system_prompt
        assert (
            client.prompt_checksum
            == hashlib.sha256(client.system_prompt.encode("utf-8")).hexdigest()
        )

    def test_llm_call_metadata_uses_system_prompt_hash(self, tmp_path: Path) -> None:
        """LLM call metadata should use the system prompt hash."""

        clear_prompt_cache()
        prompt_file = tmp_path / "metadata.yaml"
        prompt_file.write_text(
            """
version: "20250217.1"
system: |
  Metadata prompt text
""".strip()
        )

        fake_openai = MagicMock()
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "is_event": False,
                                "overflow_note": None,
                                "events": [],
                            }
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        fake_openai.chat.completions.create.return_value = fake_response

        with patch("src.adapters.llm_client.OpenAI", return_value=fake_openai):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_file=str(prompt_file),
            )

            response = client.extract_events(
                text="Sample text",
                links=[],
                message_ts_dt=datetime.utcnow(),
                channel_name="releases",
            )

        assert response.is_event is False
        metadata = client.get_call_metadata()
        expected_hash = hashlib.sha256(client.system_prompt.encode("utf-8")).hexdigest()
        assert metadata.prompt_hash == expected_hash

    def test_llm_client_prompt_file_overrides_template(self, tmp_path: Path) -> None:
        """Prompt file should win over provided template."""

        clear_prompt_cache()
        prompt_file = tmp_path / "override.txt"
        prompt_file.write_text("File prompt")

        with patch("src.adapters.llm_client.OpenAI"):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_template="Template prompt",
                prompt_file=str(prompt_file),
            )

        assert client.system_prompt == "File prompt"

    def test_llm_client_handles_empty_prompt_file(self, tmp_path: Path) -> None:
        """Empty prompt files should not raise errors."""

        clear_prompt_cache()
        prompt_file = tmp_path / "empty.txt"
        prompt_file.write_text("")

        with patch("src.adapters.llm_client.OpenAI"):
            client = LLMClient(
                api_key="test-key",
                model="gpt-4o-mini",
                temperature=0.7,
                prompt_file=str(prompt_file),
            )

        assert client.system_prompt == ""
        assert client.prompt_checksum == hashlib.sha256(b"").hexdigest()
