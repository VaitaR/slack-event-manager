"""Tests for configuration loading system."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from src.config.settings import (
    Settings,
    deep_merge,
    load_all_configs,
    load_schema,
    validate_config_section,
)


class StubLogger:
    """Capture structured logging calls."""

    def __init__(self) -> None:
        self.debug_calls: list[tuple[str, dict[str, Any]]] = []
        self.info_calls: list[tuple[str, dict[str, Any]]] = []
        self.warning_calls: list[tuple[str, dict[str, Any]]] = []
        self.error_calls: list[tuple[str, dict[str, Any]]] = []

    def debug(self, event: str, **kwargs: Any) -> None:
        self.debug_calls.append((event, kwargs))

    def info(self, event: str, **kwargs: Any) -> None:
        self.info_calls.append((event, kwargs))

    def warning(self, event: str, **kwargs: Any) -> None:
        self.warning_calls.append((event, kwargs))

    def error(self, event: str, **kwargs: Any) -> None:
        self.error_calls.append((event, kwargs))


def test_deep_merge_simple() -> None:
    """Test deep merge with simple dictionaries."""
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    result = deep_merge(base, override)

    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested() -> None:
    """Test deep merge with nested dictionaries."""
    base = {"llm": {"model": "gpt-5-nano", "temperature": 1.0}}
    override = {"llm": {"timeout_seconds": 180}}
    result = deep_merge(base, override)

    assert result == {
        "llm": {"model": "gpt-5-nano", "temperature": 1.0, "timeout_seconds": 180}
    }


def test_deep_merge_lists_replaced() -> None:
    """Test that lists are replaced, not merged."""
    base = {"items": [1, 2, 3]}
    override = {"items": [4, 5]}
    result = deep_merge(base, override)

    assert result == {"items": [4, 5]}


def test_load_schema_existing(tmp_path: Path) -> None:
    """Test loading existing schema file."""
    # Create temporary schema directory
    schema_dir = tmp_path / "config" / "schemas"
    schema_dir.mkdir(parents=True)

    schema_content = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"test": {"type": "string"}},
    }

    schema_file = schema_dir / "test.schema.json"
    with open(schema_file, "w") as f:
        json.dump(schema_content, f)

    # Change to temp directory
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        schema = load_schema("test")
        assert schema == schema_content
    finally:
        os.chdir(original_cwd)


def test_load_schema_missing() -> None:
    """Test loading non-existent schema returns empty dict."""
    schema = load_schema("nonexistent_schema_xyz")
    assert schema == {}


def test_validate_config_section_valid(tmp_path: Path) -> None:
    """Test validation passes for valid config."""
    # Create temporary schema
    schema_dir = tmp_path / "config" / "schemas"
    schema_dir.mkdir(parents=True)

    schema_content = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    schema_file = schema_dir / "test.schema.json"
    with open(schema_file, "w") as f:
        json.dump(schema_content, f)

    # Change to temp directory
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = {"name": "test"}
        # Should not raise
        validate_config_section(config, "test")
    finally:
        os.chdir(original_cwd)


def test_validate_config_section_invalid(tmp_path: Path) -> None:
    """Test validation fails for invalid config."""
    # Create temporary schema
    schema_dir = tmp_path / "config" / "schemas"
    schema_dir.mkdir(parents=True)

    schema_content = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    schema_file = schema_dir / "test.schema.json"
    with open(schema_file, "w") as f:
        json.dump(schema_content, f)

    # Change to temp directory
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = {"wrong_field": "test"}  # Missing required 'name'

        with pytest.raises(ValueError, match="Config validation failed"):
            validate_config_section(config, "test")
    finally:
        os.chdir(original_cwd)


def test_load_all_configs_empty_directory(tmp_path: Path) -> None:
    """Test loading configs when no config files exist."""
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_all_configs()
        assert config == {}
    finally:
        os.chdir(original_cwd)


def test_load_all_configs_main_only(tmp_path: Path) -> None:
    """Test loading only main config from config/main.yaml."""
    main_config = {"llm": {"model": "gpt-5-nano"}, "database": {"path": "test.db"}}

    # Create config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    config_file = config_dir / "main.yaml"
    with open(config_file, "w") as f:
        yaml.dump(main_config, f)

    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_all_configs()
        assert config == main_config
    finally:
        os.chdir(original_cwd)


def test_load_all_configs_logs_structured_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config loader should emit structured warnings when files fail to load."""

    logger_stub = StubLogger()
    monkeypatch.setattr("src.config.settings.logger", logger_stub)

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "main.yaml").write_text("invalid: [yaml", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    load_all_configs()

    assert logger_stub.warning_calls
    event, payload = logger_stub.warning_calls[0]
    assert event == "config_file_load_failed"
    assert payload["path"].endswith("main.yaml")
    assert "error" in payload


def test_settings_missing_secrets_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings must fail fast when required secrets are absent."""

    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings()


def test_load_all_configs_merge(tmp_path: Path) -> None:
    """Test loading and merging multiple config files."""
    # Create config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create main config
    main_config = {"llm": {"model": "gpt-5-nano", "temperature": 1.0}}

    config_file = config_dir / "main.yaml"
    with open(config_file, "w") as f:
        yaml.dump(main_config, f)

    # Create additional config file
    extra_config = {"llm": {"timeout_seconds": 120}, "database": {"path": "test.db"}}

    extra_file = config_dir / "extra.yaml"
    with open(extra_file, "w") as f:
        yaml.dump(extra_config, f)

    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_all_configs()

        # Should have merged configs
        assert config["llm"]["model"] == "gpt-5-nano"  # From main
        assert config["llm"]["temperature"] == 1.0  # From main
        assert config["llm"]["timeout_seconds"] == 120  # From extra
        assert config["database"]["path"] == "test.db"  # From extra
    finally:
        os.chdir(original_cwd)


def test_load_all_configs_with_channels(tmp_path: Path) -> None:
    """Test loading configs with channels.yaml."""
    # Create config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create main config
    main_config = {"llm": {"model": "gpt-5-nano"}}
    config_file = config_dir / "main.yaml"
    with open(config_file, "w") as f:
        yaml.dump(main_config, f)

    # Create channels config
    channels_config = {
        "channels": [
            {
                "channel_id": "C0000000000",
                "channel_name": "test-channel",
                "threshold_score": 0.0,
            }
        ]
    }

    channels_file = config_dir / "channels.yaml"
    with open(channels_file, "w") as f:
        yaml.dump(channels_config, f)

    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_all_configs()

        assert "channels" in config
        assert len(config["channels"]) == 1
        assert config["channels"][0]["channel_id"] == "C0000000000"
    finally:
        os.chdir(original_cwd)


def test_load_all_configs_with_object_registry(tmp_path: Path) -> None:
    """Test loading configs with object_registry.yaml."""
    # Create config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    registry_config = {
        "objects": {
            "test.system": ["Test System", "Test API"],
            "test.app": ["Test App"],
        }
    }

    registry_file = config_dir / "object_registry.yaml"
    with open(registry_file, "w") as f:
        yaml.dump(registry_config, f)

    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_all_configs()

        assert "objects" in config
        assert "test.system" in config["objects"]
        assert config["objects"]["test.system"] == ["Test System", "Test API"]
    finally:
        os.chdir(original_cwd)


def test_load_all_configs_no_config_directory(tmp_path: Path) -> None:
    """Test loading configs when config directory doesn't exist (env-only scenario)."""
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Ensure config directory doesn't exist
        assert not Path("config").exists()

        # Should not raise UnboundLocalError, should return empty config
        config = load_all_configs()
        assert config == {}
    finally:
        os.chdir(original_cwd)


def test_load_all_configs_empty_config_directory(tmp_path: Path) -> None:
    """Test loading configs when config directory exists but is empty."""
    # Create empty config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = load_all_configs()
        assert config == {}
    finally:
        os.chdir(original_cwd)
