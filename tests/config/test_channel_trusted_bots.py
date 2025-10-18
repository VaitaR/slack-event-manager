"""Regression tests for trusted bot configuration handling."""

from collections.abc import Iterable
from pathlib import Path

import pytest
import yaml

from src.config import settings as settings_module
from src.domain.models import ChannelConfig


def _write_channels_config(
    tmp_path: Path, channels: Iterable[dict[str, object]]
) -> None:
    """Write a minimal channels.yaml file for Settings to consume."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    channels_file = config_dir / "channels.yaml"
    with channels_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"channels": list(channels)}, f, sort_keys=False)


def _create_settings() -> tuple[settings_module.Settings, object]:
    """Instantiate Settings using isolated config files and return previous state."""
    original_settings = settings_module._settings
    settings_module._settings = None
    settings = settings_module.Settings()
    settings_module._settings = settings
    return settings, original_settings


def _restore_settings(previous_settings: object) -> None:
    """Restore the module-level settings singleton to its prior value."""
    settings_module._settings = previous_settings  # type: ignore[assignment]


def test_trusted_bots_default_to_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Channels without trusted_bots should produce an empty list."""
    _write_channels_config(
        tmp_path,
        [
            {
                "channel_id": "CDEFAULT0001",
                "channel_name": "default-channel",
                "threshold_score": 0.0,
            }
        ],
    )

    monkeypatch.chdir(tmp_path)
    settings, previous = _create_settings()
    try:
        channel_config = settings.get_channel_config("CDEFAULT0001")

        assert isinstance(channel_config, ChannelConfig)
        assert channel_config.trusted_bots == []
        assert settings.slack_channels[0].trusted_bots == []
    finally:
        _restore_settings(previous)


def test_trusted_bots_loaded_from_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Channels configured with trusted_bots should retain the provided values."""
    trusted_ids = ["B12345", "B99999"]
    _write_channels_config(
        tmp_path,
        [
            {
                "channel_id": "CTRUSTED0001",
                "channel_name": "trusted-channel",
                "threshold_score": 0.0,
                "trusted_bots": trusted_ids,
            }
        ],
    )

    monkeypatch.chdir(tmp_path)
    settings, previous = _create_settings()
    try:
        channel_config = settings.get_channel_config("CTRUSTED0001")

        assert isinstance(channel_config, ChannelConfig)
        assert channel_config.trusted_bots == trusted_ids
        assert settings.slack_channels[0].trusted_bots == trusted_ids
    finally:
        _restore_settings(previous)
