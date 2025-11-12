"""Tests for Telegram channel config normalization helper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.domain.models import TelegramChannelConfig
from src.use_cases.ingest_telegram_messages import _normalize_telegram_channel_configs


def test_normalize_mixed_channel_configurations() -> None:
    """Ensure normalization preserves supported inputs and drops invalid ones."""
    config_obj = TelegramChannelConfig(
        username="@typed_channel",
        channel_name="Typed Channel",
        enabled=True,
    )
    mapping_config: dict[str, object] = {
        "channel_id": "@mapping_channel",
        "enabled": False,
        "from_date": "2025-01-01T00:00:00Z",
    }

    logger_stub = SimpleNamespace(warning=MagicMock())

    with patch("src.use_cases.ingest_telegram_messages.logger", new=logger_stub):
        normalized = _normalize_telegram_channel_configs(
            [config_obj, mapping_config, "@legacy_channel", 123]
        )

    assert len(normalized) == 3
    assert normalized[0] is config_obj
    assert normalized[1] is mapping_config

    legacy_config = normalized[2]
    assert isinstance(legacy_config, dict)
    assert legacy_config["channel_id"] == "@legacy_channel"
    assert legacy_config["username"] == "@legacy_channel"
    assert legacy_config["enabled"] is True

    logger_stub.warning.assert_called_once()
    kwargs = logger_stub.warning.call_args.kwargs
    assert kwargs["reason"] == "unsupported_channel_config_type"
    assert kwargs["config_type"] == "int"


def test_normalize_respects_disabled_mapping() -> None:
    """Mappings retain their explicit enabled flags during normalization."""
    mapping_config: dict[str, object] = {
        "channel_id": "@disabled_channel",
        "enabled": False,
    }

    normalized = _normalize_telegram_channel_configs([mapping_config])

    assert len(normalized) == 1
    result = normalized[0]
    assert isinstance(result, dict)
    assert result["channel_id"] == "@disabled_channel"
    assert result["enabled"] is False
