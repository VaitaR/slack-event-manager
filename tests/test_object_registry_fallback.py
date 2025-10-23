"""Tests for ObjectRegistry fallback behavior when registry files are missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services import object_registry as object_registry_module
from src.use_cases import extract_events as extract_events_module


class _DummySettings:
    """Stub settings providing configurable registry path."""

    def __init__(self, registry_path: Path) -> None:
        self.object_registry_path = str(registry_path)


@pytest.fixture
def dummy_settings(tmp_path: Path) -> _DummySettings:
    """Create dummy settings pointing to a non-existent registry file."""

    return _DummySettings(tmp_path / "does_not_exist.yaml")


def test_object_registry_initializes_empty_when_file_missing(
    dummy_settings: _DummySettings,
) -> None:
    """ObjectRegistry should not raise if the YAML file is absent."""

    registry = object_registry_module.ObjectRegistry(
        dummy_settings.object_registry_path
    )

    assert registry.canonicalize_object("any system") is None


def test_get_object_registry_uses_empty_registry_when_file_missing(
    dummy_settings: _DummySettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_get_object_registry should fallback to an empty registry when file is missing."""

    monkeypatch.setattr(extract_events_module, "_object_registry", None)
    monkeypatch.setattr(
        "src.config.settings.get_settings", lambda: dummy_settings, raising=False
    )

    registry = extract_events_module._get_object_registry()

    assert registry.canonicalize_object("unknown system") is None
