"""Tests covering privacy-aware LLM logging."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

import pytest

from src.adapters.llm_client import LLMClient


class StubLogger:
    """Collects log calls for assertions."""

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


class StubOpenAIClient:
    """Minimal stub for OpenAI client used in tests."""

    def __init__(self) -> None:
        response_content = """
        {
            "is_event": false,
            "events": []
        }
        """.strip()
        self._response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=response_content))
            ],
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7),
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_response)
        )

    def _create_response(self, **_: Any) -> Any:
        return self._response


@pytest.fixture(autouse=True)
def allow_verbose_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow verbose logging for the duration of the test."""

    monkeypatch.setenv("LLM_ALLOW_VERBOSE_LOGS", "1")


def test_verbose_logging_redacts_sensitive_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verbose logging must never expose raw prompts or responses."""

    stub_logger = StubLogger()
    monkeypatch.setattr("src.adapters.llm_client.logger", stub_logger)
    monkeypatch.setattr(
        "src.adapters.llm_client.OpenAI",
        lambda api_key, timeout: StubOpenAIClient(),
    )

    client = LLMClient(
        api_key="test",
        model="gpt-4o-mini",
        temperature=0.7,
        timeout=10,
        verbose=True,
        prompt_template="system",
    )

    client.extract_events(
        text="This is a secret message.",
        links=[],
        message_ts_dt=dt.datetime.now(tz=dt.UTC),
        channel_name="releases",
    )

    request_logs = [
        kwargs
        for event, kwargs in stub_logger.debug_calls
        if event == "llm_request_verbose"
    ]
    assert request_logs, "Verbose request log should be emitted"
    request_payload = request_logs[0]
    assert request_payload["prompt_redacted"] is True
    assert "user_prompt_preview" not in request_payload
    assert "prompt_checksum" in request_payload

    response_logs = [
        kwargs
        for event, kwargs in stub_logger.debug_calls
        if event == "llm_response_verbose"
    ]
    assert response_logs, "Verbose response log should be emitted"
    response_payload = response_logs[0]
    assert response_payload["response_redacted"] is True
    assert "response_preview" not in response_payload
    assert "response_checksum" in response_payload
