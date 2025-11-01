"""Tests for prompt hashing stability."""

from __future__ import annotations

from datetime import UTC, datetime

from src.use_cases.extract_events import _compute_prompt_hash


class _FakeLLMClient:
    def __init__(self, model: str, version: str | None) -> None:
        self.model = model
        self.prompt_version = version
        version_tag = version or "inline"
        self.system_prompt_hash = f"{model}:{version_tag}:prompt"


def test_prompt_hash_stability_across_runs() -> None:
    """Prompt hash should remain stable for identical inputs."""

    llm_client = _FakeLLMClient(model="gpt-5-nano", version="v1")
    message_ts = datetime(2025, 1, 1, tzinfo=UTC)
    links = ["https://example.com/a", "https://example.com/b"]
    chunk_text = "release announcement"

    first = _compute_prompt_hash(
        llm_client=llm_client,
        chunk_text=chunk_text,
        links=links,
        message_ts_dt=message_ts,
        channel_name="general",
        chunk_index=0,
    )
    second = _compute_prompt_hash(
        llm_client=llm_client,
        chunk_text=chunk_text,
        links=links,
        message_ts_dt=message_ts,
        channel_name="general",
        chunk_index=0,
    )

    assert first == second

    different_chunk = _compute_prompt_hash(
        llm_client=llm_client,
        chunk_text=chunk_text,
        links=links,
        message_ts_dt=message_ts,
        channel_name="general",
        chunk_index=1,
    )
    assert different_chunk != first

    different_model = _compute_prompt_hash(
        llm_client=_FakeLLMClient(model="gpt-4o-mini", version="v1"),
        chunk_text=chunk_text,
        links=links,
        message_ts_dt=message_ts,
        channel_name="general",
        chunk_index=0,
    )
    assert different_model != first

    inline_prompt = _compute_prompt_hash(
        llm_client=_FakeLLMClient(model="gpt-5-nano", version=None),
        chunk_text=chunk_text,
        links=links,
        message_ts_dt=message_ts,
        channel_name="general",
        chunk_index=0,
    )
    assert inline_prompt != ""
