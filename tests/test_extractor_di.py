from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.adapters.llm_client import LLMClient
from src.config.settings import Settings
from src.domain.models import (
    EventCategory,
    LLMCallMetadata,
    LLMEvent,
    LLMResponse,
    MessageSource,
)
from src.domain.protocols import RepositoryProtocol
from src.use_cases.extract_events import extract_events_use_case


class FakeObjectRegistry:
    """Minimal registry double capturing canonicalization requests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def canonicalize_object(self, raw_name: str) -> str | None:
        self.calls.append(raw_name)
        return "registry.object"


class FakeImportanceScorer:
    """Importance scorer double capturing invocation details."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, int, int, bool]] = []

    def calculate_importance(
        self,
        event: object,
        *,
        llm_score: float | None,
        reaction_count: int,
        mention_count: int,
        is_duplicate: bool,
    ) -> SimpleNamespace:
        self.calls.append((event, reaction_count, mention_count, is_duplicate))
        return SimpleNamespace(final_score=99)


def test_extract_events_use_case_uses_injected_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure DI path avoids global settings and uses injected collaborators."""

    def _fail() -> Settings:
        raise AssertionError("get_settings should not be called during extraction")

    monkeypatch.setattr("src.config.settings.get_settings", _fail)

    llm_client = MagicMock(spec=LLMClient)
    llm_client.model = "test-model"
    llm_client.system_prompt_hash = "prompt-hash"
    llm_client.prompt_token_budget = 3000
    llm_client.prompt_version = "v1"

    llm_event = LLMEvent(
        action="launch",
        object_name_raw="Test Object",
        qualifiers=["major"],
        stroke="launched",
        anchor="anchor",
        category=EventCategory.PRODUCT,
        status="confirmed",
        change_type="launch",
        environment="prod",
        severity=None,
        planned_start="2025-10-24T10:00:00+00:00",
        planned_end=None,
        actual_start=None,
        actual_end=None,
        time_source="explicit",
        time_confidence=0.8,
        summary="Summary",
        why_it_matters="Why it matters",
        links=["https://example.com"],
        anchors=["anchor"],
        impact_area=["ops"],
        impact_type=["positive"],
        confidence=0.9,
    )
    llm_response = LLMResponse(is_event=True, events=[llm_event])
    call_metadata = LLMCallMetadata(
        message_id="msg-1",
        prompt_hash="hash",
        model="test-model",
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.01,
        latency_ms=100,
        cached=False,
    )
    llm_client.extract_events_with_retry.return_value = llm_response
    llm_client.get_call_metadata.return_value = call_metadata

    repository = MagicMock(spec=RepositoryProtocol)
    repository.get_candidates_for_extraction.return_value = [
        SimpleNamespace(
            message_id="msg-1",
            channel="#general",
            ts_dt=datetime.now(UTC),
            text_norm="Test message",
            links_norm=["https://example.com"],
            anchors=["anchor"],
            features=SimpleNamespace(reaction_count=2, has_mention=False),
            source_id=MessageSource.SLACK,
        )
    ]
    repository.get_daily_llm_cost.return_value = 0.0
    repository.get_cached_llm_response.return_value = None
    repository.save_llm_response.return_value = None
    repository.save_llm_call.return_value = None
    repository.save_events.return_value = None
    repository.update_candidate_status.return_value = None

    settings = MagicMock(spec=Settings)
    settings.llm_daily_budget_usd = 100.0
    settings.llm_model = "test-model"
    settings.llm_max_events_per_msg = 5
    settings.get_scoring_config.return_value = None
    settings.dedup_date_window_hours = 48
    settings.dedup_title_similarity = 0.8

    fake_registry = FakeObjectRegistry()
    fake_importance_scorer = FakeImportanceScorer()

    result = extract_events_use_case(
        llm_client=llm_client,
        repository=repository,
        settings=settings,
        source_id=MessageSource.SLACK,
        batch_size=5,
        check_budget=False,
        object_registry=fake_registry,
        importance_scorer=fake_importance_scorer,
    )

    assert result.events_extracted == 1
    assert fake_registry.calls == ["Test Object"]
    assert fake_importance_scorer.calls, "Importance scorer should be invoked"
