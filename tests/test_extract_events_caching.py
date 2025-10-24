"""Tests for LLM response caching within the extraction use case."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.config.settings import Settings
from src.domain.models import (
    EventCandidate,
    EventCategory,
    LLMCallMetadata,
    LLMEvent,
    LLMResponse,
    MessageSource,
    ScoringFeatures,
)
from src.use_cases.extract_events import extract_events_use_case


@pytest.fixture
def extraction_settings() -> MagicMock:
    """Provide minimal settings required by the extraction use case."""

    settings = MagicMock(spec=Settings)
    settings.llm_daily_budget_usd = 100.0
    settings.llm_max_events_per_msg = 2
    settings.get_scoring_config.return_value = None
    settings.dedup_date_window_hours = 48
    settings.dedup_title_similarity = 0.8
    return settings


def _make_candidate(source: MessageSource = MessageSource.SLACK) -> EventCandidate:
    return EventCandidate(
        message_id="msg-1",
        channel="general",
        ts_dt=datetime.now(tz=UTC),
        text_norm="release shipped",
        links_norm=["https://example.com/changelog"],
        anchors=["ABC-123"],
        score=0.9,
        features=ScoringFeatures(),
        source_id=source,
    )


def _make_llm_event() -> LLMEvent:
    return LLMEvent(
        action="launch",
        object_name_raw="Widget",
        qualifiers=["major"],
        stroke="launched",
        anchor="ABC-123",
        category=EventCategory.PRODUCT,
        status="confirmed",
        change_type="launch",
        environment="prod",
        severity=None,
        planned_start=datetime.now(tz=UTC).isoformat(),
        planned_end=None,
        actual_start=None,
        actual_end=None,
        time_source="explicit",
        time_confidence=0.8,
        summary="Widget launched",
        why_it_matters="Customers benefit",
        links=["https://example.com/changelog"],
        anchors=["ABC-123"],
        impact_area=["platform"],
        impact_type=["positive"],
        confidence=0.9,
    )


def test_extract_events_use_case_uses_cached_response(
    extraction_settings: MagicMock,
) -> None:
    """Cache hits should bypass the LLM client and record cached metadata."""

    candidate = _make_candidate(MessageSource.TELEGRAM)
    llm_response = LLMResponse(is_event=True, events=[_make_llm_event()])

    repository = MagicMock()
    repository.get_candidates_for_extraction.return_value = [candidate]
    repository.get_cached_llm_response.return_value = llm_response.model_dump_json()
    repository.update_candidate_status.return_value = None

    llm_client = MagicMock()
    llm_client.model = "gpt-5-nano"
    llm_client.system_prompt_hash = "prompt-hash"

    result = extract_events_use_case(
        llm_client=llm_client,
        repository=repository,
        settings=extraction_settings,
        source_id=MessageSource.TELEGRAM,
        batch_size=5,
        check_budget=False,
    )

    assert result.events_extracted == 1
    assert result.cache_hits == 1
    assert result.llm_calls == 0
    assert llm_client.extract_events_with_retry.called is False

    repository.save_llm_response.assert_not_called()
    repository.save_events.assert_called_once()

    saved_events = repository.save_events.call_args[0][0]
    assert all(event.source_id == MessageSource.TELEGRAM for event in saved_events)

    metadata = repository.save_llm_call.call_args[0][0]
    assert isinstance(metadata, LLMCallMetadata)
    assert metadata.cached is True
    assert metadata.cost_usd == 0
    assert metadata.tokens_in == 0
    assert metadata.tokens_out == 0
    assert metadata.prompt_hash == repository.get_cached_llm_response.call_args[0][0]


def test_extract_events_use_case_persists_llm_response(
    extraction_settings: MagicMock,
) -> None:
    """LLM responses should be persisted and limited per settings."""

    candidate = _make_candidate()
    llm_response = LLMResponse(
        is_event=True,
        events=[_make_llm_event(), _make_llm_event(), _make_llm_event()],
    )

    repository = MagicMock()
    repository.get_candidates_for_extraction.return_value = [candidate]
    repository.get_cached_llm_response.return_value = None
    repository.update_candidate_status.return_value = None

    llm_client = MagicMock()
    llm_client.model = "gpt-5-nano"
    llm_client.system_prompt_hash = "prompt-hash"
    llm_client.extract_events_with_retry.return_value = llm_response
    llm_client.get_call_metadata.return_value = LLMCallMetadata(
        message_id="",
        prompt_hash="prompt-hash",
        model="gpt-5-nano",
        tokens_in=100,
        tokens_out=50,
        cost_usd=1.23,
        latency_ms=500,
        cached=False,
    )

    result = extract_events_use_case(
        llm_client=llm_client,
        repository=repository,
        settings=extraction_settings,
        source_id=MessageSource.SLACK,
        batch_size=5,
        check_budget=False,
    )

    assert result.events_extracted == extraction_settings.llm_max_events_per_msg
    assert result.llm_calls == 1
    assert result.cache_hits == 0
    assert pytest.approx(result.total_cost_usd, rel=1e-6) == 1.23

    llm_client.extract_events_with_retry.assert_called_once()
    repository.save_llm_response.assert_called_once()

    saved_events = repository.save_events.call_args[0][0]
    assert len(saved_events) == extraction_settings.llm_max_events_per_msg
    assert all(event.source_id == MessageSource.SLACK for event in saved_events)

    metadata = repository.save_llm_call.call_args[0][0]
    assert isinstance(metadata, LLMCallMetadata)
    assert metadata.cached is False
    assert metadata.prompt_hash == repository.save_llm_response.call_args[0][0]
    assert metadata.cost_usd == 1.23
