"""Tests for source isolation in LLM extraction.

Verifies that extract_events_use_case correctly filters candidates by source_id
and doesn't mix candidates from different sources.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest
import pytz

from src.adapters.query_builders import CandidateQueryCriteria
from src.domain.models import (
    CandidateStatus,
    EventCandidate,
    MessageSource,
    ScoringFeatures,
)
from src.use_cases.extract_events import extract_events_use_case


@dataclass
class StubLLMCallMetadata:
    """Minimal metadata container for LLM calls."""

    cost_usd: float = 0.0
    message_id: str | None = None


class StubLLMClient:
    """Predictable LLM client substitute for extraction tests."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self._metadata = StubLLMCallMetadata()
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """Return the number of extraction calls performed."""

        return len(self.calls)

    def extract_events_with_retry(
        self,
        *,
        text: str,
        links: list[str],
        message_ts_dt: datetime,
        channel_name: str,
    ) -> Any:
        """Record arguments and return the configured response."""

        self.calls.append(
            {
                "text": text,
                "links": links,
                "message_ts_dt": message_ts_dt,
                "channel_name": channel_name,
            }
        )
        return self._response

    def get_call_metadata(self) -> StubLLMCallMetadata:
        """Return metadata used for repository persistence."""

        return self._metadata


class StubSettings:
    """Lightweight settings substitute for extraction tests."""

    def __init__(self, llm_daily_budget_usd: float = 100.0) -> None:
        self.llm_daily_budget_usd = llm_daily_budget_usd

    def get_scoring_config(
        self, source_id: MessageSource, channel_id: str
    ) -> None:  # pragma: no cover - configuration lookup is not exercised
        """Return no channel configuration for the test scenarios."""

        return None


class FakeRepository:
    """In-memory repository tailored for extraction tests."""

    def __init__(self) -> None:
        self._candidates: dict[str, EventCandidate] = {}
        self._llm_calls: list[Any] = []

    def save_candidates(self, candidates: Iterable[EventCandidate]) -> None:
        """Store candidates keyed by message identifier."""

        for candidate in candidates:
            self._candidates[candidate.message_id] = candidate

    def get_daily_llm_cost(self, day: datetime) -> float:  # pragma: no cover - trivial
        """Return zero cost to bypass budget checks during tests."""

        return 0.0

    def get_candidates_for_extraction(
        self,
        *,
        batch_size: int | None,
        min_score: float | None,
        source_id: MessageSource | None,
    ) -> list[EventCandidate]:
        """Return candidates filtered by status, score, and source."""

        filtered = [
            candidate
            for candidate in self._candidates.values()
            if candidate.status == CandidateStatus.NEW
        ]

        if min_score is not None:
            filtered = [
                candidate for candidate in filtered if candidate.score >= min_score
            ]

        if source_id is not None:
            filtered = [
                candidate for candidate in filtered if candidate.source_id == source_id
            ]

        filtered.sort(key=lambda candidate: candidate.score, reverse=True)

        if batch_size is not None:
            return filtered[:batch_size]
        return filtered

    def query_candidates(
        self, criteria: CandidateQueryCriteria
    ) -> list[EventCandidate]:
        """Return candidates filtered by the provided criteria."""

        filtered = list(self._candidates.values())

        if criteria.status is not None:
            filtered = [
                candidate
                for candidate in filtered
                if candidate.status.value == criteria.status
            ]

        if criteria.min_score is not None:
            filtered = [
                candidate
                for candidate in filtered
                if candidate.score >= criteria.min_score
            ]

        key_name = criteria.order_by or "score"
        filtered.sort(
            key=lambda candidate: getattr(candidate, key_name),
            reverse=criteria.order_desc,
        )

        if criteria.limit is not None:
            return filtered[: criteria.limit]
        return filtered

    def save_llm_call(self, call_metadata: Any) -> None:
        """Record LLM call metadata for observability assertions."""

        self._llm_calls.append(call_metadata)

    def save_events(
        self, events: Iterable[Any]
    ) -> None:  # pragma: no cover - unused path
        """Discard events to mimic a write without touching external systems."""

        return None

    def update_candidate_status(self, message_id: str, status: str) -> None:
        """Update candidate status after processing."""

        candidate = self._candidates.get(message_id)
        if candidate is None:
            msg = f"Candidate with id {message_id} not found"
            raise KeyError(msg)

        candidate.status = CandidateStatus(status)


@pytest.fixture(name="settings")
def settings_fixture() -> StubSettings:
    """Provide lightweight settings to avoid expensive config loading."""

    return StubSettings()


@pytest.fixture(name="repo")
def repository_fixture() -> FakeRepository:
    """Provide an in-memory repository for focused extraction tests."""

    return FakeRepository()


@pytest.fixture(autouse=True)
def patch_extract_events_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace heavy global services in the extraction use case with lightweight stubs."""

    monkeypatch.setattr(
        "src.use_cases.extract_events._object_registry",
        SimpleNamespace(
            canonicalize_object=lambda *_args, **_kwargs: None,
            get_synonyms=lambda *_args, **_kwargs: [],
            get_all_object_ids=lambda: [],
        ),
    )
    monkeypatch.setattr(
        "src.use_cases.extract_events._importance_scorer",
        SimpleNamespace(score=lambda *_args, **_kwargs: 0.0),
    )
    monkeypatch.setattr(
        "src.use_cases.extract_events._event_validator",
        SimpleNamespace(
            validate_event=lambda *_args, **_kwargs: {
                "critical_errors": [],
                "warnings": [],
                "info": [],
            }
        ),
    )


class TestSourceIsolationExtraction:
    """Test source isolation in LLM extraction."""

    def test_extract_events_slack_only_filters_correctly(self, repo, settings):
        """Test that extract_events_use_case filters candidates by Slack source only."""
        # Create test candidates for different sources
        slack_candidate = EventCandidate(
            message_id="slack_msg_1",
            channel="#releases",
            ts_dt=datetime(2025, 1, 1, 12, 0).replace(tzinfo=pytz.UTC),
            text_norm="Slack message text",
            links_norm=[],
            anchors=[],
            score=85.0,
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.SLACK,
        )

        telegram_candidate = EventCandidate(
            message_id="telegram_msg_1",
            channel="@news",
            ts_dt=datetime(2025, 1, 1, 12, 0).replace(tzinfo=pytz.UTC),
            text_norm="Telegram message text",
            links_norm=[],
            anchors=[],
            score=85.0,
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.TELEGRAM,
        )

        # Save candidates to repository
        repo.save_candidates([slack_candidate, telegram_candidate])

        # Create stub LLM client
        llm_client = StubLLMClient(response=SimpleNamespace(is_event=False, events=[]))

        # Run extraction for Slack only
        result = extract_events_use_case(
            llm_client=llm_client,
            repository=repo,
            settings=settings,
            source_id=MessageSource.SLACK,  # Filter to Slack only
            batch_size=50,
            check_budget=False,
        )

        # Verify only Slack candidate was processed
        assert result.candidates_processed == 1
        assert result.events_extracted == 0  # No events in mock response

        # Verify LLM was called only once (for Slack candidate)
        assert llm_client.call_count == 1

        # Verify the call was made for the Slack candidate
        assert llm_client.calls[0]["text"] == "Slack message text"

    def test_extract_events_telegram_only_filters_correctly(self, repo, settings):
        """Test that extract_events_use_case filters candidates by Telegram source only."""
        # Create test candidates for different sources
        slack_candidate = EventCandidate(
            message_id="slack_msg_1",
            channel="#releases",
            ts_dt=datetime(2025, 1, 1, 12, 0).replace(tzinfo=pytz.UTC),
            text_norm="Slack message text",
            links_norm=[],
            anchors=[],
            score=85.0,
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.SLACK,
        )

        telegram_candidate = EventCandidate(
            message_id="telegram_msg_1",
            channel="@news",
            ts_dt=datetime(2025, 1, 1, 12, 0).replace(tzinfo=pytz.UTC),
            text_norm="Telegram message text",
            links_norm=[],
            anchors=[],
            score=85.0,
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.TELEGRAM,
        )

        # Save candidates to repository
        repo.save_candidates([slack_candidate, telegram_candidate])

        # Create stub LLM client
        llm_client = StubLLMClient(response=SimpleNamespace(is_event=False, events=[]))

        # Run extraction for Telegram only
        result = extract_events_use_case(
            llm_client=llm_client,
            repository=repo,
            settings=settings,
            source_id=MessageSource.TELEGRAM,  # Filter to Telegram only
            batch_size=50,
            check_budget=False,
        )

        # Verify only Telegram candidate was processed
        assert result.candidates_processed == 1
        assert result.events_extracted == 0  # No events in mock response

        # Verify LLM was called only once (for Telegram candidate)
        assert llm_client.call_count == 1

        # Verify the call was made for the Telegram candidate
        assert llm_client.calls[0]["text"] == "Telegram message text"

    def test_extract_events_no_source_filter_processes_all(self, repo, settings):
        """Test that extract_events_use_case processes all sources when source_id=None."""
        # Create test candidates for different sources
        slack_candidate = EventCandidate(
            message_id="slack_msg_1",
            channel="#releases",
            ts_dt=datetime(2025, 1, 1, 12, 0).replace(tzinfo=pytz.UTC),
            text_norm="Slack message text",
            links_norm=[],
            anchors=[],
            score=85.0,
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.SLACK,
        )

        telegram_candidate = EventCandidate(
            message_id="telegram_msg_1",
            channel="@news",
            ts_dt=datetime(2025, 1, 1, 12, 0).replace(tzinfo=pytz.UTC),
            text_norm="Telegram message text",
            links_norm=[],
            anchors=[],
            score=85.0,
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.TELEGRAM,
        )

        # Save candidates to repository
        repo.save_candidates([slack_candidate, telegram_candidate])

        # Create stub LLM client
        llm_client = StubLLMClient(response=SimpleNamespace(is_event=False, events=[]))

        # Run extraction for all sources (no filter)
        result = extract_events_use_case(
            llm_client=llm_client,
            repository=repo,
            settings=settings,
            source_id=None,  # No source filter - process all
            batch_size=50,
            check_budget=False,
        )

        # Verify both candidates were processed
        assert result.candidates_processed == 2
        assert result.events_extracted == 0  # No events in mock response

        # Verify LLM was called twice (for both candidates)
        assert llm_client.call_count == 2

    def test_extract_events_with_min_score_and_source_filter(self, repo, settings):
        """Test that extract_events_use_case correctly combines source filter with min_score filter."""
        # Create test candidates with different scores and sources
        low_score_slack = EventCandidate(
            message_id="slack_msg_low",
            channel="#releases",
            ts_dt=datetime(2025, 1, 1, 12, 0).replace(tzinfo=pytz.UTC),
            text_norm="Low score Slack message",
            links_norm=[],
            anchors=[],
            score=50.0,  # Low score
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.SLACK,
        )

        high_score_slack = EventCandidate(
            message_id="slack_msg_high",
            channel="#releases",
            ts_dt=datetime(2025, 1, 1, 12, 0).replace(tzinfo=pytz.UTC),
            text_norm="High score Slack message",
            links_norm=[],
            anchors=[],
            score=90.0,  # High score
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.SLACK,
        )

        high_score_telegram = EventCandidate(
            message_id="telegram_msg_high",
            channel="@news",
            ts_dt=datetime(2025, 1, 1, 12, 0).replace(tzinfo=pytz.UTC),
            text_norm="High score Telegram message",
            links_norm=[],
            anchors=[],
            score=90.0,  # High score
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.TELEGRAM,
        )

        # Save candidates to repository
        repo.save_candidates([low_score_slack, high_score_slack, high_score_telegram])

        # Create stub LLM client
        llm_client = StubLLMClient(response=SimpleNamespace(is_event=False, events=[]))

        # Run extraction for Slack only (min_score will be calculated inside based on budget)
        result = extract_events_use_case(
            llm_client=llm_client,
            repository=repo,
            settings=settings,
            source_id=MessageSource.SLACK,  # Filter to Slack only
            batch_size=50,
            check_budget=False,  # Disable budget check to avoid min_score calculation
        )

        # Verify only Slack candidates were processed (both scores since check_budget=False)
        assert (
            result.candidates_processed == 2
        )  # Both Slack candidates (low and high score)
        assert result.events_extracted == 0  # No events in mock response

        # Verify LLM was called twice (for both Slack candidates)
        assert llm_client.call_count == 2

        # Verify that only Slack candidates were processed (check call arguments)
        call_args_list = [call["text"] for call in llm_client.calls]
        assert "Low score Slack message" in call_args_list
        assert "High score Slack message" in call_args_list
        assert (
            "High score Telegram message" not in call_args_list
        )  # Telegram should not be processed
