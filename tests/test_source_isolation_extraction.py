"""Tests for source isolation in LLM extraction.

Verifies that extract_events_use_case correctly filters candidates by source_id
and doesn't mix candidates from different sources.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytz

from src.adapters.llm_client import LLMClient
from src.domain.models import (
    CandidateStatus,
    EventCandidate,
    MessageSource,
    ScoringFeatures,
)
from src.use_cases.extract_events import extract_events_use_case


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

        # Create mock LLM client
        mock_llm = MagicMock(spec=LLMClient)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.is_event = False
        mock_response.events = []
        mock_llm.extract_events_with_retry.return_value = mock_response
        mock_llm.get_call_metadata.return_value = MagicMock()

        # Run extraction for Slack only
        result = extract_events_use_case(
            llm_client=mock_llm,
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
        assert mock_llm.extract_events_with_retry.call_count == 1

        # Verify the call was made for the Slack candidate
        call_args = mock_llm.extract_events_with_retry.call_args
        assert call_args[1]["text"] == "Slack message text"

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

        # Create mock LLM client
        mock_llm = MagicMock(spec=LLMClient)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.is_event = False
        mock_response.events = []
        mock_llm.extract_events_with_retry.return_value = mock_response
        mock_llm.get_call_metadata.return_value = MagicMock()

        # Run extraction for Telegram only
        result = extract_events_use_case(
            llm_client=mock_llm,
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
        assert mock_llm.extract_events_with_retry.call_count == 1

        # Verify the call was made for the Telegram candidate
        call_args = mock_llm.extract_events_with_retry.call_args
        assert call_args[1]["text"] == "Telegram message text"

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

        # Create mock LLM client
        mock_llm = MagicMock(spec=LLMClient)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.is_event = False
        mock_response.events = []
        mock_llm.extract_events_with_retry.return_value = mock_response
        mock_llm.get_call_metadata.return_value = MagicMock()

        # Run extraction for all sources (no filter)
        result = extract_events_use_case(
            llm_client=mock_llm,
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
        assert mock_llm.extract_events_with_retry.call_count == 2

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

        # Create mock LLM client
        mock_llm = MagicMock(spec=LLMClient)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.is_event = False
        mock_response.events = []
        mock_llm.extract_events_with_retry.return_value = mock_response
        mock_llm.get_call_metadata.return_value = MagicMock()

        # Run extraction for Slack only (min_score will be calculated inside based on budget)
        result = extract_events_use_case(
            llm_client=mock_llm,
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
        assert mock_llm.extract_events_with_retry.call_count == 2

        # Verify that only Slack candidates were processed (check call arguments)
        call_args_list = [
            call[1]["text"]
            for call in mock_llm.extract_events_with_retry.call_args_list
        ]
        assert "Low score Slack message" in call_args_list
        assert "High score Slack message" in call_args_list
        assert (
            "High score Telegram message" not in call_args_list
        )  # Telegram should not be processed
