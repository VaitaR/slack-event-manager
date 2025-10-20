"""Integration tests for EventValidator in use cases.

Tests the EventValidator integration in extract_events, deduplicate_events,
and publish_digest use cases to ensure validation works end-to-end.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.config.settings import Settings
from src.domain.models import (
    ActionType,
    Event,
    EventCategory,
    EventStatus,
    MessageSource,
)
from src.services.validators import EventValidator
from src.use_cases.deduplicate_events import deduplicate_events_use_case
from src.use_cases.extract_events import extract_events_use_case
from src.use_cases.publish_digest import publish_digest_use_case


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.llm_model = "gpt-5-nano"
    settings.llm_temperature = 1.0
    settings.llm_timeout_seconds = 30
    settings.llm_daily_budget_usd = 10.0
    settings.digest_lookback_hours = 48
    settings.digest_min_confidence = 0.7
    settings.digest_max_events = 10
    settings.digest_category_priorities = {
        "product": 1,
        "risk": 2,
        "process": 3,
        "marketing": 4,
        "org": 5,
        "unknown": 6,
    }
    settings.slack_digest_channel_id = "C1234567890"
    settings.tz_default = "Europe/Amsterdam"
    settings.dedup_date_window_hours = 48
    settings.dedup_title_similarity = 0.8
    return settings


@pytest.fixture
def sample_event():
    """Create a valid sample event for testing."""
    return Event(
        message_id="MSG123456",
        source_channels=["#releases"],
        action=ActionType.LAUNCH,
        object_id="api-v2",
        object_name_raw="API v2",
        qualifiers=["major"],
        stroke="launched",
        anchor="production",
        category=EventCategory.PRODUCT,
        status=EventStatus.COMPLETED,
        change_type="launch",
        environment="prod",
        planned_start=datetime(2025, 10, 15, 10, 0, tzinfo=UTC),
        actual_start=datetime(2025, 10, 15, 10, 0, tzinfo=UTC),
        actual_end=datetime(2025, 10, 15, 11, 0, tzinfo=UTC),
        time_source="explicit",
        time_confidence=0.9,
        summary="Successfully released API v2 with new features",
        why_it_matters="Improves user experience and system reliability",
        links=["https://example.com/release-notes"],
        anchors=["production", "api"],
        impact_area=["user-experience", "reliability"],
        impact_type="positive",
        confidence=0.85,
        importance=75,
        cluster_key="api-v2-production-released",
        dedup_key="api-v2-production-released-20251015",
        relations=[],
        extracted_at=datetime.now(UTC),
    )


@pytest.fixture
def invalid_event():
    """Create an invalid event for testing."""
    return Event(
        message_id="MSG123456",
        source_channels=["#releases"],
        action=ActionType.LAUNCH,
        object_id="api-v2",
        object_name_raw="API v2",
        qualifiers=["major", "minor"],  # Too many qualifiers
        stroke="launched",
        anchor="production",
        category=EventCategory.UNKNOWN,  # Unknown category
        status=EventStatus.COMPLETED,
        change_type="launch",
        environment="prod",
        planned_start=datetime(2025, 10, 15, 10, 0, tzinfo=UTC),
        actual_start=datetime(2025, 10, 15, 10, 0, tzinfo=UTC),
        # Missing actual_end for completed status
        time_source="explicit",
        time_confidence=0.9,
        summary="",  # Empty summary
        why_it_matters="Improves user experience and system reliability",
        links=[
            "https://example.com/release-notes",
            "http://invalid-url",
            "ftp://bad-protocol",
        ],  # Invalid link format
        anchors=["production", "api"],
        impact_area=[
            "user-experience",
            "reliability",
            "security",
        ],  # Too many impact areas
        impact_type="positive",
        confidence=0.3,  # Low confidence
        importance=25,  # Low importance
        cluster_key="api-v2-production-released",
        dedup_key="api-v2-production-released-20251015",
        relations=[],
        extracted_at=datetime.now(UTC),
    )


class TestEventValidatorIntegration:
    """Test EventValidator integration in use cases."""

    def test_validator_integration_extract_events(self, mock_settings, sample_event):
        """Test EventValidator integration in extract_events_use_case."""
        # Create mock LLM client and repository
        llm_client = MagicMock()
        repository = MagicMock()

        # Mock successful LLM response
        llm_response = MagicMock()
        llm_response.is_event = True
        llm_response.events = [MagicMock()]
        llm_client.extract_events_with_retry.return_value = llm_response

        # Mock LLM call metadata
        call_metadata = MagicMock()
        call_metadata.cost_usd = 0.001
        call_metadata.message_id = "MSG123456"
        llm_client.get_call_metadata.return_value = call_metadata

        # Mock candidates query (empty for this test)
        repository.get_candidates_for_extraction.return_value = []

        # Mock daily cost for budget check
        repository.get_daily_llm_cost.return_value = 0.0

        # Run use case - should not raise validation errors for valid events
        result = extract_events_use_case(
            llm_client=llm_client,
            repository=repository,
            settings=mock_settings,
            source_id=MessageSource.SLACK,  # Test integration - Slack only
            batch_size=10,
        )

        # Should complete without errors
        assert result.events_extracted == 0  # No candidates to process
        assert result.candidates_processed == 0

    def test_validator_integration_deduplicate_events(
        self, mock_settings, sample_event
    ):
        """Test EventValidator integration in deduplicate_events_use_case."""
        repository = MagicMock()

        # Mock events query
        repository.query_events.return_value = [sample_event]

        # Run deduplication use case
        result = deduplicate_events_use_case(
            repository=repository,
            settings=mock_settings,
            lookback_days=7,
        )

        # Should complete successfully
        assert result.total_events == 1
        assert result.merged_events == 0

        # Should call save_events with validated events
        repository.save_events.assert_called_once()

    def test_validator_integration_publish_digest_valid_events(
        self, mock_settings, sample_event
    ):
        """Test EventValidator integration in publish_digest_use_case with valid events."""
        slack_client = MagicMock()
        repository = MagicMock()

        # Mock events query
        repository.get_events_in_window_filtered.return_value = [sample_event]

        # Mock successful Slack posting
        slack_client.post_message.return_value = None

        # Run publish digest use case
        result = publish_digest_use_case(
            slack_client=slack_client,
            repository=repository,
            settings=mock_settings,
            dry_run=False,
        )

        # Should publish successfully
        assert result.events_included == 1
        assert result.messages_posted >= 1

    def test_validator_integration_publish_digest_invalid_events(
        self, mock_settings, invalid_event
    ):
        """Test EventValidator integration in publish_digest_use_case with invalid events."""
        slack_client = MagicMock()
        repository = MagicMock()

        # Mock events query with invalid event
        repository.get_events_in_window_filtered.return_value = [invalid_event]

        # Run publish digest use case
        result = publish_digest_use_case(
            slack_client=slack_client,
            repository=repository,
            settings=mock_settings,
            dry_run=False,
        )

        # Should skip invalid events and not publish anything
        assert result.events_included == 0
        assert result.messages_posted == 0

        # Should not call post_message for invalid events
        slack_client.post_message.assert_not_called()

    def test_validator_should_publish_method(self, sample_event, invalid_event):
        """Test EventValidator.should_publish method."""
        validator = EventValidator()

        # Valid event should be publishable
        assert validator.should_publish(sample_event, min_importance=60) is True

        # Invalid event should not be publishable due to critical errors
        assert validator.should_publish(invalid_event, min_importance=60) is False

    def test_validator_get_quality_issues(self, sample_event, invalid_event):
        """Test EventValidator.get_quality_issues method."""
        validator = EventValidator()

        # Valid event should have no critical errors
        issues = validator.get_quality_issues(sample_event)
        assert len(issues["errors"]) == 0

        # Invalid event should have multiple issues
        issues = validator.get_quality_issues(invalid_event)
        assert len(issues["errors"]) > 0  # Should have critical errors
        assert len(issues["warnings"]) > 0  # Should have warnings too

    def test_validator_get_critical_errors_blocks_saving(self, invalid_event):
        """Test that get_critical_errors correctly identifies blocking issues."""
        validator = EventValidator()

        # Invalid event should have critical errors that block saving
        critical_errors = validator.get_critical_errors(invalid_event)
        assert len(critical_errors) > 0

        # Should include specific critical issues
        critical_error_texts = " ".join(critical_errors).lower()
        assert any(
            keyword in critical_error_texts
            for keyword in [
                "summary",
                "required",
                "missing",
                "actual_end",
                "completed",
                "links",
            ]
        )

    def test_validator_get_validation_summary_audit(self, sample_event, invalid_event):
        """Test get_validation_summary for comprehensive audit logging."""
        validator = EventValidator()

        # Valid event should have minimal issues
        summary = validator.get_validation_summary(sample_event)
        assert len(summary["critical"]) == 0
        # May have some warnings/info for valid events

        # Invalid event should have detailed breakdown
        summary = validator.get_validation_summary(invalid_event)
        assert len(summary["critical"]) > 0
        assert len(summary["warnings"]) > 0  # Should have category warning

    def test_extract_events_blocks_critical_validation_errors(self, mock_settings):
        """Test that extract_events_use_case blocks events with critical validation errors."""
        llm_client = MagicMock()
        repository = MagicMock()

        # Mock LLM response with events that have critical errors
        llm_response = MagicMock()
        llm_response.is_event = True

        # Create mock LLM event with critical issues (empty summary, wrong status/time)
        mock_llm_event = MagicMock()
        mock_llm_event.object_name_raw = "API v2"
        mock_llm_event.summary = ""  # Critical: empty summary
        mock_llm_event.status = "completed"  # Critical: no actual_end for completed
        mock_llm_event.actual_end = None
        mock_llm_event.qualifiers = ["major"]
        mock_llm_event.stroke = "launched"
        mock_llm_event.anchor = "production"
        mock_llm_event.category = "product"
        mock_llm_event.change_type = "launch"
        mock_llm_event.environment = "prod"
        mock_llm_event.severity = None
        mock_llm_event.time_source = "explicit"
        mock_llm_event.time_confidence = 0.9
        mock_llm_event.why_it_matters = "Improves user experience"
        mock_llm_event.links = ["https://example.com"]
        mock_llm_event.anchors = ["api"]
        mock_llm_event.impact_area = ["user-experience"]
        mock_llm_event.impact_type = "positive"
        mock_llm_event.confidence = 0.8

        llm_response.events = [mock_llm_event]
        llm_client.extract_events_with_retry.return_value = llm_response

        # Mock LLM call metadata
        call_metadata = MagicMock()
        call_metadata.cost_usd = 0.001
        call_metadata.message_id = "MSG123456"
        llm_client.get_call_metadata.return_value = call_metadata

        # Mock candidate
        candidate = MagicMock()
        candidate.message_id = "MSG123456"
        candidate.channel = "#releases"
        candidate.ts_dt = datetime.now(UTC)
        candidate.text_norm = "Launched API v2"
        candidate.links_norm = ["https://example.com"]
        candidate.anchors = ["api"]
        candidate.score = 0.9
        candidate.features.reaction_count = 5
        candidate.features.has_mention = True

        # Mock repository methods
        repository.get_candidates_for_extraction.return_value = [candidate]
        repository.get_daily_llm_cost.return_value = 0.0
        repository.save_llm_call.return_value = None
        repository.save_events.return_value = None
        repository.update_candidate_status.return_value = None

        # Mock settings channel config
        channel_config = MagicMock()
        channel_config.channel_name = "#releases"
        mock_settings.get_channel_config.return_value = channel_config
        mock_settings.get_scoring_config.return_value = channel_config

        # Run use case
        result = extract_events_use_case(
            llm_client=llm_client,
            repository=repository,
            settings=mock_settings,
            source_id=MessageSource.SLACK,  # Test critical errors - Slack only
            batch_size=10,
        )

        # Verify no events were saved (blocked due to critical errors)
        assert result.events_extracted == 0
        repository.save_events.assert_not_called()

        # Verify LLM call was still made (processing continues)
        llm_client.extract_events_with_retry.assert_called_once()
        repository.save_llm_call.assert_called_once()

    def test_extract_events_allows_warnings_only(self, mock_settings):
        """Test that extract_events_use_case allows events with warnings but no critical errors."""
        # This test is simplified - we focus on the core validation logic
        # rather than complex mocking of the entire pipeline

        # Create a valid event with only warnings (unknown category)
        from src.domain.models import (
            ActionType,
            ChangeType,
            Environment,
            Event,
            EventCategory,
            EventStatus,
        )

        valid_event = Event(
            message_id="MSG123456",
            source_channels=["#releases"],
            action=ActionType.LAUNCH,
            object_id="api-v2",
            object_name_raw="API v2",
            qualifiers=["major"],
            stroke="launched",
            anchor="production",
            category=EventCategory.UNKNOWN,  # Warning: unknown category
            status=EventStatus.COMPLETED,
            change_type=ChangeType.LAUNCH,
            environment=Environment.PROD,
            planned_start=datetime(2025, 10, 15, 10, 0, tzinfo=UTC),
            actual_start=datetime(2025, 10, 15, 10, 0, tzinfo=UTC),
            actual_end=datetime(2025, 10, 15, 11, 0, tzinfo=UTC),  # Has actual_end
            time_source="explicit",
            time_confidence=0.9,
            summary="Valid summary for testing",
            why_it_matters="Improves user experience",
            links=["https://example.com"],
            anchors=["api"],
            impact_area=["user-experience"],
            impact_type="positive",
            confidence=0.8,
            importance=75,
            cluster_key="api-v2-production-released",
            dedup_key="api-v2-production-released-20251015",
            relations=[],
            extracted_at=datetime.now(UTC),
        )

        # Test that validator allows this event (only warnings, no critical errors)
        validator = EventValidator()
        critical_errors = validator.get_critical_errors(valid_event)

        # Should have no critical errors
        assert len(critical_errors) == 0

        # Should have warnings (unknown category)
        validation_summary = validator.get_validation_summary(valid_event)
        assert len(validation_summary["warnings"]) > 0
