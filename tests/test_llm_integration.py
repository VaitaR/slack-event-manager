"""Integration tests for LLM event conversion.

Tests the critical convert_llm_event_to_domain function that integrates
all new services: ObjectRegistry, ImportanceScorer, TitleRenderer, Deduplicator.
"""

from datetime import datetime

import pytest
import pytz

from src.domain.models import EventCategory, LLMEvent
from src.domain.validation_constants import (
    MAX_IMPACT_AREAS,
    MAX_LINKS,
    MAX_QUALIFIERS,
)
from src.use_cases.extract_events import convert_llm_event_to_domain

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
st = hypothesis.strategies


@pytest.fixture(autouse=True)
def reset_global_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset global caches before each test to ensure isolation.

    This prevents issues where tests run in different working directories
    and the global cache holds stale paths.
    """
    from pathlib import Path

    import src.use_cases.extract_events as extract_module

    # Reset global caches
    extract_module._object_registry = None
    extract_module._importance_scorer = None

    # Ensure we're in the correct working directory for relative paths
    project_root = Path(__file__).parent.parent
    monkeypatch.chdir(project_root)


@pytest.fixture
def mock_llm_event() -> LLMEvent:
    """Create mock LLM extraction result."""
    return LLMEvent(
        action="Launch",
        object_name_raw="Stocks & ETFs",  # Should be canonicalized
        qualifiers=["alpha", "Wallet team"],
        stroke="degradation possible",
        anchor="INV-1024",
        category=EventCategory.PRODUCT,
        status="started",
        change_type="launch",
        environment="prod",
        severity=None,
        planned_start=None,
        planned_end=None,
        actual_start="2025-10-20T10:00:00Z",
        actual_end=None,
        time_source="explicit",
        time_confidence=0.9,
        summary="Launching Stocks & ETFs trading in alpha for Wallet team",
        why_it_matters="New trading capability for alpha users",
        links=["https://confluence.example.com/page1"],
        anchors=["INV-1024"],
        impact_area=["wallet", "trading"],
        impact_type=["perf_degradation"],
        confidence=0.85,
    )


def test_convert_llm_event_basic(mock_llm_event: LLMEvent) -> None:
    """Test basic conversion from LLM event to domain event."""
    result = convert_llm_event_to_domain(
        llm_event=mock_llm_event,
        message_id="test123",
        message_ts_dt=datetime(2025, 10, 20, 8, 0, tzinfo=pytz.UTC),
        channel_name="releases",
        reaction_count=5,
        mention_count=1,
    )

    # Check basic fields
    assert result.message_id == "test123"
    assert result.object_name_raw == "Stocks & ETFs"
    assert result.qualifiers == ["alpha", "Wallet team"]
    assert result.stroke == "degradation possible"
    assert result.anchor == "INV-1024"


def test_convert_llm_event_object_canonicalization(mock_llm_event: LLMEvent) -> None:
    """Test that object_id is canonicalized via ObjectRegistry."""
    result = convert_llm_event_to_domain(
        llm_event=mock_llm_event,
        message_id="test123",
        message_ts_dt=datetime.utcnow().replace(tzinfo=pytz.UTC),
        channel_name="releases",
    )

    # "Stocks & ETFs" should be canonicalized to "wallet.stocks" by registry
    assert result.object_id == "wallet.stocks"


def test_convert_llm_event_importance_calculated(mock_llm_event: LLMEvent) -> None:
    """Test that importance score is calculated."""
    result = convert_llm_event_to_domain(
        llm_event=mock_llm_event,
        message_id="test123",
        message_ts_dt=datetime.utcnow().replace(tzinfo=pytz.UTC),
        channel_name="releases",
        reaction_count=10,  # Should boost importance
        mention_count=1,
    )

    # Importance should be calculated (not 0)
    assert result.importance > 0
    assert result.importance <= 100


def test_convert_llm_event_cluster_key_generated(mock_llm_event: LLMEvent) -> None:
    """Test that cluster_key is generated."""
    result = convert_llm_event_to_domain(
        llm_event=mock_llm_event,
        message_id="test123",
        message_ts_dt=datetime.utcnow().replace(tzinfo=pytz.UTC),
        channel_name="releases",
    )

    assert result.cluster_key != ""
    assert len(result.cluster_key) == 40  # SHA1 hex digest


def test_convert_llm_event_dedup_key_generated(mock_llm_event: LLMEvent) -> None:
    """Test that dedup_key is generated."""
    result = convert_llm_event_to_domain(
        llm_event=mock_llm_event,
        message_id="test123",
        message_ts_dt=datetime.utcnow().replace(tzinfo=pytz.UTC),
        channel_name="releases",
    )

    assert result.dedup_key != ""
    assert len(result.dedup_key) == 40  # SHA1 hex digest


def test_convert_llm_event_time_parsing(mock_llm_event: LLMEvent) -> None:
    """Test that time fields are parsed correctly."""
    result = convert_llm_event_to_domain(
        llm_event=mock_llm_event,
        message_id="test123",
        message_ts_dt=datetime.utcnow().replace(tzinfo=pytz.UTC),
        channel_name="releases",
    )

    assert result.actual_start is not None
    assert result.actual_start.year == 2025
    assert result.actual_start.month == 10
    assert result.actual_start.day == 20


@given(
    qualifiers=st.lists(st.text(min_size=1, max_size=10), max_size=6),
    links=st.lists(st.text(min_size=5, max_size=30), max_size=6),
    impact_areas=st.lists(st.text(min_size=1, max_size=20), max_size=6),
)
def test_convert_llm_event_respects_domain_limits(
    mock_llm_event: LLMEvent,
    qualifiers: list[str],
    links: list[str],
    impact_areas: list[str],
) -> None:
    """Conversion should respect domain-configured limits."""

    llm_event = mock_llm_event.model_copy(
        update={
            "qualifiers": qualifiers,
            "links": links,
            "impact_area": impact_areas,
        }
    )

    result = convert_llm_event_to_domain(
        llm_event=llm_event,
        message_id="msg",
        message_ts_dt=datetime(2025, 10, 20, 8, 0, tzinfo=pytz.UTC),
        channel_name="releases",
    )

    assert len(result.qualifiers) <= MAX_QUALIFIERS
    assert result.qualifiers == llm_event.qualifiers[:MAX_QUALIFIERS]

    assert len(result.links) <= MAX_LINKS
    assert result.links == llm_event.links[:MAX_LINKS]

    assert len(result.impact_area) <= MAX_IMPACT_AREAS
    assert result.impact_area == llm_event.impact_area[:MAX_IMPACT_AREAS]


def test_convert_llm_event_enum_parsing(mock_llm_event: LLMEvent) -> None:
    """Test that enums are parsed correctly."""
    result = convert_llm_event_to_domain(
        llm_event=mock_llm_event,
        message_id="test123",
        message_ts_dt=datetime.utcnow().replace(tzinfo=pytz.UTC),
        channel_name="releases",
    )

    from src.domain.models import (
        ActionType,
        ChangeType,
        Environment,
        EventStatus,
        TimeSource,
    )

    assert result.action == ActionType.LAUNCH
    assert result.status == EventStatus.STARTED
    assert result.change_type == ChangeType.LAUNCH
    assert result.environment == Environment.PROD
    assert result.time_source == TimeSource.EXPLICIT


def test_convert_llm_event_invalid_enum_fallback(mock_llm_event: LLMEvent) -> None:
    """Test that invalid enums fallback to defaults."""
    mock_llm_event.action = "INVALID_ACTION"
    mock_llm_event.status = "invalid_status"

    result = convert_llm_event_to_domain(
        llm_event=mock_llm_event,
        message_id="test123",
        message_ts_dt=datetime.utcnow().replace(tzinfo=pytz.UTC),
        channel_name="releases",
    )

    from src.domain.models import ActionType, EventStatus

    # Should fallback to OTHER/UPDATED
    assert result.action == ActionType.OTHER
    assert result.status == EventStatus.UPDATED


def test_convert_llm_event_max_constraints(mock_llm_event: LLMEvent) -> None:
    """Test that max constraints are enforced."""
    mock_llm_event.qualifiers = ["q1", "q2", "q3", "q4"]  # Too many
    mock_llm_event.links = ["link1", "link2", "link3", "link4", "link5"]  # Too many
    mock_llm_event.impact_area = ["a1", "a2", "a3", "a4", "a5"]  # Too many

    result = convert_llm_event_to_domain(
        llm_event=mock_llm_event,
        message_id="test123",
        message_ts_dt=datetime.utcnow().replace(tzinfo=pytz.UTC),
        channel_name="releases",
    )

    # Should be truncated
    assert len(result.qualifiers) <= 2
    assert len(result.links) <= 3
    assert len(result.impact_area) <= 3


def test_convert_llm_event_unknown_object_no_canonicalization(
    mock_llm_event: LLMEvent,
) -> None:
    """Test that unknown objects don't get object_id."""
    mock_llm_event.object_name_raw = "Unknown System XYZ"

    result = convert_llm_event_to_domain(
        llm_event=mock_llm_event,
        message_id="test123",
        message_ts_dt=datetime.utcnow().replace(tzinfo=pytz.UTC),
        channel_name="releases",
    )

    # Unknown object should have None object_id
    assert result.object_id is None
    assert result.object_name_raw == "Unknown System XYZ"


def test_convert_llm_event_full_integration() -> None:
    """Full integration test: LLM event -> Domain event with all services."""
    llm_event = LLMEvent(
        action="Deploy",
        object_name_raw="ClickHouse",  # Should match in registry
        qualifiers=["production"],
        stroke=None,
        anchor="DEPLOY-789",
        category=EventCategory.PRODUCT,
        status="completed",
        change_type="deploy",
        environment="prod",
        severity=None,
        planned_start="2025-10-18T08:00:00Z",
        planned_end=None,
        actual_start="2025-10-18T08:15:00Z",
        actual_end="2025-10-18T09:00:00Z",
        time_source="explicit",
        time_confidence=1.0,
        summary="Deployed ClickHouse cluster upgrade to production",
        why_it_matters="Performance improvements and bug fixes",
        links=["https://jira.example.com/DEPLOY-789"],
        anchors=["DEPLOY-789"],
        impact_area=["database", "analytics"],
        impact_type=["perf_degradation"],
        confidence=0.95,
    )

    result = convert_llm_event_to_domain(
        llm_event=llm_event,
        message_id="msg456",
        message_ts_dt=datetime(2025, 10, 18, 8, 0, tzinfo=pytz.UTC),
        channel_name="deployments",
        reaction_count=3,
        mention_count=0,
    )

    # Verify all integrations
    assert result.object_id == "data.clickhouse"  # ObjectRegistry
    assert result.importance > 0  # ImportanceScorer
    assert result.cluster_key != ""  # Deduplicator
    assert result.dedup_key != ""  # Deduplicator
    assert result.actual_start is not None  # Time parsing
    assert result.actual_end is not None

    # Can be rendered
    from src.services.title_renderer import TitleRenderer

    renderer = TitleRenderer()
    title = renderer.render_canonical_title(result)
    assert "Deploy" in title
    assert "ClickHouse" in title or "production" in title
