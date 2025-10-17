"""Tests for TitleRenderer service.

Critical tests for title rendering from slots (single source of truth).
"""

from datetime import datetime

import pytest
import pytz

from src.domain.models import (
    ActionType,
    ChangeType,
    Environment,
    Event,
    EventCategory,
    EventStatus,
    Severity,
    TimeSource,
)
from src.services.title_renderer import TitleRenderer


@pytest.fixture
def renderer() -> TitleRenderer:
    """Create TitleRenderer instance."""
    return TitleRenderer()


@pytest.fixture
def base_event() -> Event:
    """Create base event for testing."""
    return Event(
        message_id="test123",
        source_channels=["test-channel"],
        action=ActionType.LAUNCH,
        object_name_raw="Test Feature",
        category=EventCategory.PRODUCT,
        status=EventStatus.COMPLETED,
        change_type=ChangeType.LAUNCH,
        environment=Environment.PROD,
        actual_start=datetime.utcnow().replace(tzinfo=pytz.UTC),
        time_source=TimeSource.EXPLICIT,
        time_confidence=0.9,
        summary="Test feature launched",
        confidence=0.85,
        importance=75,
        cluster_key="test_cluster",
        dedup_key="test_dedup",
    )


def test_render_canonical_basic(renderer: TitleRenderer, base_event: Event) -> None:
    """Test basic canonical title rendering."""
    title = renderer.render_canonical_title(base_event)
    assert title == "Launch: Test Feature"


def test_render_canonical_with_qualifiers(
    renderer: TitleRenderer, base_event: Event
) -> None:
    """Test canonical title with qualifiers."""
    base_event.qualifiers = ["alpha", "backend team"]
    title = renderer.render_canonical_title(base_event)
    assert title == "Launch: Test Feature — alpha, backend team"


def test_render_canonical_with_stroke(
    renderer: TitleRenderer, base_event: Event
) -> None:
    """Test canonical title with stroke."""
    base_event.qualifiers = ["alpha"]
    base_event.stroke = "degradation possible"
    title = renderer.render_canonical_title(base_event)
    assert title == "Launch: Test Feature — alpha, degradation possible"


def test_render_canonical_with_anchor(
    renderer: TitleRenderer, base_event: Event
) -> None:
    """Test canonical title with anchor."""
    base_event.anchor = "TEST-123"
    title = renderer.render_canonical_title(base_event)
    assert title == "Launch: Test Feature (TEST-123)"


def test_render_canonical_full(renderer: TitleRenderer, base_event: Event) -> None:
    """Test canonical title with all slots."""
    base_event.qualifiers = ["alpha", "backend team"]
    base_event.stroke = "completed"
    base_event.anchor = "TEST-123"
    title = renderer.render_canonical_title(base_event)
    assert title == "Launch: Test Feature — alpha, backend team, completed (TEST-123)"


def test_render_object_first(renderer: TitleRenderer, base_event: Event) -> None:
    """Test object-first format."""
    base_event.qualifiers = ["alpha"]
    title = renderer.render_canonical_title(base_event, format_style="object_first")
    assert title == "Test Feature: Launch — alpha"


def test_render_severity_first(renderer: TitleRenderer, base_event: Event) -> None:
    """Test severity-first format for risk events."""
    base_event.category = EventCategory.RISK
    base_event.action = ActionType.INCIDENT
    base_event.severity = Severity.SEV2
    base_event.stroke = "degradation"

    title = renderer.render_canonical_title(base_event, format_style="severity_first")
    assert title == "SEV2: incident — Test Feature, degradation"


def test_truncate_long_title(renderer: TitleRenderer, base_event: Event) -> None:
    """Test title truncation when too long."""
    base_event.object_name_raw = "Very Long Feature Name" * 10  # Make it super long
    base_event.qualifiers = ["qual1", "qual2"]
    base_event.stroke = "some stroke"
    base_event.anchor = "ANCHOR-123"

    title = renderer.render_canonical_title(base_event)

    # Should be truncated to reasonable length (allow small margin for rounding)
    assert len(title) <= 145


def test_validate_title_structure_valid(
    renderer: TitleRenderer, base_event: Event
) -> None:
    """Test validation of valid title structure."""
    base_event.qualifiers = ["alpha"]
    base_event.anchor = "TEST-123"

    errors = renderer.validate_title_structure(base_event)
    assert len(errors) == 0


def test_validate_title_structure_too_many_qualifiers(
    renderer: TitleRenderer, base_event: Event
) -> None:
    """Test validation rejects too many qualifiers."""
    base_event.qualifiers = ["qual1", "qual2", "qual3"]  # Max is 2

    errors = renderer.validate_title_structure(base_event)
    assert len(errors) > 0
    assert any("qualifiers" in e.lower() for e in errors)


def test_validate_title_structure_no_urls(
    renderer: TitleRenderer, base_event: Event
) -> None:
    """Test validation rejects URLs in slots."""
    base_event.object_name_raw = "Feature https://example.com"

    errors = renderer.validate_title_structure(base_event)
    assert len(errors) > 0
    assert any("url" in e.lower() for e in errors)


def test_validate_title_structure_no_dates(
    renderer: TitleRenderer, base_event: Event
) -> None:
    """Test validation rejects dates in slots."""
    base_event.qualifiers = ["2025-10-15"]

    errors = renderer.validate_title_structure(base_event)
    assert len(errors) > 0
    assert any("date" in e.lower() for e in errors)
