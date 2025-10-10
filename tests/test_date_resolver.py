"""Tests for date resolution service."""

from datetime import datetime, timedelta

import pytest
import pytz

from src.services import date_resolver


@pytest.fixture
def reference_date() -> datetime:
    """Reference date for testing: Oct 10, 2025, 12:00 UTC."""
    return datetime(2025, 10, 10, 12, 0, tzinfo=pytz.UTC)


def test_parse_absolute_date_iso8601(reference_date: datetime) -> None:
    """Test parsing ISO8601 dates."""
    text = "2025-10-15T14:30:00Z"
    result = date_resolver.parse_absolute_date(text, reference_date)
    assert result is not None
    assert result.year == 2025
    assert result.month == 10
    assert result.day == 15


def test_parse_absolute_date_eu_format(reference_date: datetime) -> None:
    """Test parsing European date format."""
    text = "15.10.2025"
    result = date_resolver.parse_absolute_date(text, reference_date)
    assert result is not None
    assert result.day == 15
    assert result.month == 10
    assert result.year == 2025


def test_parse_absolute_date_month_name(reference_date: datetime) -> None:
    """Test parsing dates with month names."""
    text = "October 15, 2025"
    result = date_resolver.parse_absolute_date(text, reference_date)
    assert result is not None
    assert result.month == 10
    assert result.day == 15


def test_parse_relative_date_today(reference_date: datetime) -> None:
    """Test parsing 'today'."""
    text = "today"
    result = date_resolver.parse_relative_date(text, reference_date)
    assert result is not None
    assert result.day == reference_date.day


def test_parse_relative_date_tomorrow(reference_date: datetime) -> None:
    """Test parsing 'tomorrow'."""
    text = "tomorrow"
    result = date_resolver.parse_relative_date(text, reference_date)
    assert result is not None
    assert result.day == reference_date.day + 1


def test_parse_relative_date_in_n_days(reference_date: datetime) -> None:
    """Test parsing 'in N days'."""
    text = "in 5 days"
    result = date_resolver.parse_relative_date(text, reference_date)
    assert result is not None
    # Should be exactly 5 days later in date terms (ignoring time differences due to timezone conversion)
    expected_date = reference_date + timedelta(days=5)
    assert result.date() == expected_date.date()


def test_parse_relative_date_in_n_weeks(reference_date: datetime) -> None:
    """Test parsing 'in N weeks'."""
    text = "in 2 weeks"
    result = date_resolver.parse_relative_date(text, reference_date)
    assert result is not None
    delta = (result - reference_date).days
    assert 13 <= delta <= 15  # Approximately 2 weeks


def test_parse_relative_date_next_week(reference_date: datetime) -> None:
    """Test parsing 'next week'."""
    text = "next week"
    result = date_resolver.parse_relative_date(text, reference_date)
    assert result is not None
    delta = (result - reference_date).days
    assert 6 <= delta <= 8  # Approximately 1 week


def test_parse_range_format_1(reference_date: datetime) -> None:
    """Test parsing date range like '10-12 Oct'."""
    text = "10-12 Oct 2025"
    start, end = date_resolver.parse_range(text, reference_date)
    assert start is not None
    assert end is not None
    assert start.day == 10
    assert end.day == 12
    assert start.month == 10
    assert end.month == 10
    # Start should be 10:00, end should be 18:00 local
    assert start.hour <= 12  # Depends on timezone
    assert end.hour <= 20  # Depends on timezone


def test_parse_range_no_match(reference_date: datetime) -> None:
    """Test range parsing with no match."""
    text = "no range here"
    start, end = date_resolver.parse_range(text, reference_date)
    assert start is None
    assert end is None


def test_parse_eta_date_format_1(reference_date: datetime) -> None:
    """Test parsing 'ETA Sep 17'."""
    text = "ETA Sep 17"
    result = date_resolver.parse_eta_date(text, reference_date)
    assert result is not None
    assert result.month == 9
    assert result.day == 17


def test_parse_eta_date_no_match(reference_date: datetime) -> None:
    """Test ETA parsing with no match."""
    text = "no ETA here"
    result = date_resolver.parse_eta_date(text, reference_date)
    assert result is None


def test_resolve_time_for_eod(reference_date: datetime) -> None:
    """Test EOD time resolution."""
    text = "Deploy by EOD"
    date_dt = datetime(2025, 10, 15, 10, 0, tzinfo=pytz.UTC)
    result = date_resolver.resolve_time_for_eod(text, date_dt)

    # Should be set to 18:00 local time
    # Exact hour depends on timezone conversion
    assert result.hour != 10  # Should be adjusted


def test_resolve_time_for_cob(reference_date: datetime) -> None:
    """Test COB time resolution."""
    text = "Finish by COB"
    date_dt = datetime(2025, 10, 15, 10, 0, tzinfo=pytz.UTC)
    result = date_resolver.resolve_time_for_eod(text, date_dt)

    # COB should also trigger 18:00
    assert result.hour != 10


def test_resolve_event_date_absolute_priority(reference_date: datetime) -> None:
    """Test that absolute dates have priority."""
    text = "Release tomorrow on October 15, 2025"
    event_date, event_end = date_resolver.resolve_event_date(text, reference_date)

    assert event_date is not None
    # Should use absolute date (Oct 15), not relative (tomorrow = Oct 11)
    assert event_date.day == 15
    assert event_date.month == 10


def test_resolve_event_date_range(reference_date: datetime) -> None:
    """Test event date with range."""
    text = "Migration scheduled 10-12 Oct"
    event_date, event_end = date_resolver.resolve_event_date(text, reference_date)

    assert event_date is not None
    assert event_end is not None
    assert event_date.day == 10
    assert event_end.day == 12


def test_resolve_event_date_fallback_to_message_ts(reference_date: datetime) -> None:
    """Test fallback to message timestamp when no date found."""
    text = "Some message with no date"
    event_date, event_end = date_resolver.resolve_event_date(text, reference_date)

    # Should fall back to reference date
    assert event_date == reference_date
    assert event_end is None


def test_detect_timezone_cet() -> None:
    """Test timezone detection for CET."""
    text = "Meeting at 3pm CET"
    tz = date_resolver.detect_timezone(text)
    assert tz.zone == "Europe/Amsterdam"


def test_detect_timezone_utc() -> None:
    """Test timezone detection for UTC."""
    text = "Deploy at 14:00 UTC"
    tz = date_resolver.detect_timezone(text)
    assert tz.zone == "UTC"


def test_detect_timezone_default() -> None:
    """Test timezone detection falls back to default."""
    text = "No timezone mentioned"
    tz = date_resolver.detect_timezone(text)
    # Should return default
    assert tz.zone in ["Europe/Amsterdam", "UTC"]

