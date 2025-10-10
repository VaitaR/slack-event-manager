"""Date and time resolution service.

Handles:
- Absolute dates (ISO8601, EU, US formats)
- Relative dates (today, tomorrow, next week, in N days)
- Date ranges (10-12 Oct)
- Timezone detection and conversion to UTC
- Conflict resolution (absolute > relative)
"""

import re
from datetime import datetime, timedelta
from typing import Final

import pytz
from dateutil import parser as dateutil_parser

from src.config.settings import get_settings

# Timezone abbreviations mapping
TZ_MAP: Final[dict[str, str]] = {
    "CET": "Europe/Amsterdam",
    "CEST": "Europe/Amsterdam",
    "UTC": "UTC",
    "GMT": "UTC",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "EST": "America/New_York",
    "EDT": "America/New_York",
}
"""Mapping of common timezone abbreviations to IANA timezone names."""

# Relative date keywords mapping (days offset from today)
RELATIVE_PATTERNS: Final[dict[str, int]] = {
    "today": 0,
    "tomorrow": 1,
    "yesterday": -1,
}
"""Relative date keywords and their day offsets from current date."""

# Pattern for "in N days/weeks/months"
IN_N_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"in\s+(\d+)\s+(day|days|week|weeks|month|months)", flags=re.IGNORECASE
)
"""Pattern to match phrases like 'in 3 days', 'in 2 weeks', 'in 1 month'."""

# Pattern for "next week/month"
NEXT_PATTERN: Final[re.Pattern[str]] = re.compile(r"next\s+(week|month)", flags=re.IGNORECASE)
"""Pattern to match 'next week' or 'next month' phrases."""

# Pattern for date ranges like "10-12 Oct" or "Sep 5-7"
RANGE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\d{1,2})\s*-\s*(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    flags=re.IGNORECASE,
)
"""Pattern to match date ranges like '10-12 Oct 2025' or 'Sep 5-7'."""

# Pattern for "ETA Sep 17" or "ETA 17 Sep"
ETA_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"ETA\s+(?:(\d{1,2})\s+)?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})?",
    flags=re.IGNORECASE,
)
"""Pattern to match ETA (Estimated Time of Arrival) dates like 'ETA Sep 17' or 'ETA 17 Sep'."""

# EOD/COB patterns (End of Day / Close of Business)
EOD_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b(EOD|COB)\b", flags=re.IGNORECASE)
"""Pattern to match End of Day (EOD) or Close of Business (COB) indicators."""


def get_default_timezone() -> pytz.BaseTzInfo:
    """Get default timezone from settings.

    Returns:
        Timezone object
    """
    settings = get_settings()
    try:
        return pytz.timezone(settings.tz_default)
    except pytz.UnknownTimeZoneError:
        return pytz.timezone("Europe/Amsterdam")


def detect_timezone(text: str) -> pytz.BaseTzInfo:
    """Detect timezone from text.

    Args:
        text: Message text

    Returns:
        Detected timezone or default

    Example:
        >>> detect_timezone("Meeting at 3pm CET")
        <DstTzInfo 'Europe/Amsterdam' ...>
    """
    text_upper = text.upper()
    for abbr, tz_name in TZ_MAP.items():
        if abbr in text_upper:
            return pytz.timezone(tz_name)

    return get_default_timezone()


def parse_absolute_date(text: str, reference_dt: datetime | None = None) -> datetime | None:
    """Parse absolute date/time from text.

    Args:
        text: Text to parse
        reference_dt: Reference datetime for relative resolution

    Returns:
        Parsed datetime in UTC or None

    Example:
        >>> parse_absolute_date("2025-10-15 14:30")
        datetime(2025, 10, 15, 12, 30, tzinfo=<UTC>)  # Assuming CET
    """
    if not text:
        return None

    try:
        # Detect timezone
        tz = detect_timezone(text)

        # Try dateutil parser
        parsed = dateutil_parser.parse(text, fuzzy=True, default=reference_dt)

        # If parsed datetime is naive, localize it
        if parsed.tzinfo is None:
            parsed = tz.localize(parsed)

        # Convert to UTC
        return parsed.astimezone(pytz.UTC)
    except (ValueError, TypeError, OverflowError):
        return None


def parse_relative_date(
    text: str, reference_dt: datetime | None = None
) -> datetime | None:
    """Parse relative date expressions.

    Args:
        text: Text containing relative date
        reference_dt: Reference datetime (default: now)

    Returns:
        Resolved datetime in UTC or None

    Example:
        >>> parse_relative_date("tomorrow", datetime(2025, 10, 10))
        datetime(2025, 10, 11, 8, 0, tzinfo=<UTC>)  # 10:00 local
    """
    if reference_dt is None:
        reference_dt = datetime.utcnow().replace(tzinfo=pytz.UTC)

    text_lower = text.lower()

    # Handle simple relative dates (today, tomorrow, yesterday)
    for keyword, offset in RELATIVE_PATTERNS.items():
        if keyword in text_lower:
            target_date = reference_dt + timedelta(days=offset)
            # Default time: 10:00 local
            tz = get_default_timezone()
            local_dt = target_date.astimezone(tz).replace(
                hour=10, minute=0, second=0, microsecond=0
            )
            return local_dt.astimezone(pytz.UTC)

    # Handle "in N days/weeks/months"
    match = IN_N_PATTERN.search(text_lower)
    if match:
        quantity = int(match.group(1))
        unit = match.group(2).rstrip("s")  # Remove plural

        if unit == "day":
            target_date = reference_dt + timedelta(days=quantity)
        elif unit == "week":
            target_date = reference_dt + timedelta(weeks=quantity)
        elif unit == "month":
            target_date = reference_dt + timedelta(days=quantity * 30)  # Approximate
        else:
            return None

        tz = get_default_timezone()
        local_dt = target_date.astimezone(tz).replace(
            hour=10, minute=0, second=0, microsecond=0
        )
        return local_dt.astimezone(pytz.UTC)

    # Handle "next week/month"
    match = NEXT_PATTERN.search(text_lower)
    if match:
        unit = match.group(1)
        if unit == "week":
            target_date = reference_dt + timedelta(weeks=1)
        elif unit == "month":
            target_date = reference_dt + timedelta(days=30)
        else:
            return None

        tz = get_default_timezone()
        local_dt = target_date.astimezone(tz).replace(
            hour=10, minute=0, second=0, microsecond=0
        )
        return local_dt.astimezone(pytz.UTC)

    return None


def parse_range(text: str, reference_dt: datetime | None = None) -> tuple[datetime | None, datetime | None]:
    """Parse date ranges like '10-12 Oct' or 'Sep 5-7'.

    Args:
        text: Text containing range
        reference_dt: Reference datetime for year

    Returns:
        Tuple of (start_date, end_date) or (None, None)

    Example:
        >>> parse_range("10-12 Oct 2025")
        (datetime(2025, 10, 10, 8, 0), datetime(2025, 10, 12, 16, 0))
    """
    if reference_dt is None:
        reference_dt = datetime.utcnow().replace(tzinfo=pytz.UTC)

    match = RANGE_PATTERN.search(text)
    if not match:
        return None, None

    start_day = int(match.group(1))
    end_day = int(match.group(2))
    month_str = match.group(3)

    try:
        # Construct date strings
        year = reference_dt.year
        start_str = f"{start_day} {month_str} {year}"
        end_str = f"{end_day} {month_str} {year}"

        start_dt = parse_absolute_date(start_str, reference_dt)
        end_dt = parse_absolute_date(end_str, reference_dt)

        if start_dt and end_dt:
            # Set start to 10:00, end to 18:00 local time
            tz = get_default_timezone()
            start_local = start_dt.astimezone(tz).replace(hour=10, minute=0)
            end_local = end_dt.astimezone(tz).replace(hour=18, minute=0)

            return start_local.astimezone(pytz.UTC), end_local.astimezone(pytz.UTC)

    except (ValueError, AttributeError):
        pass

    return None, None


def parse_eta_date(text: str, reference_dt: datetime | None = None) -> datetime | None:
    """Parse ETA-style dates like 'ETA Sep 17' or 'ETA 26.09'.

    Args:
        text: Text containing ETA
        reference_dt: Reference datetime

    Returns:
        Parsed datetime or None

    Example:
        >>> parse_eta_date("ETA Sep 17")
        datetime(2025, 9, 17, 8, 0, tzinfo=<UTC>)
    """
    if reference_dt is None:
        reference_dt = datetime.utcnow().replace(tzinfo=pytz.UTC)

    match = ETA_PATTERN.search(text)
    if match:
        day1 = match.group(1)
        month = match.group(2)
        day2 = match.group(3)

        day = day1 if day1 else day2
        if not day:
            return None

        year = reference_dt.year
        date_str = f"{day} {month} {year}"
        return parse_absolute_date(date_str, reference_dt)

    return None


def resolve_time_for_eod(text: str, date_dt: datetime) -> datetime:
    """Set time to 18:00 for EOD/COB mentions.

    Args:
        text: Message text
        date_dt: Date to adjust

    Returns:
        Adjusted datetime

    Example:
        >>> resolve_time_for_eod("Deploy by EOD", datetime(2025, 10, 10, 10, 0))
        datetime(2025, 10, 10, 16, 0, tzinfo=<UTC>)  # 18:00 local
    """
    if EOD_PATTERN.search(text):
        tz = get_default_timezone()
        local_dt = date_dt.astimezone(tz).replace(hour=18, minute=0, second=0)
        return local_dt.astimezone(pytz.UTC)
    return date_dt


def resolve_event_date(
    text: str, reference_dt: datetime | None = None
) -> tuple[datetime | None, datetime | None]:
    """Resolve event date and optional end date from text.

    Priority: absolute > relative. Handles ranges.

    Args:
        text: Message text
        reference_dt: Reference datetime (message timestamp)

    Returns:
        Tuple of (event_date, event_end) in UTC

    Example:
        >>> resolve_event_date("Release on Oct 15")
        (datetime(2025, 10, 15, 8, 0, tzinfo=<UTC>), None)
    """
    if reference_dt is None:
        reference_dt = datetime.utcnow().replace(tzinfo=pytz.UTC)

    # Check for ranges first
    start_dt, end_dt = parse_range(text, reference_dt)
    if start_dt and end_dt:
        return start_dt, end_dt

    # Try ETA pattern
    eta_dt = parse_eta_date(text, reference_dt)
    if eta_dt:
        eta_dt = resolve_time_for_eod(text, eta_dt)
        return eta_dt, None

    # Try absolute date parsing
    absolute_dt = parse_absolute_date(text, reference_dt)
    if absolute_dt:
        absolute_dt = resolve_time_for_eod(text, absolute_dt)
        return absolute_dt, None

    # Fall back to relative
    relative_dt = parse_relative_date(text, reference_dt)
    if relative_dt:
        relative_dt = resolve_time_for_eod(text, relative_dt)
        return relative_dt, None

    # No date found, use reference (message timestamp)
    return reference_dt, None

