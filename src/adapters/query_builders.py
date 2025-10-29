"""Query builders for constructing type-safe database queries.

Instead of building SQL WHERE clauses with string literals, use these
builders to create queries in a type-safe, testable way.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.domain.models import EventCategory


class BaseQueryCriteria(ABC):
    """Base class for database query criteria with dialect support."""

    @abstractmethod
    def to_where_clause(self) -> tuple[str, list[Any]]:
        """Build SQL WHERE clause with parameters."""
        pass

    @abstractmethod
    def to_order_clause(self) -> str:
        """Build SQL ORDER BY clause."""
        pass

    @abstractmethod
    def to_limit_clause(self) -> tuple[str, list[Any]]:
        """Build SQL LIMIT/OFFSET clause."""
        pass


@dataclass
class EventQueryCriteria:
    """Criteria for querying events from the database.

    This class encapsulates all possible filter conditions for events,
    providing a clean API for building queries without SQL string manipulation.

    Example:
        >>> criteria = EventQueryCriteria(
        ...     start_date=datetime(2025, 10, 1),
        ...     end_date=datetime(2025, 10, 31),
        ...     categories=[EventCategory.PRODUCT],
        ...     min_confidence=0.8
        ... )
        >>> where, params = criteria.to_where_clause()
        >>> # Use in SQL: SELECT * FROM events WHERE {where}
    """

    # Date filters (based on primary event time)
    start_date: datetime | None = None
    """Filter events with primary time >= start_date (COALESCE(actual_start, actual_end, planned_start, planned_end))"""

    end_date: datetime | None = None
    """Filter events with primary time <= end_date (COALESCE(actual_start, actual_end, planned_start, planned_end))"""

    extracted_after: datetime | None = None
    """Filter events extracted after this timestamp"""

    extracted_before: datetime | None = None
    """Filter events extracted before this timestamp"""

    # Category and confidence filters
    categories: list[EventCategory] | None = None
    """Filter by event categories (OR logic)"""

    min_confidence: float | None = None
    """Minimum confidence score"""

    max_confidence: float | None = None
    """Maximum confidence score"""

    # Channel filters
    source_channels: list[str] | None = None
    """Filter by source channel names (OR logic)"""

    # Source filters
    source_id: str | None = None
    """Filter by message source (slack, telegram, etc.)"""

    # Message filters
    message_ids: list[str] | None = None
    """Filter by specific message IDs (OR logic)"""

    # Text search (removed - title no longer stored)
    # title_contains: str | None = None
    # """Search for text in event title (case-insensitive)"""

    # Version filters (removed - version no longer tracked)
    # min_version: int | None = None
    # """Minimum version number (for merged events)"""

    # Limits
    limit: int | None = None
    """Maximum number of results"""

    offset: int = 0
    """Offset for pagination"""

    # Ordering
    order_by: str = "extracted_at"
    """Column to order by (default: extracted_at for chronological processing)"""

    order_desc: bool = True
    """Order descending (default: True for newest first)"""

    def to_where_clause(self) -> tuple[str, list[Any]]:
        """Build SQL WHERE clause with parameters.

        Returns:
            Tuple of (where_clause, parameters)

        Example:
            >>> criteria = EventQueryCriteria(start_date=datetime(...))
            >>> where, params = criteria.to_where_clause()
            >>> where
            'COALESCE(actual_start, actual_end, planned_start, planned_end, extracted_at) >= %s'
            >>> params
            ['2025-10-01T00:00:00+00:00']
        """
        conditions: list[str] = []
        params: list[Any] = []

        # Date range filters (using primary event time)
        # Primary time: COALESCE(actual_start, actual_end, planned_start, planned_end)
        if self.start_date:
            conditions.append(
                "COALESCE(actual_start, actual_end, planned_start, planned_end, extracted_at) >= %s"
            )
            params.append(self.start_date.isoformat())

        if self.end_date:
            conditions.append(
                "COALESCE(actual_start, actual_end, planned_start, planned_end, extracted_at) <= %s"
            )
            params.append(self.end_date.isoformat())

        # Extraction time filters
        if self.extracted_after:
            conditions.append("extracted_at >= %s")
            params.append(self.extracted_after.isoformat())

        if self.extracted_before:
            conditions.append("extracted_at <= %s")
            params.append(self.extracted_before.isoformat())

        # Category filter (OR logic)
        if self.categories:
            placeholders = ",".join("%s" * len(self.categories))
            conditions.append(f"category IN ({placeholders})")
            params.extend([cat.value for cat in self.categories])

        # Confidence filters
        if self.min_confidence is not None:
            conditions.append("confidence >= %s")
            params.append(self.min_confidence)

        if self.max_confidence is not None:
            conditions.append("confidence <= %s")
            params.append(self.max_confidence)

        # Channel filter (JSON array search)
        if self.source_channels:
            # JSON array search: check if any channel is in source_channels array
            # Works for both SQLite (TEXT JSON) and PostgreSQL (JSONB)
            channel_conditions = []
            for channel in self.source_channels:
                channel_conditions.append("source_channels LIKE %s")
                params.append(f'%"{channel}"%')
            conditions.append(f"({' OR '.join(channel_conditions)})")

        # Source filter
        if self.source_id:
            conditions.append("source_id = %s")
            params.append(self.source_id)

        # Message ID filter
        if self.message_ids:
            placeholders = ",".join("%s" * len(self.message_ids))
            conditions.append(f"message_id IN ({placeholders})")
            params.extend(self.message_ids)

        # Title search - REMOVED (title column no longer exists)
        # Use object_name_raw or summary for text search instead
        # if self.title_contains:
        #     conditions.append("LOWER(title) LIKE %s")
        #     params.append(f"%{self.title_contains.lower()}%")

        # Version filter - REMOVED (version column no longer exists)
        # if self.min_version is not None:
        #     conditions.append("version >= %s")
        #     params.append(self.min_version)

        # Build WHERE clause
        where = " AND ".join(conditions) if conditions else "1=1"

        return where, params

    def to_order_clause(self) -> str:
        """Build SQL ORDER BY clause.

        Returns:
            ORDER BY clause string

        Example:
            >>> criteria = EventQueryCriteria(order_by="confidence", order_desc=True)
            >>> criteria.to_order_clause()
            'confidence DESC'
        """
        direction = "DESC" if self.order_desc else "ASC"
        return f"{self.order_by} {direction}"

    def to_limit_clause(self) -> tuple[str, list[Any]]:
        """Build SQL LIMIT/OFFSET clause.

        Returns:
            Tuple of (limit_clause, parameters)

        Example:
            >>> criteria = EventQueryCriteria(limit=10, offset=20)
            >>> clause, params = criteria.to_limit_clause()
            >>> clause
            'LIMIT %s OFFSET %s'
            >>> params
            [10, 20]
        """
        if self.limit is None:
            return "", []

        return "LIMIT %s OFFSET %s", [self.limit, self.offset]


@dataclass
class CandidateQueryCriteria:
    """Criteria for querying event candidates.

    Similar to EventQueryCriteria but for the event_candidates table.

    Example:
        >>> criteria = CandidateQueryCriteria(
        ...     status="new",
        ...     min_score=15.0,
        ...     limit=50
        ... )
        >>> where, params = criteria.to_where_clause()
    """

    # Score filters
    min_score: float | None = None
    """Minimum candidate score"""

    max_score: float | None = None
    """Maximum candidate score"""

    # Status filter
    status: str | None = None
    """Filter by processing status (new, processing, llm_ok, llm_fail)"""

    # Channel filter
    channel: str | None = None
    """Filter by specific channel ID"""

    # Date filters
    created_after: datetime | None = None
    """Filter candidates created after this date"""

    created_before: datetime | None = None
    """Filter candidates created before this date"""

    # Feature filters
    has_links: bool | None = None
    """Filter by presence of links"""

    has_anchors: bool | None = None
    """Filter by presence of anchors"""

    # Limits
    limit: int | None = None
    """Maximum number of results"""

    offset: int = 0
    """Offset for pagination"""

    # Ordering
    order_by: str = "score"
    """Column to order by (default: score)"""

    order_desc: bool = True
    """Order descending (default: True for highest scores first)"""

    def to_where_clause(self) -> tuple[str, list[Any]]:
        """Build SQL WHERE clause with parameters."""
        conditions: list[str] = []
        params: list[Any] = []

        # Score filters
        if self.min_score is not None:
            conditions.append("score >= %s")
            params.append(self.min_score)

        if self.max_score is not None:
            conditions.append("score <= %s")
            params.append(self.max_score)

        # Status filter
        if self.status:
            conditions.append("status = %s")
            params.append(self.status)

        # Channel filter
        if self.channel:
            conditions.append("channel = %s")
            params.append(self.channel)

        # Date filters
        if self.created_after:
            conditions.append("ts_dt >= %s")
            params.append(self.created_after.isoformat())

        if self.created_before:
            conditions.append("ts_dt <= %s")
            params.append(self.created_before.isoformat())

        # Feature filters
        if self.has_links is not None:
            if self.has_links:
                conditions.append("links_norm != '[]'")
            else:
                conditions.append("links_norm = '[]'")

        if self.has_anchors is not None:
            if self.has_anchors:
                conditions.append("anchors != '[]'")
            else:
                conditions.append("anchors = '[]'")

        where = " AND ".join(conditions) if conditions else "1=1"
        return where, params

    def to_order_clause(self) -> str:
        """Build SQL ORDER BY clause."""
        direction = "DESC" if self.order_desc else "ASC"
        return f"{self.order_by} {direction}"

    def to_limit_clause(self) -> tuple[str, list[Any]]:
        """Build SQL LIMIT/OFFSET clause."""
        if self.limit is None:
            return "", []
        return "LIMIT %s OFFSET %s", [self.limit, self.offset]


# Helper functions for common queries


def recent_events_criteria(days: int = 7) -> EventQueryCriteria:
    """Create criteria for events from last N days.

    Args:
        days: Number of days to look back

    Returns:
        EventQueryCriteria configured for recent events
    """
    from datetime import timedelta

    now = datetime.now(tz=UTC)
    return EventQueryCriteria(
        extracted_after=now - timedelta(days=days),
        order_by="extracted_at",
        order_desc=True,
    )


def high_priority_candidates_criteria(
    threshold: float = 15.0,
) -> CandidateQueryCriteria:
    """Create criteria for high-priority unprocessed candidates.

    Args:
        threshold: Minimum score threshold

    Returns:
        CandidateQueryCriteria for high-priority candidates
    """
    return CandidateQueryCriteria(
        status="new",
        min_score=threshold,
        order_by="score",
        order_desc=True,
    )


def product_events_criteria(
    start_date: datetime, end_date: datetime
) -> EventQueryCriteria:
    """Create criteria for product events in date range.

    Args:
        start_date: Start of date range (filters on primary event time)
        end_date: End of date range (filters on primary event time)

    Returns:
        EventQueryCriteria for product category events
    """
    return EventQueryCriteria(
        start_date=start_date,
        end_date=end_date,
        categories=[EventCategory.PRODUCT],
        min_confidence=0.7,
        order_by="actual_start",  # Order by primary time field
        order_desc=False,  # Chronological order
    )


@dataclass
class PostgresEventQueryCriteria(BaseQueryCriteria):
    """PostgreSQL-specific event query criteria using %s placeholders."""

    # Date filters (based on primary event time)
    start_date: datetime | None = None
    """Filter events with primary time >= start_date (COALESCE(actual_start, actual_end, planned_start, planned_end))"""

    end_date: datetime | None = None
    """Filter events with primary time <= end_date (COALESCE(actual_start, actual_end, planned_start, planned_end))"""

    extracted_after: datetime | None = None
    """Filter events extracted after this timestamp"""

    extracted_before: datetime | None = None
    """Filter events extracted before this timestamp"""

    # Category and confidence filters
    categories: list[EventCategory] | None = None
    """Filter by event categories (OR logic)"""

    min_confidence: float | None = None
    """Minimum confidence score"""

    max_confidence: float | None = None
    """Maximum confidence score"""

    # Channel filters
    source_channels: list[str] | None = None
    """Filter by source channel names (OR logic)"""

    # Source filters
    source_id: str | None = None
    """Filter by message source (slack, telegram, etc.)"""

    # Message filters
    message_ids: list[str] | None = None
    """Filter by specific message IDs (OR logic)"""

    # Limits
    limit: int | None = None
    """Maximum number of results"""

    offset: int = 0
    """Offset for pagination"""

    # Ordering
    order_by: str = "extracted_at"
    """Column to order by (default: extracted_at for chronological processing)"""

    order_desc: bool = True
    """Order descending (default: True for newest first)"""

    def to_where_clause(self) -> tuple[str, list[Any]]:
        """Build PostgreSQL WHERE clause with %s parameters."""
        conditions: list[str] = []
        params: list[Any] = []

        # Date range filters (using primary event time)
        if self.start_date:
            conditions.append(
                "COALESCE(actual_start, actual_end, planned_start, planned_end, extracted_at) >= %s"
            )
            params.append(self.start_date)

        if self.end_date:
            conditions.append(
                "COALESCE(actual_start, actual_end, planned_start, planned_end, extracted_at) <= %s"
            )
            params.append(self.end_date)

        # Extraction time filters
        if self.extracted_after:
            conditions.append("extracted_at >= %s")
            params.append(self.extracted_after)

        if self.extracted_before:
            conditions.append("extracted_at <= %s")
            params.append(self.extracted_before)

        # Category filter (OR logic)
        if self.categories:
            placeholders = ",".join(["%s"] * len(self.categories))
            conditions.append(f"category IN ({placeholders})")
            params.extend([cat.value for cat in self.categories])

        # Confidence filters
        if self.min_confidence is not None:
            conditions.append("confidence >= %s")
            params.append(self.min_confidence)

        if self.max_confidence is not None:
            conditions.append("confidence <= %s")
            params.append(self.max_confidence)

        # Channel filter (JSON array search for PostgreSQL JSONB)
        if self.source_channels:
            # PostgreSQL JSONB array contains operator
            channel_conditions = []
            for channel in self.source_channels:
                channel_conditions.append("source_channels @> %s::jsonb")
                params.append(f'["{channel}"]')
            conditions.append(f"({' OR '.join(channel_conditions)})")

        # Source filter
        if self.source_id:
            conditions.append("source_id = %s")
            params.append(self.source_id)

        # Message ID filter
        if self.message_ids:
            placeholders = ",".join(["%s"] * len(self.message_ids))
            conditions.append(f"message_id IN ({placeholders})")
            params.extend(self.message_ids)

        where = " AND ".join(conditions) if conditions else "1=1"
        return where, params

    def to_order_clause(self) -> str:
        """Build PostgreSQL ORDER BY clause."""
        direction = "DESC" if self.order_desc else "ASC"
        return f"{self.order_by} {direction}"

    def to_limit_clause(self) -> tuple[str, list[Any]]:
        """Build PostgreSQL LIMIT/OFFSET clause."""
        if self.limit is None:
            return "", []

        return "LIMIT %s OFFSET %s", [self.limit, self.offset]


@dataclass
class PostgresCandidateQueryCriteria(BaseQueryCriteria):
    """PostgreSQL-specific candidate query criteria using %s placeholders."""

    # Score filters
    min_score: float | None = None
    """Minimum candidate score"""

    max_score: float | None = None
    """Maximum candidate score"""

    # Status filter
    status: str | None = None
    """Filter by processing status (new, processing, llm_ok, llm_fail)"""

    # Channel filter
    channel: str | None = None
    """Filter by specific channel ID"""

    # Date filters
    created_after: datetime | None = None
    """Filter candidates created after this date"""

    created_before: datetime | None = None
    """Filter candidates created before this date"""

    # Feature filters
    has_links: bool | None = None
    """Filter by presence of links"""

    has_anchors: bool | None = None
    """Filter by presence of anchors"""

    # Limits
    limit: int | None = None
    """Maximum number of results"""

    offset: int = 0
    """Offset for pagination"""

    # Ordering
    order_by: str = "score"
    """Column to order by (default: score)"""

    order_desc: bool = True
    """Order descending (default: True for highest scores first)"""

    def to_where_clause(self) -> tuple[str, list[Any]]:
        """Build PostgreSQL WHERE clause with %s parameters."""
        conditions: list[str] = []
        params: list[Any] = []

        # Score filters
        if self.min_score is not None:
            conditions.append("score >= %s")
            params.append(self.min_score)

        if self.max_score is not None:
            conditions.append("score <= %s")
            params.append(self.max_score)

        # Status filter
        if self.status:
            conditions.append("status = %s")
            params.append(self.status)

        # Channel filter
        if self.channel:
            conditions.append("channel = %s")
            params.append(self.channel)

        # Date filters
        if self.created_after:
            conditions.append("ts_dt >= %s")
            params.append(self.created_after)

        if self.created_before:
            conditions.append("ts_dt <= %s")
            params.append(self.created_before)

        # Feature filters (PostgreSQL JSONB array checks)
        if self.has_links is not None:
            if self.has_links:
                conditions.append("links_norm != %s")
                params.append("[]")
            else:
                conditions.append("links_norm = %s")
                params.append("[]")

        if self.has_anchors is not None:
            if self.has_anchors:
                conditions.append("anchors != %s")
                params.append("[]")
            else:
                conditions.append("anchors = %s")
                params.append("[]")

        where = " AND ".join(conditions) if conditions else "1=1"
        return where, params

    def to_order_clause(self) -> str:
        """Build PostgreSQL ORDER BY clause."""
        direction = "DESC" if self.order_desc else "ASC"
        return f"{self.order_by} {direction}"

    def to_limit_clause(self) -> tuple[str, list[Any]]:
        """Build PostgreSQL LIMIT/OFFSET clause."""
        if self.limit is None:
            return "", []

        return "LIMIT %s OFFSET %s", [self.limit, self.offset]
