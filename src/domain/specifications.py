"""Specification pattern for domain filtering and validation.

Specifications encapsulate business rules and conditions that can be:
- Combined with logical operators (AND, OR, NOT)
- Reused across different contexts
- Tested independently
- Documented as business logic
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Generic, TypeVar

from src.domain.models import Event, EventCandidate, SlackMessage

T = TypeVar("T")


class Specification(ABC, Generic[T]):
    """Base specification interface.

    A specification represents a business rule that can be checked
    against a candidate object. Specifications can be combined using
    logical operators to create complex conditions.
    """

    @abstractmethod
    def is_satisfied_by(self, candidate: T) -> bool:
        """Check if candidate satisfies this specification.

        Args:
            candidate: Object to check

        Returns:
            True if candidate satisfies the specification
        """
        pass

    def and_(self, other: "Specification[T]") -> "AndSpecification[T]":
        """Combine with AND logic."""
        return AndSpecification(self, other)

    def or_(self, other: "Specification[T]") -> "OrSpecification[T]":
        """Combine with OR logic."""
        return OrSpecification(self, other)

    def not_(self) -> "NotSpecification[T]":
        """Negate this specification."""
        return NotSpecification(self)


class AndSpecification(Specification[T]):
    """AND combination of two specifications."""

    def __init__(self, left: Specification[T], right: Specification[T]) -> None:
        self.left = left
        self.right = right

    def is_satisfied_by(self, candidate: T) -> bool:
        """Both specifications must be satisfied."""
        return self.left.is_satisfied_by(candidate) and self.right.is_satisfied_by(
            candidate
        )


class OrSpecification(Specification[T]):
    """OR combination of two specifications."""

    def __init__(self, left: Specification[T], right: Specification[T]) -> None:
        self.left = left
        self.right = right

    def is_satisfied_by(self, candidate: T) -> bool:
        """At least one specification must be satisfied."""
        return self.left.is_satisfied_by(candidate) or self.right.is_satisfied_by(
            candidate
        )


class NotSpecification(Specification[T]):
    """NOT negation of a specification."""

    def __init__(self, spec: Specification[T]) -> None:
        self.spec = spec

    def is_satisfied_by(self, candidate: T) -> bool:
        """Specification must NOT be satisfied."""
        return not self.spec.is_satisfied_by(candidate)


# Event Candidate Specifications


class ScoreAboveThresholdSpec(Specification[EventCandidate]):
    """Specification for candidates with score above threshold."""

    def __init__(self, threshold: float) -> None:
        """Initialize with score threshold.

        Args:
            threshold: Minimum score required
        """
        self.threshold = threshold

    def is_satisfied_by(self, candidate: EventCandidate) -> bool:
        """Check if candidate score meets threshold."""
        return candidate.score >= self.threshold


class CandidateHasStatusSpec(Specification[EventCandidate]):
    """Specification for candidates with specific status."""

    def __init__(self, status: str) -> None:
        """Initialize with required status.

        Args:
            status: Required status value (e.g., 'new', 'llm_ok')
        """
        self.status = status

    def is_satisfied_by(self, candidate: EventCandidate) -> bool:
        """Check if candidate has required status."""
        return candidate.status.value == self.status


class CandidateHasLinksSpec(Specification[EventCandidate]):
    """Specification for candidates with links."""

    def is_satisfied_by(self, candidate: EventCandidate) -> bool:
        """Check if candidate has any links."""
        return len(candidate.links_norm) > 0


class CandidateHasAnchorsSpec(Specification[EventCandidate]):
    """Specification for candidates with anchors."""

    def is_satisfied_by(self, candidate: EventCandidate) -> bool:
        """Check if candidate has any anchors."""
        return len(candidate.anchors) > 0


# Event Specifications


class EventInDateRangeSpec(Specification[Event]):
    """Specification for events within a date range."""

    def __init__(self, start_date: datetime, end_date: datetime) -> None:
        """Initialize with date range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)
        """
        self.start_date = start_date
        self.end_date = end_date

    def is_satisfied_by(self, event: Event) -> bool:
        """Check if event date is within range."""
        primary_time = (
            event.actual_start
            or event.actual_end
            or event.planned_start
            or event.planned_end
            or event.extracted_at
        )
        return self.start_date <= primary_time <= self.end_date


class EventRecentSpec(Specification[Event]):
    """Specification for events created recently."""

    def __init__(self, days_ago: int) -> None:
        """Initialize with lookback period.

        Args:
            days_ago: Number of days to look back
        """
        self.cutoff_date = datetime.utcnow() - timedelta(days=days_ago)

    def is_satisfied_by(self, event: Event) -> bool:
        """Check if event was extracted recently."""
        return event.extracted_at >= self.cutoff_date


class EventHasCategorySpec(Specification[Event]):
    """Specification for events with specific category."""

    def __init__(self, category: str) -> None:
        """Initialize with required category.

        Args:
            category: Required category (e.g., 'product', 'risk')
        """
        self.category = category

    def is_satisfied_by(self, event: Event) -> bool:
        """Check if event has required category."""
        return event.category == self.category


class EventHighConfidenceSpec(Specification[Event]):
    """Specification for high-confidence events."""

    def __init__(self, min_confidence: float = 0.8) -> None:
        """Initialize with confidence threshold.

        Args:
            min_confidence: Minimum confidence score (default: 0.8)
        """
        self.min_confidence = min_confidence

    def is_satisfied_by(self, event: Event) -> bool:
        """Check if event confidence is above threshold."""
        return event.confidence >= self.min_confidence


class EventFromChannelSpec(Specification[Event]):
    """Specification for events from specific channel."""

    def __init__(self, channel_name: str) -> None:
        """Initialize with channel name.

        Args:
            channel_name: Required channel name
        """
        self.channel_name = channel_name

    def is_satisfied_by(self, event: Event) -> bool:
        """Check if event is from required channel."""
        return self.channel_name in event.source_channels


# Message Specifications


class MessageFromBotSpec(Specification[SlackMessage]):
    """Specification for bot messages."""

    def is_satisfied_by(self, message: SlackMessage) -> bool:
        """Check if message is from a bot."""
        return message.is_bot


class MessageHasReactionsSpec(Specification[SlackMessage]):
    """Specification for messages with reactions."""

    def __init__(self, min_reactions: int = 1) -> None:
        """Initialize with minimum reaction count.

        Args:
            min_reactions: Minimum number of reactions required
        """
        self.min_reactions = min_reactions

    def is_satisfied_by(self, message: SlackMessage) -> bool:
        """Check if message has enough reactions."""
        if not message.reactions:
            return False
        total_reactions = sum(message.reactions.values())
        return total_reactions >= self.min_reactions


class MessageInChannelSpec(Specification[SlackMessage]):
    """Specification for messages from specific channel."""

    def __init__(self, channel_id: str) -> None:
        """Initialize with channel ID.

        Args:
            channel_id: Required Slack channel ID
        """
        self.channel_id = channel_id

    def is_satisfied_by(self, message: SlackMessage) -> bool:
        """Check if message is from required channel."""
        return message.channel == self.channel_id


# Common filter combinations (factory methods)


def high_priority_candidates(threshold: float = 15.0) -> Specification[EventCandidate]:
    """Factory for high-priority candidate specification.

    High priority means:
    - Score above threshold
    - Status is 'new' (not yet processed)
    - Has links or anchors (better quality)

    Args:
        threshold: Minimum score threshold

    Returns:
        Combined specification for high-priority candidates

    Example:
        >>> spec = high_priority_candidates(15.0)
        >>> high_priority = [c for c in candidates if spec.is_satisfied_by(c)]
    """
    return (
        ScoreAboveThresholdSpec(threshold)
        .and_(CandidateHasStatusSpec("new"))
        .and_(CandidateHasLinksSpec().or_(CandidateHasAnchorsSpec()))
    )


def recent_high_confidence_events(
    days_ago: int = 7, min_confidence: float = 0.8
) -> Specification[Event]:
    """Factory for recent high-confidence event specification.

    Args:
        days_ago: Number of days to look back
        min_confidence: Minimum confidence threshold

    Returns:
        Combined specification for recent high-confidence events

    Example:
        >>> spec = recent_high_confidence_events(days_ago=7, min_confidence=0.8)
        >>> events = [e for e in all_events if spec.is_satisfied_by(e)]
    """
    return EventRecentSpec(days_ago).and_(EventHighConfidenceSpec(min_confidence))


def product_events_for_digest() -> Specification[Event]:
    """Factory for product events suitable for digest.

    Returns:
        Specification for product category events with high confidence

    Example:
        >>> spec = product_events_for_digest()
        >>> digest_events = [e for e in events if spec.is_satisfied_by(e)]
    """
    return EventHasCategorySpec("product").and_(EventHighConfidenceSpec(0.7))
