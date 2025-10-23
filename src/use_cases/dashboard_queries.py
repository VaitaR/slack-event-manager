"""Use cases for presentation-layer dashboard queries."""

from src.domain.models import Event, EventCandidate, SlackMessage
from src.domain.protocols import RepositoryProtocol


def fetch_recent_messages(
    repository: RepositoryProtocol, *, limit: int = 100
) -> list[SlackMessage]:
    """Return recent Slack messages for dashboard display."""

    messages = repository.get_recent_slack_messages(limit=limit)
    return list(messages)


def fetch_recent_candidates(
    repository: RepositoryProtocol, *, limit: int = 100
) -> list[EventCandidate]:
    """Return recent event candidates for dashboard display."""

    candidates = repository.get_recent_candidates(limit=limit)
    return list(candidates)


def fetch_recent_events(
    repository: RepositoryProtocol, *, limit: int = 100
) -> list[Event]:
    """Return recent events for dashboard display."""

    events = repository.get_recent_events(limit=limit)
    return list(events)
