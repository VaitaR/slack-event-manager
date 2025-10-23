"""Protocol definitions for dependency inversion.

These abstract interfaces define contracts that adapters must implement.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from src.domain.models import (
    Event,
    EventCandidate,
    LLMCallMetadata,
    LLMResponse,
    MessageSource,
    SlackMessage,
    TelegramMessage,
)

if TYPE_CHECKING:
    from src.adapters.query_builders import CandidateQueryCriteria, EventQueryCriteria


class MessageRecord(Protocol):
    """Protocol for source-agnostic message representation for scoring.

    This protocol defines the minimal interface that any message source
    (Slack, Telegram, etc.) must provide for candidate scoring.
    """

    message_id: str
    channel: str
    ts_dt: datetime
    text_norm: str
    links_norm: list[str]
    anchors: list[str]
    source_id: MessageSource


class MessageClientProtocol(Protocol):
    """Generic protocol for message source clients (Slack, Telegram, etc.)."""

    def fetch_messages(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch messages from source channel.

        Args:
            channel_id: Channel ID or username
            oldest_ts: Oldest timestamp/message_id to fetch (inclusive)
            latest_ts: Latest timestamp/message_id to fetch (inclusive)
            limit: Maximum messages to fetch (None = unlimited)

        Returns:
            List of raw message dictionaries

        Raises:
            Exception: On API communication errors
        """
        ...

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        """Get user/sender information by ID (with optional caching).

        Args:
            user_id: User ID

        Returns:
            User info dictionary

        Raises:
            Exception: On API communication errors
        """
        ...


class SlackClientProtocol(Protocol):
    """Protocol for Slack API interactions."""

    def fetch_messages(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int | None = None,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Slack channel.

        Args:
            channel_id: Slack channel ID
            oldest_ts: Oldest timestamp to fetch (inclusive)
            latest_ts: Latest timestamp to fetch (inclusive)
            limit: Maximum total messages to return (None = unlimited)
            page_size: Optional per-page limit override

        Returns:
            List of raw Slack message dictionaries

        Raises:
            SlackAPIError: On API communication errors
            RateLimitError: On rate limit exceeded
        """
        ...

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        """Get user information by ID (with caching).

        Args:
            user_id: Slack user ID

        Returns:
            User info dictionary

        Raises:
            SlackAPIError: On API communication errors
        """
        ...

    def post_message(self, channel_id: str, blocks: list[dict[str, Any]]) -> str:
        """Post message with Block Kit to channel.

        Args:
            channel_id: Target channel ID
            blocks: Slack Block Kit blocks

        Returns:
            Message timestamp

        Raises:
            SlackAPIError: On API communication errors
        """
        ...


class RepositoryProtocol(Protocol):
    """Protocol for data storage operations."""

    def save_messages(self, messages: list[SlackMessage]) -> int:
        """Save messages to storage (idempotent upsert).

        Args:
            messages: List of slack messages

        Returns:
            Number of messages saved

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def get_watermark(self, channel: str) -> str | None:
        """Get committed watermark timestamp for channel.

        Args:
            channel: Channel ID

        Returns:
            Last committed timestamp or None

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def update_watermark(self, channel: str, ts: str) -> None:
        """Update committed watermark for channel.

        Args:
            channel: Channel ID
            ts: New watermark timestamp

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def get_new_messages_for_candidates(self) -> list[SlackMessage]:
        """Get messages not yet in candidates table.

        Returns:
            List of messages to process

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def get_new_messages_for_candidates_by_source(
        self, source_id: MessageSource
    ) -> list[MessageRecord]:
        """Get messages not yet in candidates table for a specific source.

        This method is source-agnostic and returns messages that implement
        the MessageRecord protocol, allowing scoring logic to work with any source.

        Args:
            source_id: Message source (SLACK, TELEGRAM, etc.)

        Returns:
            List of messages implementing MessageRecord protocol

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def save_candidates(self, candidates: list[EventCandidate]) -> int:
        """Save event candidates (idempotent).

        Args:
            candidates: List of candidates

        Returns:
            Number of candidates saved

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def get_candidates_by_source(
        self, source_id: MessageSource
    ) -> list[EventCandidate]:
        """Get candidates for a specific message source."""

        ...

    def get_candidates_for_extraction(
        self,
        batch_size: int | None = 50,
        min_score: float | None = None,
        source_id: MessageSource | None = None,
    ) -> list[EventCandidate]:
        """Get candidates ready for LLM extraction.

        Args:
            batch_size: Maximum candidates to return
            min_score: Minimum score filter (for budget control)
            source_id: Filter by message source (None = all sources)

        Returns:
            List of candidates ordered by score DESC

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def get_recent_slack_messages(self, limit: int = 100) -> list[SlackMessage]:
        """Get most recent Slack messages for presentation use."""

        ...

    def get_recent_candidates(self, limit: int = 100) -> list[EventCandidate]:
        """Get most recent event candidates for presentation use."""

        ...

    def get_recent_events(self, limit: int = 100) -> list[Event]:
        """Get most recently extracted events for presentation use."""

        ...

    def query_candidates(
        self, criteria: "CandidateQueryCriteria"
    ) -> list[EventCandidate]:
        """Query event candidates using structured criteria.

        Args:
            criteria: Query builder criteria object

        Returns:
            List of event candidates matching criteria

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def update_candidate_status(self, message_id: str, status: str) -> None:
        """Update candidate processing status.

        Args:
            message_id: Message ID
            status: New status (llm_ok, llm_fail)

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def save_events(self, events: list[Event]) -> int:
        """Save events with versioning (upsert by dedup_key).

        Args:
            events: List of events

        Returns:
            Number of events saved

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def get_events_by_source(self, source_id: MessageSource) -> list[Event]:
        """Get events filtered by message source."""

        ...

    def get_events_in_window(self, start_dt: datetime, end_dt: datetime) -> list[Event]:
        """Get events within date window.

        Args:
            start_dt: Start datetime (UTC)
            end_dt: End datetime (UTC)

        Returns:
            List of events

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def get_events_in_window_filtered(
        self,
        start_dt: datetime,
        end_dt: datetime,
        min_confidence: float = 0.0,
        max_events: int | None = None,
    ) -> list[Event]:
        """Get filtered events within date window.

        Args:
            start_dt: Start datetime (UTC)
            end_dt: End datetime (UTC)
            min_confidence: Minimum confidence threshold
            max_events: Optional maximum number of events

        Returns:
            List of filtered events

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def query_events(self, criteria: "EventQueryCriteria") -> list[Event]:
        """Query events using structured criteria.

        Args:
            criteria: Query builder criteria object

        Returns:
            List of events matching criteria

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def save_llm_call(self, metadata: LLMCallMetadata) -> None:
        """Save LLM call metadata.

        Args:
            metadata: Call metadata

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def get_daily_llm_cost(self, date: datetime) -> float:
        """Get total LLM cost for a day.

        Args:
            date: Date to check

        Returns:
            Total cost in USD

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def get_cached_llm_response(self, prompt_hash: str) -> str | None:
        """Get cached LLM response by prompt hash.

        Args:
            prompt_hash: SHA256 hash of prompt

        Returns:
            Cached JSON response or None

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def save_telegram_messages(self, messages: list[TelegramMessage]) -> int:
        """Save Telegram messages for candidate generation."""

        ...

    def get_telegram_messages(
        self, channel: str | None = None, limit: int = 100
    ) -> list[TelegramMessage]:
        """Get Telegram messages for debugging or validation."""

        ...

    def get_last_processed_ts(
        self, channel: str, source_id: MessageSource | None = None
    ) -> float | None:
        """Get last processed timestamp for channel (source-specific).

        Args:
            channel: Channel ID
            source_id: Optional source identifier for multi-source tracking

        Returns:
            Last processed timestamp or None

        Raises:
            RepositoryError: On storage errors
        """

    def get_last_processed_message_id(
        self, channel: str, source_id: MessageSource | None = None
    ) -> str | None:
        """Get last processed message ID for Telegram channel.

        Args:
            channel: Channel ID (Telegram username)
            source_id: Message source (must be TELEGRAM for this method)

        Returns:
            Last processed message ID or None if first run

        Raises:
            RepositoryError: On storage errors
        """

    def update_last_processed_message_id(
        self, channel: str, message_id: str, source_id: MessageSource | None = None
    ) -> None:
        """Update last processed message ID for Telegram channel.

        Args:
            channel: Channel ID (Telegram username)
            message_id: Message ID to set
            source_id: Message source (must be TELEGRAM for this method)

        Raises:
            RepositoryError: On storage errors
        """
        ...

    def update_last_processed_ts(
        self, channel: str, ts: float, source_id: MessageSource | None = None
    ) -> None:
        """Update last processed timestamp for channel (source-specific).

        Args:
            channel: Channel ID
            ts: New timestamp
            source_id: Optional source identifier for multi-source tracking

        Raises:
            RepositoryError: On storage errors
        """
        ...


class LLMClientProtocol(Protocol):
    """Protocol for LLM API interactions."""

    def extract_events(
        self, text: str, links: list[str], message_ts_dt: datetime
    ) -> LLMResponse:
        """Extract events from message text using LLM.

        Args:
            text: Normalized message text
            links: Top 3 most relevant links
            message_ts_dt: Message timestamp for date resolution fallback

        Returns:
            Structured LLM response

        Raises:
            LLMAPIError: On API communication errors
            BudgetExceededError: When daily budget exceeded
            ValidationError: On response validation failure
        """
        ...

    def get_call_metadata(self) -> LLMCallMetadata:
        """Get metadata for last LLM call.

        Returns:
            Call metadata including tokens and cost

        Raises:
            RuntimeError: If no call has been made
        """
        ...
