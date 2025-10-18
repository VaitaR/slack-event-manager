"""Domain models for Slack Event Manager.

All models use Pydantic v2 for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from src.domain.validation_constants import (
    MAX_IMPACT_AREAS,
    MAX_LINKS,
    MAX_QUALIFIERS,
)


class MessageSource(str, Enum):
    """Message source type."""

    SLACK = "slack"
    TELEGRAM = "telegram"


class MessageSourceConfig(BaseModel):
    """Configuration for a message source (Slack, Telegram, etc.).

    Each source has its own configuration including channels, LLM settings,
    and database table names.
    """

    source_id: MessageSource = Field(..., description="Message source identifier")
    enabled: bool = Field(default=True, description="Whether source is enabled")
    bot_token_env: str = Field(
        default="",
        description="Environment variable name for bot token (optional for source-specific token)",
    )
    raw_table: str = Field(
        ..., description="Database table name for raw messages from this source"
    )
    state_table: str = Field(
        ..., description="Database table name for ingestion state tracking"
    )
    prompt_file: str = Field(default="", description="Path to LLM prompt template file")
    llm_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-source LLM settings (temperature, timeout)",
    )
    channels: list[str] = Field(
        default_factory=list, description="List of channel IDs to monitor"
    )


class CandidateStatus(str, Enum):
    """Status of event candidate processing."""

    NEW = "new"
    LLM_OK = "llm_ok"
    LLM_FAIL = "llm_fail"


class EventCategory(str, Enum):
    """Event category classification."""

    PRODUCT = "product"
    PROCESS = "process"
    MARKETING = "marketing"
    RISK = "risk"
    ORG = "org"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """Action type for event title slots (English only)."""

    LAUNCH = "Launch"
    DEPLOY = "Deploy"
    MIGRATION = "Migration"
    MOVE = "Move"
    ROLLBACK = "Rollback"
    POLICY = "Policy"
    CAMPAIGN = "Campaign"
    WEBINAR = "Webinar"
    INCIDENT = "Incident"
    RCA = "RCA"
    AB_TEST = "A/B Test"
    OTHER = "Other"


class EventStatus(str, Enum):
    """Event lifecycle status."""

    PLANNED = "planned"
    CONFIRMED = "confirmed"
    STARTED = "started"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELED = "canceled"
    ROLLED_BACK = "rolled_back"
    UPDATED = "updated"


class ChangeType(str, Enum):
    """Type of change event represents."""

    LAUNCH = "launch"
    DEPLOY = "deploy"
    MIGRATION = "migration"
    ROLLBACK = "rollback"
    AB_TEST = "ab_test"
    POLICY = "policy"
    CAMPAIGN = "campaign"
    INCIDENT = "incident"
    RCA = "rca"
    OTHER = "other"


class Environment(str, Enum):
    """Deployment environment."""

    PROD = "prod"
    STAGING = "staging"
    DEV = "dev"
    MULTI = "multi"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Severity level for risk events."""

    SEV1 = "sev1"
    SEV2 = "sev2"
    SEV3 = "sev3"
    INFO = "info"
    UNKNOWN = "unknown"


class TimeSource(str, Enum):
    """Source of time information."""

    EXPLICIT = "explicit"
    RELATIVE = "relative"
    TS_FALLBACK = "ts_fallback"


class RelationType(str, Enum):
    """Type of relationship between events."""

    PLANNED_FOR = "planned_for"
    REALIZED_FROM = "realized_from"
    RESCHEDULED_FROM = "rescheduled_from"
    CANCELED_OF = "canceled_of"
    UPDATES = "updates"


class ChannelConfig(BaseModel):
    """Per-channel configuration for scoring and processing."""

    channel_id: str = Field(..., description="Slack channel ID")
    channel_name: str = Field(..., description="Human-readable channel name")
    threshold_score: float = Field(
        default=0.0,
        description="Minimum score for candidate selection (0.0 = process all)",
    )
    whitelist_keywords: list[str] = Field(
        default_factory=list, description="Keywords that boost score"
    )
    trusted_bots: list[str] = Field(
        default_factory=list,
        description="List of bot user IDs that should be treated as trusted",
    )
    keyword_weight: float = Field(default=10.0, description="Weight for keywords")
    mention_weight: float = Field(default=8.0, description="Weight for @channel/@here")
    reply_weight: float = Field(default=5.0, description="Weight for replies")
    reaction_weight: float = Field(default=3.0, description="Weight for reactions")
    anchor_weight: float = Field(default=4.0, description="Weight per anchor")
    link_weight: float = Field(default=2.0, description="Weight per link")
    file_weight: float = Field(default=3.0, description="Weight for attachments")
    bot_penalty: float = Field(default=-15.0, description="Penalty for bot messages")


class SlackMessage(BaseModel):
    """Raw Slack message model."""

    message_id: str = Field(..., description="SHA1 hash of channel|ts")
    channel: str = Field(..., description="Channel ID")
    ts: str = Field(..., description="Slack timestamp")
    ts_dt: datetime = Field(..., description="Timestamp as UTC datetime")
    user: str | None = Field(default=None, description="User ID")
    bot_id: str | None = Field(default=None, description="Slack bot identifier")
    user_real_name: str | None = Field(default=None, description="User real name")
    user_display_name: str | None = Field(default=None, description="User display name")
    user_email: str | None = Field(default=None, description="User email")
    user_profile_image: str | None = Field(
        default=None, description="User profile image URL"
    )
    is_bot: bool = Field(default=False, description="Is message from bot")
    subtype: str | None = Field(default=None, description="Message subtype")
    text: str = Field(default="", description="Raw message text")
    blocks_text: str = Field(default="", description="Text extracted from blocks")
    text_norm: str = Field(default="", description="Normalized text")
    links_raw: list[str] = Field(default_factory=list, description="Raw URLs")
    links_norm: list[str] = Field(default_factory=list, description="Normalized URLs")
    anchors: list[str] = Field(default_factory=list, description="Extracted anchors")
    attachments_count: int = Field(default=0, description="Number of attachments")
    files_count: int = Field(default=0, description="Number of files")
    reactions: dict[str, int] = Field(
        default_factory=dict, description="Emoji reactions with counts"
    )
    total_reactions: int = Field(default=0, description="Total number of reactions")
    reply_count: int = Field(default=0, description="Number of thread replies")
    permalink: str | None = Field(default=None, description="Permanent link to message")
    edited_ts: str | None = Field(default=None, description="Edit timestamp if edited")
    edited_user: str | None = Field(default=None, description="User who edited")
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow, description="Ingestion timestamp"
    )
    source_id: MessageSource = Field(
        default=MessageSource.SLACK, description="Message source (always slack)"
    )


class TelegramMessage(BaseModel):
    """Raw Telegram message model (stub for future implementation)."""

    message_id: str = Field(..., description="Telegram message ID")
    channel: str = Field(..., description="Channel username or ID")
    message_date: datetime = Field(..., description="Message date as UTC datetime")
    sender_id: str | None = Field(default=None, description="Sender user ID")
    sender_name: str | None = Field(default=None, description="Sender display name")
    user: str | None = Field(default=None, description="User ID")
    bot_id: str | None = Field(default=None, description="Bot identifier")
    is_bot: bool = Field(default=False, description="Whether sender is a bot")
    text: str = Field(default="", description="Raw message text")
    text_norm: str = Field(default="", description="Normalized text")
    blocks_text: str = Field(
        default="", description="Formatted text (for scoring compatibility)"
    )
    forward_from_channel: str | None = Field(
        default=None, description="Original channel if forwarded"
    )
    forward_from_message_id: str | None = Field(
        default=None, description="Original message ID if forwarded"
    )
    media_type: str | None = Field(
        default=None, description="Media type (photo, video, document, etc.)"
    )
    links_raw: list[str] = Field(default_factory=list, description="Raw URLs")
    links_norm: list[str] = Field(default_factory=list, description="Normalized URLs")
    anchors: list[str] = Field(default_factory=list, description="Extracted anchors")
    views: int = Field(default=0, description="View count")
    reply_count: int = Field(default=0, description="Reply/comment count")
    reactions: dict[str, int] = Field(default_factory=dict, description="Reactions")
    attachments_count: int = Field(default=0, description="Number of attachments")
    files_count: int = Field(default=0, description="Number of files")
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow, description="Ingestion timestamp"
    )
    source_id: MessageSource = Field(
        default=MessageSource.TELEGRAM, description="Message source (always telegram)"
    )

    @property
    def ts_dt(self) -> datetime:
        """Alias for message_date for compatibility with SlackMessage."""
        return self.message_date


class NormalizedMessage(BaseModel):
    """Message after normalization and extraction."""

    message: SlackMessage
    text_norm: str
    links_norm: list[str]
    anchors: list[str]


class ScoringFeatures(BaseModel):
    """Feature vector for scoring audit trail."""

    has_keywords: bool = False
    keyword_count: int = 0
    has_mention: bool = False
    reply_count: int = 0
    reaction_count: int = 0
    anchor_count: int = 0
    link_count: int = 0
    has_files: bool = False
    is_bot: bool = False
    channel_name: str = ""
    author_id: str | None = None
    bot_id: str | None = None
    explanations: list[str] = Field(
        default_factory=list,
        description="Human-readable scoring explanation entries",
    )


class EventCandidate(BaseModel):
    """Message candidate for event extraction."""

    message_id: str
    channel: str
    ts_dt: datetime
    text_norm: str
    links_norm: list[str]
    anchors: list[str]
    score: float
    status: CandidateStatus = CandidateStatus.NEW
    features: ScoringFeatures
    source_id: MessageSource = Field(
        default=MessageSource.SLACK, description="Message source"
    )


class EventRelation(BaseModel):
    """Relationship between two events."""

    relation_type: RelationType = Field(..., description="Type of relationship")
    target_event_id: UUID = Field(..., description="Target event ID")


class ImportanceScore(BaseModel):
    """Importance scoring breakdown."""

    heuristic_score: int = Field(..., ge=0, le=100, description="Heuristic score (H)")
    llm_score: float = Field(..., ge=0.0, le=1.0, description="LLM score (S)")
    final_score: int = Field(..., ge=0, le=100, description="Final importance")


class Event(BaseModel):
    """Extracted event from Slack message with comprehensive structure.

    Title is rendered from slots, not stored directly.
    """

    # 3.1 Identification
    event_id: UUID = Field(default_factory=uuid4, description="Unique event ID")
    message_id: str = Field(..., description="Source message ID")
    source_channels: list[str] = Field(
        default_factory=list, description="Source channel names"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow, description="Extraction timestamp"
    )

    # 3.2 Title Slots (source of truth for title generation)
    action: ActionType = Field(
        ..., description="Action type from controlled vocabulary"
    )
    object_id: str | None = Field(
        default=None, description="Canonical object ID from registry"
    )
    object_name_raw: str = Field(..., description="Raw object name from text")
    qualifiers: list[str] = Field(
        default_factory=list, max_length=2, description="Max 2 qualifiers"
    )
    stroke: str | None = Field(
        default=None, description="Short semantic stroke from whitelist"
    )
    anchor: str | None = Field(
        default=None, description="Brief identifier (ABC-123, repo#421)"
    )

    # 3.3 Classification & Lifecycle
    category: EventCategory = Field(default=EventCategory.UNKNOWN)
    status: EventStatus = Field(..., description="Lifecycle status")
    change_type: ChangeType = Field(..., description="Type of change")
    environment: Environment = Field(
        default=Environment.UNKNOWN, description="Deployment environment"
    )
    severity: Severity | None = Field(
        default=None, description="Severity for risk events"
    )

    # 3.4 Time Fields
    planned_start: datetime | None = Field(default=None, description="Planned start")
    planned_end: datetime | None = Field(default=None, description="Planned end")
    actual_start: datetime | None = Field(default=None, description="Actual start")
    actual_end: datetime | None = Field(default=None, description="Actual end")
    time_source: TimeSource = Field(..., description="Source of time information")
    time_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in time data"
    )

    # 3.5 Content & Links
    summary: str = Field(..., max_length=320, description="1-3 sentence summary")
    why_it_matters: str | None = Field(
        default=None, max_length=160, description="Why this matters (1 line)"
    )
    links: list[str] = Field(
        default_factory=list, max_length=3, description="Canonicalized URLs (max 3)"
    )
    anchors: list[str] = Field(default_factory=list, description="Jira/PR/Doc IDs")
    impact_area: list[str] = Field(
        default_factory=list, max_length=3, description="Subsystems/components (max 3)"
    )
    impact_type: list[str] = Field(
        default_factory=list,
        description="Impact types (perf_degradation, downtime, etc.)",
    )

    @field_validator("impact_type", mode="before")
    @classmethod
    def validate_impact_type(cls, v: Any) -> list[str]:
        """Convert impact_type to list if it's a string or None."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        if isinstance(v, list):
            return v
        return []

    # 3.6 Quality & Importance
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    importance: int = Field(
        ..., ge=0, le=100, description="Final importance score (0-100)"
    )

    # 3.7 Clusters & Relations
    cluster_key: str = Field(
        ..., description="Common initiative key (without status/time)"
    )
    dedup_key: str = Field(
        ..., description="Specific instance key (with status/time/environment)"
    )
    relations: list[EventRelation] = Field(
        default_factory=list, description="Relationships to other events"
    )

    # 3.8 Source Tracking
    source_id: MessageSource = Field(
        default=MessageSource.SLACK, description="Message source (slack, telegram)"
    )

    @field_validator("qualifiers")
    @classmethod
    def validate_qualifiers(cls, v: list[str]) -> list[str]:
        """Validate max qualifiers."""
        if len(v) > MAX_QUALIFIERS:
            raise ValueError(f"Maximum {MAX_QUALIFIERS} qualifiers allowed")
        return v

    @field_validator("impact_area")
    @classmethod
    def validate_impact_area(cls, v: list[str]) -> list[str]:
        """Validate max impact areas."""
        if len(v) > MAX_IMPACT_AREAS:
            raise ValueError(f"Maximum {MAX_IMPACT_AREAS} impact areas allowed")
        return v

    @field_validator("links")
    @classmethod
    def validate_links(cls, v: list[str]) -> list[str]:
        """Validate max links."""
        if len(v) > MAX_LINKS:
            raise ValueError(f"Maximum {MAX_LINKS} links allowed")
        return v

    @property
    def event_date(self) -> datetime | None:
        """Get primary event date for backward compatibility.

        Returns first non-None value from: actual_start, actual_end, planned_start, planned_end.
        """
        return (
            self.actual_start
            or self.actual_end
            or self.planned_start
            or self.planned_end
        )

    @property
    def title(self) -> str:
        """Generate title from slots for backward compatibility.

        Returns:
            Rendered title string
        """
        from src.services.title_renderer import TitleRenderer

        renderer = TitleRenderer()
        return renderer.render_canonical_title(self)


class LLMEvent(BaseModel):
    """Single event from LLM extraction (before domain conversion).

    Matches new extraction schema with title slots and lifecycle fields.
    """

    # Title slots
    action: str = Field(..., description="Action from vocabulary")
    object_name_raw: str = Field(..., description="Raw object name")
    qualifiers: list[str] = Field(
        default_factory=list, max_length=2, description="Max 2 qualifiers"
    )
    stroke: str | None = Field(default=None, description="Semantic stroke or null")
    anchor: str | None = Field(default=None, description="Brief identifier or null")

    # Classification
    category: EventCategory
    status: str = Field(..., description="Lifecycle status")
    change_type: str = Field(..., description="Type of change")
    environment: str = Field(default="unknown", description="Environment")
    severity: str | None = Field(default=None, description="Severity or null")

    # Time
    planned_start: str | None = Field(default=None, description="ISO8601 or null")
    planned_end: str | None = Field(default=None, description="ISO8601 or null")
    actual_start: str | None = Field(default=None, description="ISO8601 or null")
    actual_end: str | None = Field(default=None, description="ISO8601 or null")
    time_source: str = Field(..., description="explicit|relative|ts_fallback")
    time_confidence: float = Field(..., ge=0.0, le=1.0)

    # Content
    summary: str = Field(..., max_length=320, description="1-3 sentences")
    why_it_matters: str | None = Field(
        default=None, max_length=160, description="Why it matters or null"
    )
    links: list[str] = Field(default_factory=list, max_length=3)
    anchors: list[str] = Field(default_factory=list)
    impact_area: list[str] = Field(default_factory=list, max_length=3)
    impact_type: list[str] = Field(default_factory=list)

    @field_validator("impact_type", mode="before")
    @classmethod
    def validate_impact_type(cls, v: Any) -> list[str]:
        """Convert impact_type to list if it's a string or None."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        if isinstance(v, list):
            return v
        return []

    # Quality
    confidence: float = Field(..., ge=0.0, le=1.0)


class LLMResponse(BaseModel):
    """Structured LLM response for event extraction."""

    is_event: bool = Field(..., description="Does message contain events?")
    overflow_note: str | None = Field(
        default=None, description="Note about events not included (>K)"
    )
    events: list[LLMEvent] = Field(
        default_factory=list, max_length=5, description="Extracted events (max 5)"
    )


class LLMCallMetadata(BaseModel):
    """Metadata for LLM API calls."""

    message_id: str
    prompt_hash: str = Field(..., description="SHA256 of prompt")
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    cached: bool
    ts: datetime = Field(default_factory=datetime.utcnow)


class IngestResult(BaseModel):
    """Result of message ingestion."""

    messages_fetched: int
    messages_saved: int
    channels_processed: list[str]
    errors: list[str] = Field(default_factory=list)


class CandidateResult(BaseModel):
    """Result of candidate building."""

    candidates_created: int
    messages_processed: int
    average_score: float
    max_score: float = 0.0
    min_score: float = 0.0


class ExtractionResult(BaseModel):
    """Result of event extraction."""

    events_extracted: int
    candidates_processed: int
    llm_calls: int
    cache_hits: int
    total_cost_usd: float
    errors: list[str] = Field(default_factory=list)


class DeduplicationResult(BaseModel):
    """Result of deduplication."""

    new_events: int
    merged_events: int
    total_events: int


class DigestResult(BaseModel):
    """Result of digest publication."""

    messages_posted: int
    events_included: int
    channel: str
