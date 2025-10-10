"""Domain models for Slack Event Manager.

All models use Pydantic v2 for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


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


class ChannelConfig(BaseModel):
    """Per-channel configuration for scoring and processing."""

    channel_id: str = Field(..., description="Slack channel ID")
    channel_name: str = Field(..., description="Human-readable channel name")
    threshold_score: float = Field(
        default=0.0, description="Minimum score for candidate selection (0.0 = process all)"
    )
    whitelist_keywords: list[str] = Field(
        default_factory=list, description="Keywords that boost score"
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
    user_real_name: str | None = Field(default=None, description="User real name")
    user_display_name: str | None = Field(default=None, description="User display name")
    user_email: str | None = Field(default=None, description="User email")
    user_profile_image: str | None = Field(default=None, description="User profile image URL")
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


class Event(BaseModel):
    """Extracted event from Slack message."""

    event_id: UUID = Field(default_factory=uuid4)
    version: int = Field(default=1, description="Version for deduplication")
    message_id: str = Field(..., description="Source message ID")
    source_msg_event_idx: int = Field(
        ..., description="Index within source message (0-4)"
    )
    dedup_key: str = Field(..., description="SHA1 for deduplication")
    event_date: datetime = Field(..., description="Event date/time in UTC")
    event_end: datetime | None = Field(
        default=None, description="End date for intervals"
    )
    category: EventCategory = Field(default=EventCategory.UNKNOWN)
    title: str = Field(..., max_length=140, description="Event title")
    summary: str = Field(..., description="1-3 sentence summary")
    impact_area: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list, max_length=3)
    anchors: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source_channels: list[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Validate title is not empty."""
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v


class LLMEvent(BaseModel):
    """Single event from LLM extraction (before domain conversion)."""

    title: str = Field(..., max_length=140)
    summary: str = Field(..., description="1-3 sentences")
    category: EventCategory
    event_date: str = Field(..., description="ISO8601 datetime string")
    event_end: str | None = Field(default=None, description="ISO8601 or null")
    impact_area: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list, max_length=3)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_channels: list[str] = Field(default_factory=list)


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

