"""Scoring engine for event candidate selection.

Calculates score based on configurable features:
- Whitelist keywords
- @channel/@here mentions
- Reply count
- Reaction count
- Anchors
- Links
- File attachments
- Bot penalty
"""

import re
from typing import Final

from src.domain.models import (
    ChannelConfig,
    ScoringFeatures,
    SlackMessage,
    TelegramChannelConfig,
)
from src.domain.protocols import MessageRecord as MessageRecordProtocol
from src.domain.scoring_constants import (
    MAX_ANCHOR_SCORE,
    MAX_LINK_SCORE,
    MIN_REACTIONS_FOR_SCORE,
    MIN_REPLIES_FOR_SCORE,
)

# Mention pattern for @channel or @here
CHANNEL_MENTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<!channel>|<!here>|@channel|@here"
)
"""Pattern to match channel-wide mentions (@channel, @here)."""


def extract_features_from_slack_message(
    message: SlackMessage, channel_config: ChannelConfig
) -> ScoringFeatures:
    """Extract scoring features from Slack message.

    Args:
        message: Slack message
        channel_config: Channel configuration

    Returns:
        Feature vector for audit trail

    Example:
        >>> msg = SlackMessage(...)
        >>> config = ChannelConfig(...)
        >>> features = extract_features_from_slack_message(msg, config)
        >>> features.has_keywords
        True
    """
    text_lower = (message.text + " " + message.blocks_text).lower()

    # Check for whitelist keywords
    has_keywords = False
    keyword_count = 0
    for keyword in channel_config.whitelist_keywords:
        if keyword.lower() in text_lower:
            has_keywords = True
            keyword_count += text_lower.count(keyword.lower())

    # Check for @channel/@here mentions
    has_mention = bool(
        CHANNEL_MENTION_PATTERN.search(message.text)
        or CHANNEL_MENTION_PATTERN.search(message.blocks_text)
    )

    # Reply and reaction counts
    reply_count = message.reply_count
    reaction_count = sum(message.reactions.values())

    # Anchors and links
    anchor_count = len(message.anchors)
    link_count = len(message.links_norm)

    # File attachments
    has_files = message.attachments_count > 0 or message.files_count > 0

    return ScoringFeatures(
        has_keywords=has_keywords,
        keyword_count=keyword_count,
        has_mention=has_mention,
        reply_count=reply_count,
        reaction_count=reaction_count,
        anchor_count=anchor_count,
        link_count=link_count,
        has_files=has_files,
        is_bot=message.is_bot,
        channel_name=channel_config.channel_name,
        author_id=message.user,
        bot_id=message.bot_id,
    )


def extract_features(
    message: MessageRecordProtocol,
    channel_config: ChannelConfig | TelegramChannelConfig,
) -> ScoringFeatures:
    """Extract scoring features from any message source.

    This is the universal version that works with MessageRecord protocol
    and any channel configuration type.

    Args:
        message: Message record from any source
        channel_config: Channel configuration (Slack or Telegram)

    Returns:
        Feature vector for audit trail

    Example:
        >>> # Works with any message source
        >>> msg = MessageRecord(...)
        >>> config = ChannelConfig(...)  # or TelegramChannelConfig(...)
        >>> features = extract_features(msg, config)
        >>> features.has_keywords
        True
    """
    text_lower = message.text_norm.lower()

    # Check for whitelist keywords
    has_keywords = False
    keyword_count = 0
    for keyword in channel_config.whitelist_keywords:
        if keyword.lower() in text_lower:
            has_keywords = True
            keyword_count += text_lower.count(keyword.lower())

    # Check for @channel/@here mentions (universal pattern)
    has_mention = bool(CHANNEL_MENTION_PATTERN.search(message.text_norm))

    # Reply and reaction data from extended protocol
    reply_count = getattr(message, "reply_count", 0)
    if getattr(message, "is_reply", False):
        reply_count = max(reply_count, 1)
    reaction_count = getattr(message, "reactions_count", 0)

    # Anchors and links from protocol
    anchor_count = len(message.anchors)
    link_count = len(message.links_norm)

    # File attachments and bot metadata now exposed on protocol
    has_files = getattr(message, "has_file", False)
    is_bot = getattr(message, "is_bot", False)
    bot_id = getattr(message, "bot_id", None)

    return ScoringFeatures(
        has_keywords=has_keywords,
        keyword_count=keyword_count,
        has_mention=has_mention,
        reply_count=reply_count,
        reaction_count=reaction_count,
        anchor_count=anchor_count,
        link_count=link_count,
        has_files=has_files,
        is_bot=is_bot,
        channel_name=channel_config.channel_name,
        author_id="",  # Not available in MessageRecord protocol
        bot_id=bot_id,
    )


def calculate_score(
    features: ScoringFeatures, channel_config: ChannelConfig | TelegramChannelConfig
) -> float:
    """Calculate candidate score based on features and config.

    Args:
        features: Extracted features
        channel_config: Channel configuration with weights

    Returns:
        Total score

    Example:
        >>> features = ScoringFeatures(has_keywords=True, keyword_count=2, ...)
        >>> config = ChannelConfig(keyword_weight=10.0, ...)
        >>> calculate_score(features, config)
        20.0
    """
    score = 0.0
    features.explanations.clear()

    # Positive features
    if features.has_keywords:
        score += channel_config.keyword_weight * features.keyword_count

    if features.has_mention:
        score += channel_config.mention_weight

    if features.reply_count >= MIN_REPLIES_FOR_SCORE:
        score += channel_config.reply_weight

    if features.reaction_count >= MIN_REACTIONS_FOR_SCORE:
        score += channel_config.reaction_weight

    # Anchors (capped at MAX_ANCHOR_SCORE)
    anchor_score = features.anchor_count * channel_config.anchor_weight
    score += min(anchor_score, MAX_ANCHOR_SCORE)

    # Links (capped at MAX_LINK_SCORE)
    link_score = features.link_count * channel_config.link_weight
    score += min(link_score, MAX_LINK_SCORE)

    if features.has_files:
        score += channel_config.file_weight
        if channel_config.file_weight:
            features.explanations.append(
                f"attachments weight +{channel_config.file_weight}"
            )

    # Negative features
    if features.is_bot:
        trusted_ids = set(channel_config.trusted_bots or [])
        bot_identifiers = [features.author_id, features.bot_id]
        trusted_match = next(
            (identifier for identifier in bot_identifiers if identifier in trusted_ids),
            None,
        )
        if trusted_match:
            features.explanations.append("trusted bot bypass")
        else:
            score += channel_config.bot_penalty
            features.explanations.append("bot penalty applied")

    return score


def score_message(
    message: SlackMessage | MessageRecordProtocol,
    channel_config: ChannelConfig | TelegramChannelConfig,
) -> tuple[float, ScoringFeatures]:
    """Score a message for candidate selection.

    Universal function that works with any message source and channel configuration.

    Args:
        message: Message to score (SlackMessage or MessageRecord)
        channel_config: Channel configuration (ChannelConfig or TelegramChannelConfig)

    Returns:
        Tuple of (score, features)

    Example:
        >>> # Works with SlackMessage
        >>> msg = SlackMessage(...)
        >>> config = ChannelConfig(threshold_score=15.0, ...)
        >>> score, features = score_message(msg, config)

        >>> # Works with MessageRecord (Telegram, etc.)
        >>> msg = MessageRecord(...)
        >>> config = TelegramChannelConfig(threshold_score=15.0, ...)
        >>> score, features = score_message(msg, config)
    """
    # Use appropriate feature extraction based on message type
    if isinstance(message, SlackMessage):
        features = extract_features_from_slack_message(message, channel_config)  # type: ignore[arg-type]
    else:
        # MessageRecord protocol
        features = extract_features(message, channel_config)

    score = calculate_score(features, channel_config)
    return score, features


def is_candidate(score: float, threshold: float) -> bool:
    """Check if score meets candidate threshold.

    Args:
        score: Calculated score
        threshold: Minimum threshold

    Returns:
        True if score >= threshold

    Example:
        >>> is_candidate(18.0, 15.0)
        True
        >>> is_candidate(12.0, 15.0)
        False
    """
    return score >= threshold
