"""Scoring constants and limits for event candidate selection.

This module defines scoring limits and thresholds used in the scoring engine.
The actual weights are configured per-channel in ChannelConfig, but the limits
and business rules are defined here.
"""

from typing import Final

# Scoring limits
MAX_ANCHOR_SCORE: Final[float] = 12.0
"""Maximum total score contribution from anchors.

Business rule: Even if a message has many anchors (Jira tickets, GitHub issues),
we cap the score at +12 to prevent over-weighting. With default anchor_weight=4.0,
this means max 3 anchors are counted (3 × 4 = 12).

Reasoning: More than 3 anchors doesn't necessarily mean higher importance.
"""

MAX_LINK_SCORE: Final[float] = 6.0
"""Maximum total score contribution from links.

Business rule: Even if a message has many links, we cap the score at +6.
With default link_weight=2.0, this means max 3 links are counted (3 × 2 = 6).

Reasoning: Excessive links might indicate spam or automated messages.
"""

# Reaction thresholds
MIN_REACTIONS_FOR_SCORE: Final[int] = 2
"""Minimum reactions required to add reaction score.

Business rule: A message needs at least 2 reactions to be considered
"community-validated". Single reactions can be accidental or from the author.

Example:
    - 1 reaction → no score
    - 2+ reactions → add reaction_weight to score
"""

MIN_REPLIES_FOR_SCORE: Final[int] = 1
"""Minimum replies required to add reply score.

Business rule: Any replies (≥1) indicate the message sparked discussion
or required clarification, suggesting it's important.

Example:
    - 0 replies → no score
    - 1+ replies → add reply_weight to score
"""

# Default scoring weights (reference values, actual values in ChannelConfig)
DEFAULT_KEYWORD_WEIGHT: Final[float] = 10.0
"""Default weight per whitelist keyword occurrence."""

DEFAULT_MENTION_WEIGHT: Final[float] = 8.0
"""Default weight for @channel or @here mentions."""

DEFAULT_REPLY_WEIGHT: Final[float] = 5.0
"""Default weight when message has replies."""

DEFAULT_REACTION_WEIGHT: Final[float] = 3.0
"""Default weight when message has reactions."""

DEFAULT_ANCHOR_WEIGHT: Final[float] = 4.0
"""Default weight per anchor (Jira, GitHub, etc.)."""

DEFAULT_LINK_WEIGHT: Final[float] = 2.0
"""Default weight per link/URL."""

DEFAULT_FILE_WEIGHT: Final[float] = 3.0
"""Default weight for file attachments."""

DEFAULT_BOT_PENALTY: Final[float] = -15.0
"""Default penalty for bot messages (unless whitelisted)."""

DEFAULT_THRESHOLD_SCORE: Final[float] = 0.0
"""Default threshold - process all messages by default.

Can be raised to 10-15 in production to filter low-quality messages.
"""


