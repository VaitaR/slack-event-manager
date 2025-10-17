"""Channel configurations for Slack Event Manager.

DEPRECATED: This file is kept for backward compatibility.
Channel configuration has been moved to config.yaml.

To add/modify channels, edit config.yaml:
```yaml
channels:
  - channel_id: YOUR_CHANNEL_ID
    channel_name: your-channel-name
    threshold_score: 0.0
    keyword_weight: 0.0
    whitelist_keywords:
      - keyword1
      - keyword2
```
"""

import warnings
from typing import TYPE_CHECKING

from src.domain.models import ChannelConfig

if TYPE_CHECKING:
    from src.config.settings import Settings


def _get_settings() -> "Settings":
    """Get settings instance (lazy import to avoid circular dependency)."""
    from src.config.settings import get_settings

    return get_settings()


# Backward compatibility: MONITORED_CHANNELS now loads from config.yaml
# Note: This is a function, not a property, to maintain backward compatibility
def MONITORED_CHANNELS() -> list[ChannelConfig]:  # noqa: N802
    """Get monitored channels from config.yaml.

    DEPRECATED: Use get_settings().slack_channels instead.
    """
    warnings.warn(
        "MONITORED_CHANNELS is deprecated. Use get_settings().slack_channels instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    settings = _get_settings()
    return list(settings.slack_channels)


def get_channel_config(channel_id: str) -> ChannelConfig | None:
    """Get configuration for a specific channel.

    DEPRECATED: Use get_settings().slack_channels instead.

    Args:
        channel_id: Slack channel ID

    Returns:
        ChannelConfig or None if not found
    """
    warnings.warn(
        "get_channel_config() is deprecated. Use get_settings().slack_channels instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    settings = _get_settings()
    for config in settings.slack_channels:
        if config.channel_id == channel_id:
            return config
    return None


def get_all_channel_ids() -> list[str]:
    """Get list of all monitored channel IDs.

    DEPRECATED: Use [ch.channel_id for ch in get_settings().slack_channels] instead.

    Returns:
        List of channel IDs
    """
    warnings.warn(
        "get_all_channel_ids() is deprecated. "
        "Use [ch.channel_id for ch in get_settings().slack_channels] instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return [config.channel_id for config in _get_settings().slack_channels]


def get_all_channels() -> list[ChannelConfig]:
    """Get all monitored channels.

    DEPRECATED: Use get_settings().slack_channels instead.

    Returns:
        List of ChannelConfig objects
    """
    warnings.warn(
        "get_all_channels() is deprecated. Use get_settings().slack_channels instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    settings = _get_settings()
    return list(settings.slack_channels)
