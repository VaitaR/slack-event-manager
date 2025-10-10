"""Channel configurations for Slack Event Manager.

This file contains the list of Slack channels to monitor and their specific settings.
"""

from src.domain.models import ChannelConfig

# List of channels to monitor
MONITORED_CHANNELS = [
    ChannelConfig(
        channel_id="C06B5NJLY4B",
        channel_name="product-alerts-test",
        threshold_score=0.0,  # Process all messages by default
        keyword_weight=0.0,  # Disable keyword filtering
        whitelist_keywords=[
            "release", "deploy", "launch", "update", "new", "announcement",
            "scheduled", "security", "metrics", "maintenance", "incident",
            "daily", "weekly", "monthly", "report", "summary", "alert"
        ]
    ),
    ChannelConfig(
        channel_id="C04V0TK7UG6",
        channel_name="releases",
        threshold_score=0.0,  # Process all messages by default
        keyword_weight=0.0,  # Disable keyword filtering
        whitelist_keywords=[
            "release", "deploy", "launch", "update", "new", "announcement",
            "scheduled", "security", "metrics", "maintenance", "incident",
            "daily", "weekly", "monthly", "report", "summary", "alert"
        ]
    ),
        ChannelConfig(
        channel_id="C0770K7FV43",
        channel_name="t-onboarding-activation",
        threshold_score=0.0,  # Process all messages by default
        keyword_weight=0.0,  # Disable keyword filtering
        whitelist_keywords=[
            "release", "deploy", "launch", "update", "new", "announcement",
            "scheduled", "security", "metrics", "maintenance", "incident",
            "daily", "weekly", "monthly", "report", "summary", "alert"
        ]
    ),
]

def get_channel_config(channel_id: str) -> ChannelConfig | None:
    """Get configuration for a specific channel.

    Args:
        channel_id: Slack channel ID

    Returns:
        ChannelConfig or None if not found
    """
    for config in MONITORED_CHANNELS:
        if config.channel_id == channel_id:
            return config
    return None

def get_all_channel_ids() -> list[str]:
    """Get list of all monitored channel IDs.

    Returns:
        List of channel IDs
    """
    return [config.channel_id for config in MONITORED_CHANNELS]

def get_all_channels() -> list[ChannelConfig]:
    """Get all monitored channels.

    Returns:
        List of ChannelConfig objects
    """
    return MONITORED_CHANNELS.copy()
