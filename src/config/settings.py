"""Application settings with Pydantic Settings validation.

Secrets (tokens, API keys) are loaded from .env file.
Non-sensitive configuration is loaded from config.yaml.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.channels import MONITORED_CHANNELS
from src.domain.deduplication_constants import (
    DEFAULT_DATE_WINDOW_HOURS,
    DEFAULT_MESSAGE_LOOKBACK_DAYS,
    DEFAULT_TITLE_SIMILARITY,
)
from src.domain.models import ChannelConfig
from src.domain.scoring_constants import DEFAULT_THRESHOLD_SCORE


def load_config_yaml() -> dict[str, Any]:
    """Load configuration from config.yaml file.

    Returns:
        Configuration dictionary
    """
    config_path = Path("config.yaml")
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """Application settings.

    Secrets are loaded from .env file.
    Non-sensitive config is loaded from config.yaml with fallback to defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === SECRETS (from .env) ===

    # Slack configuration
    slack_bot_token: SecretStr = Field(
        ..., description="Slack Bot User OAuth Token (from .env)"
    )

    # OpenAI configuration
    openai_api_key: SecretStr = Field(..., description="OpenAI API key (from .env)")

    # === NON-SENSITIVE CONFIG (from config.yaml or defaults) ===

    def __init__(self, **data: Any):
        """Initialize settings with config.yaml defaults."""
        config = load_config_yaml()

        # Apply config.yaml values as defaults (can be overridden by .env)
        if "llm" in config:
            data.setdefault("llm_model", config["llm"].get("model", "gpt-5-nano"))
            data.setdefault("llm_temperature", config["llm"].get("temperature", 1.0))
            data.setdefault(
                "llm_timeout_seconds", config["llm"].get("timeout_seconds", 120)
            )
            data.setdefault(
                "llm_daily_budget_usd", config["llm"].get("daily_budget_usd", 10.0)
            )
            data.setdefault(
                "llm_max_events_per_msg", config["llm"].get("max_events_per_msg", 5)
            )

        if "database" in config:
            data.setdefault(
                "db_path", config["database"].get("path", "data/slack_events.db")
            )

        if "slack" in config:
            data.setdefault(
                "slack_digest_channel_id",
                config["slack"].get("digest_channel_id", "YOUR_DIGEST_CHANNEL_ID"),
            )
            data.setdefault(
                "lookback_hours_default",
                config["slack"].get("lookback_hours_default", 24),
            )

        if "processing" in config:
            data.setdefault(
                "tz_default", config["processing"].get("tz_default", "Europe/Amsterdam")
            )
            data.setdefault(
                "threshold_score_default",
                config["processing"].get(
                    "threshold_score_default", DEFAULT_THRESHOLD_SCORE
                ),
            )

        if "deduplication" in config:
            data.setdefault(
                "dedup_date_window_hours",
                config["deduplication"].get(
                    "date_window_hours", DEFAULT_DATE_WINDOW_HOURS
                ),
            )
            data.setdefault(
                "dedup_title_similarity",
                config["deduplication"].get(
                    "title_similarity", DEFAULT_TITLE_SIMILARITY
                ),
            )
            data.setdefault(
                "dedup_message_lookback_days",
                config["deduplication"].get(
                    "message_lookback_days", DEFAULT_MESSAGE_LOOKBACK_DAYS
                ),
            )

        if "logging" in config:
            data.setdefault("log_level", config["logging"].get("level", "INFO"))

        super().__init__(**data)

    # Slack channels (from code, not config file)
    slack_channels: list[ChannelConfig] = Field(
        default=MONITORED_CHANNELS,
        description="List of channels to monitor",
    )
    slack_digest_channel_id: str = Field(
        default="YOUR_DIGEST_CHANNEL_ID",
        description="Channel for daily digest publication",
    )

    # LLM configuration
    llm_model: str = Field(default="gpt-5-nano", description="OpenAI model to use")
    llm_daily_budget_usd: float = Field(
        default=10.0, description="Daily LLM budget in USD"
    )
    llm_max_events_per_msg: int = Field(
        default=5, description="Maximum events to extract per message"
    )
    llm_temperature: float = Field(
        default=1.0, description="LLM temperature (1.0 for gpt-5-nano)"
    )
    llm_timeout_seconds: int = Field(default=120, description="LLM request timeout")

    # Database configuration
    db_path: str = Field(
        default="data/slack_events.db", description="SQLite database path"
    )

    # Processing configuration
    tz_default: str = Field(
        default="Europe/Amsterdam", description="Default timezone for date parsing"
    )
    threshold_score_default: float = Field(
        default=DEFAULT_THRESHOLD_SCORE,
        description="Default candidate threshold (0.0 = process all)",
    )
    dedup_date_window_hours: int = Field(
        default=DEFAULT_DATE_WINDOW_HOURS,
        description="Date window for event deduplication (hours)",
    )
    dedup_title_similarity: float = Field(
        default=DEFAULT_TITLE_SIMILARITY,
        description="Minimum fuzzy title similarity for merge (0.0-1.0)",
    )
    dedup_message_lookback_days: int = Field(
        default=DEFAULT_MESSAGE_LOOKBACK_DAYS,
        description="Days to look back when processing messages for deduplication",
    )
    lookback_hours_default: int = Field(
        default=24, description="Default lookback for message ingestion (hours)"
    )

    # Observability
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("slack_channels", mode="before")
    @classmethod
    def validate_slack_channels(cls, v: Any) -> list[ChannelConfig]:
        """Validate that channels is a list of ChannelConfig objects."""
        if isinstance(v, list):
            # Already parsed or list of dicts
            if v and isinstance(v[0], ChannelConfig):
                return v
            return [ChannelConfig(**ch) for ch in v]
        return v

    def get_channel_config(self, channel_id: str) -> ChannelConfig | None:
        """Get configuration for specific channel.

        Args:
            channel_id: Slack channel ID

        Returns:
            Channel config or None if not found
        """
        from src.config.channels import get_channel_config as get_config

        return get_config(channel_id)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
