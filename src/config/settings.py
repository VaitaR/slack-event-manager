"""Application settings with Pydantic Settings validation.

Secrets (tokens, API keys) are loaded from .env file.
Non-sensitive configuration is loaded from config.yaml.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
        default_factory=lambda: SecretStr("test-slack-token"),
        description="Slack Bot User OAuth Token (from .env)",
    )

    # OpenAI configuration
    openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr("test-openai-key"),
        description="OpenAI API key (from .env)",
    )

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

        # Load monitored channels from config.yaml
        if "channels" in config:
            channels = []
            for ch in config["channels"]:
                channels.append(
                    ChannelConfig(
                        channel_id=ch["channel_id"],
                        channel_name=ch["channel_name"],
                        threshold_score=ch.get("threshold_score", 0.0),
                        keyword_weight=ch.get("keyword_weight", 0.0),
                        whitelist_keywords=ch.get("whitelist_keywords", []),
                    )
                )
            data.setdefault("slack_channels", channels)

        if "digest" in config:
            data.setdefault("digest_max_events", config["digest"].get("max_events", 10))
            data.setdefault(
                "digest_min_confidence", config["digest"].get("min_confidence", 0.7)
            )
            data.setdefault(
                "digest_lookback_hours", config["digest"].get("lookback_hours", 48)
            )
            data.setdefault(
                "digest_category_priorities",
                config["digest"].get(
                    "category_priorities",
                    {
                        "product": 1,
                        "risk": 2,
                        "process": 3,
                        "marketing": 4,
                        "org": 5,
                        "unknown": 6,
                    },
                ),
            )

        if "importance" in config:
            data.setdefault(
                "importance_min_publish_threshold",
                config["importance"].get("min_publish_threshold", 60),
            )
            data.setdefault(
                "importance_high_priority_threshold",
                config["importance"].get("high_priority_threshold", 80),
            )
            data.setdefault(
                "importance_category_base_scores",
                config["importance"].get(
                    "category_base_scores",
                    {
                        "product": 30,
                        "risk": 35,
                        "process": 20,
                        "marketing": 15,
                        "org": 25,
                        "unknown": 10,
                    },
                ),
            )
            data.setdefault(
                "importance_critical_subsystems",
                config["importance"].get(
                    "critical_subsystems",
                    [
                        "authentication",
                        "payment",
                        "trading",
                        "wallet",
                        "database",
                        "api-gateway",
                    ],
                ),
            )

        if "validation" in config:
            data.setdefault(
                "validation_min_confidence",
                config["validation"].get("min_confidence", 0.6),
            )
            data.setdefault(
                "validation_max_title_length",
                config["validation"].get("max_title_length", 140),
            )
            data.setdefault(
                "validation_max_qualifiers",
                config["validation"].get("max_qualifiers", 2),
            )
            data.setdefault(
                "validation_max_links", config["validation"].get("max_links", 3)
            )
            data.setdefault(
                "validation_max_impact_area",
                config["validation"].get("max_impact_area", 3),
            )

        super().__init__(**data)

    # Slack channels (loaded from config.yaml)
    slack_channels: list[ChannelConfig] = Field(
        default_factory=list,
        description="List of channels to monitor (loaded from config.yaml)",
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

    # Digest configuration
    digest_max_events: int | None = Field(
        default=10, description="Maximum events per digest (None = unlimited)"
    )
    digest_min_confidence: float = Field(
        default=0.7, description="Minimum confidence score for digest inclusion"
    )
    digest_lookback_hours: int = Field(
        default=48, description="Default lookback window for digest events"
    )
    digest_category_priorities: dict[str, int] = Field(
        default_factory=lambda: {
            "product": 1,
            "risk": 2,
            "process": 3,
            "marketing": 4,
            "org": 5,
            "unknown": 6,
        },
        description="Category priority mapping for digest sorting",
    )

    # Importance scoring configuration
    importance_min_publish_threshold: int = Field(
        default=60, description="Minimum importance score to publish (0-100)"
    )
    importance_high_priority_threshold: int = Field(
        default=80, description="Threshold for high-priority events"
    )
    importance_category_base_scores: dict[str, int] = Field(
        default_factory=lambda: {
            "product": 30,
            "risk": 35,
            "process": 20,
            "marketing": 15,
            "org": 25,
            "unknown": 10,
        },
        description="Base importance scores by category",
    )
    importance_critical_subsystems: list[str] = Field(
        default_factory=lambda: [
            "authentication",
            "payment",
            "trading",
            "wallet",
            "database",
            "api-gateway",
        ],
        description="Critical subsystems for importance boost",
    )

    # Validation configuration
    validation_min_confidence: float = Field(
        default=0.6, description="Minimum confidence for quality filter"
    )
    validation_max_title_length: int = Field(
        default=140, description="Maximum title length"
    )
    validation_max_qualifiers: int = Field(
        default=2, description="Maximum number of qualifiers"
    )
    validation_max_links: int = Field(default=3, description="Maximum number of links")
    validation_max_impact_area: int = Field(
        default=3, description="Maximum number of impact areas"
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
