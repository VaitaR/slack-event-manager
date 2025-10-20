"""Application settings with Pydantic Settings validation.

Secrets (tokens, API keys) are loaded from .env file.
Non-sensitive configuration is loaded from config.yaml and config/*.yaml files.
All configs are automatically merged and validated against JSON schemas.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.deduplication_constants import (
    DEFAULT_DATE_WINDOW_HOURS,
    DEFAULT_MESSAGE_LOOKBACK_DAYS,
    DEFAULT_TITLE_SIMILARITY,
)
from src.domain.models import (
    ChannelConfig,
    MessageSource,
    MessageSourceConfig,
    TelegramChannelConfig,
)
from src.domain.scoring_constants import DEFAULT_THRESHOLD_SCORE

logger = logging.getLogger(__name__)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary to merge into base (takes precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_schema(schema_name: str) -> dict[str, Any]:
    """Load JSON Schema from config/schemas/.

    Args:
        schema_name: Schema name without extension (e.g., "main")

    Returns:
        JSON Schema dictionary or empty dict if not found
    """
    schema_path = Path("config/schemas") / f"{schema_name}.schema.json"
    if not schema_path.exists():
        return {}

    try:
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load schema {schema_name}: {e}")
        return {}


def validate_config_section(
    config: dict[str, Any], schema_name: str, file_path: str = ""
) -> None:
    """Validate config section against JSON Schema.

    Args:
        config: Configuration dictionary to validate
        schema_name: Name of schema to validate against
        file_path: Optional file path for error messages

    Raises:
        ValueError: If validation fails
    """
    schema = load_schema(schema_name)
    if not schema:
        # No schema available, skip validation
        return

    try:
        validate(instance=config, schema=schema)
        logger.debug(f"Config validation passed for {schema_name}")
    except JSONSchemaValidationError as e:
        error_msg = f"Config validation failed for {schema_name}"
        if file_path:
            error_msg += f" (file: {file_path})"
        error_msg += f": {e.message}"
        raise ValueError(error_msg) from e


def load_all_configs() -> dict[str, Any]:
    """Load and merge all YAML configs from config/ directory.

    Loading order (later overrides earlier):
    1. config/main.yaml (main config)
    2. config/object_registry.yaml
    3. config/channels.yaml
    4. All other config/*.yaml files (sorted alphabetically)

    Each config is validated against its JSON Schema if available.

    Returns:
        Merged configuration dictionary
    """
    merged_config: dict[str, Any] = {}

    # 1. Load main config from config/main.yaml
    main_path = Path("config/main.yaml")
    if main_path.exists():
        try:
            with open(main_path, encoding="utf-8") as f:
                main_config = yaml.safe_load(f) or {}
                validate_config_section(main_config, "main", str(main_path))
                merged_config = main_config
                logger.debug(f"Loaded config from {main_path}")
        except (yaml.YAMLError, OSError) as e:
            logger.warning(f"Failed to load {main_path}: {e}")

    # 2. Load config directory files
    config_dir = Path("config")
    yaml_files: list[Path] = []
    if config_dir.exists() and config_dir.is_dir():
        # Get all YAML files, excluding already loaded main.yaml
        yaml_files = sorted(
            [f for f in config_dir.glob("*.yaml") if f.name != "main.yaml"]
        )

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    file_config = yaml.safe_load(f) or {}

                    # Determine schema name from filename
                    schema_name = yaml_file.stem  # e.g., "object_registry"
                    validate_config_section(file_config, schema_name, str(yaml_file))

                    # Deep merge
                    merged_config = deep_merge(merged_config, file_config)
                    logger.debug(f"Loaded and merged config from {yaml_file}")
            except (yaml.YAMLError, OSError) as e:
                logger.warning(f"Failed to load {yaml_file}: {e}")
            except ValueError as e:
                # Validation error
                logger.error(str(e))
                raise

    file_count = (1 if main_path.exists() else 0) + (
        len(yaml_files) if config_dir.exists() and config_dir.is_dir() else 0
    )
    logger.info(f"Configuration loaded and merged from {file_count} file(s)")
    return merged_config


class Settings(BaseSettings):
    """Application settings.

    Secrets are loaded from .env file.
    Non-sensitive config is loaded from config.yaml with fallback to defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env" if Path(".env").exists() else None,
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

    # PostgreSQL password (optional, only needed if using PostgreSQL)
    postgres_password: SecretStr | None = Field(
        default=None, description="PostgreSQL password (from .env, optional)"
    )

    # Telegram configuration (user client)
    telegram_api_id: int | None = Field(
        default=None, description="Telegram API ID (from .env)"
    )
    telegram_api_hash: SecretStr | None = Field(
        default=None, description="Telegram API hash (from .env)"
    )

    # === NON-SENSITIVE CONFIG (from config.yaml or defaults) ===

    def __init__(self, **data: Any):
        """Initialize settings with auto-loaded configs from all YAML files."""
        config = load_all_configs()

        # Apply config values as defaults (env vars override via setdefault)
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
            # Load database type (sqlite or postgres) - env var takes precedence
            data.setdefault("database_type", config["database"].get("type", "sqlite"))
            # Load SQLite path
            data.setdefault(
                "db_path", config["database"].get("path", "data/slack_events.db")
            )
            # Load PostgreSQL settings if present
            if "postgres" in config["database"]:
                pg = config["database"]["postgres"]
                data.setdefault("postgres_host", pg.get("host", "localhost"))
                data.setdefault("postgres_port", pg.get("port", 5432))
                data.setdefault("postgres_database", pg.get("database", "slack_events"))
                data.setdefault("postgres_user", pg.get("user", "postgres"))

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

        # Load monitored channels from config
        if "channels" in config:
            channels = []
            for ch in config["channels"]:
                channels.append(
                    ChannelConfig(
                        channel_id=ch["channel_id"],
                        channel_name=ch["channel_name"],
                        threshold_score=ch.get("threshold_score", 0.0),
                        keyword_weight=ch.get("keyword_weight", 10.0),
                        mention_weight=ch.get("mention_weight", 8.0),
                        reply_weight=ch.get("reply_weight", 5.0),
                        reaction_weight=ch.get("reaction_weight", 3.0),
                        anchor_weight=ch.get("anchor_weight", 4.0),
                        link_weight=ch.get("link_weight", 2.0),
                        file_weight=ch.get("file_weight", 3.0),
                        bot_penalty=ch.get("bot_penalty", -15.0),
                        whitelist_keywords=ch.get("whitelist_keywords", []),
                        trusted_bots=ch.get("trusted_bots", []),
                        prompt_file=ch.get("prompt_file", ""),
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

        # Multi-source configuration with auto-migration
        if "message_sources" in config:
            # New format: explicit message_sources
            message_sources = []
            for source_config in config["message_sources"]:
                message_sources.append(MessageSourceConfig(**source_config))
            data.setdefault("message_sources", message_sources)
        elif "channels" in config and len(config["channels"]) > 0:
            # Legacy format: auto-migrate from slack_channels
            logger.info(
                "Auto-migrating legacy 'channels' config to 'message_sources' format"
            )
            channel_ids = [ch["channel_id"] for ch in config["channels"]]
            slack_source = MessageSourceConfig(
                source_id=MessageSource.SLACK,
                enabled=True,
                bot_token_env="SLACK_BOT_TOKEN",
                raw_table="raw_slack_messages",
                state_table="ingestion_state_slack",
                prompt_file="config/prompts/slack.yaml",
                llm_settings={
                    "temperature": config.get("llm", {}).get("temperature", 1.0),
                    "timeout_seconds": config.get("llm", {}).get(
                        "timeout_seconds", 120
                    ),
                },
                channels=channel_ids,
            )
            data.setdefault("message_sources", [slack_source])
        else:
            # No sources configured - empty list
            data.setdefault("message_sources", [])

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

        # Load Telegram channels from config
        if "telegram_channels" in config:
            telegram_channels = []
            for ch in config["telegram_channels"]:
                # Handle both new format (username) and legacy format (channel_id)
                username = ch.get("username") or ch.get("channel_id", "")
                if not username:
                    logger.warning(
                        f"Telegram channel missing username/channel_id: {ch}"
                    )
                    continue

                telegram_channels.append(
                    TelegramChannelConfig(
                        username=username,
                        channel_name=ch["channel_name"],
                        threshold_score=ch.get("threshold_score", 0.0),
                        keyword_weight=ch.get("keyword_weight", 10.0),
                        mention_weight=ch.get("mention_weight", 8.0),
                        reply_weight=ch.get("reply_weight", 5.0),
                        reaction_weight=ch.get("reaction_weight", 3.0),
                        anchor_weight=ch.get("anchor_weight", 4.0),
                        link_weight=ch.get("link_weight", 2.0),
                        file_weight=ch.get("file_weight", 3.0),
                        bot_penalty=ch.get("bot_penalty", -15.0),
                        whitelist_keywords=ch.get("whitelist_keywords", []),
                        prompt_file=ch.get("prompt_file", ""),
                    )
                )
            data.setdefault("telegram_channels", telegram_channels)

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
    database_type: Literal["sqlite", "postgres"] = Field(
        default="sqlite", description="Database type: sqlite or postgres"
    )
    db_path: str = Field(
        default="data/slack_events.db", description="SQLite database path"
    )
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_database: str = Field(
        default="slack_events", description="PostgreSQL database name"
    )
    postgres_user: str = Field(default="postgres", description="PostgreSQL user")

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

    # Configuration paths
    object_registry_path: str = Field(
        default="config/object_registry.yaml",
        description="Path to object registry YAML",
    )
    channels_config_path: str = Field(
        default="config/channels.yaml",
        description="Path to channels configuration YAML",
    )

    # Multi-source configuration
    message_sources: list[MessageSourceConfig] = Field(
        default_factory=list,
        description="List of message sources (Slack, Telegram, etc.) to monitor",
    )

    # Telegram configuration
    telegram_channels: list[TelegramChannelConfig] = Field(
        default_factory=list,
        description="List of Telegram channels to monitor (loaded from config.yaml)",
    )
    telegram_session_path: str = Field(
        default="data/telegram_session",
        description="Path to Telethon session file (without .session extension)",
    )

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

    def get_source_config(self, source_id: MessageSource) -> MessageSourceConfig | None:
        """Get configuration for specific message source.

        Args:
            source_id: Message source identifier (SLACK, TELEGRAM, etc.)

        Returns:
            Message source config or None if not found
        """
        for source_config in self.message_sources:
            if source_config.source_id == source_id:
                return source_config
        return None

    def get_enabled_sources(self) -> list[MessageSourceConfig]:
        """Get list of enabled message sources.

        Returns:
            List of enabled message source configurations
        """
        return [
            source_config
            for source_config in self.message_sources
            if source_config.enabled
        ]


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
