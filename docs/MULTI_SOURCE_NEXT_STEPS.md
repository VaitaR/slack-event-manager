# Multi-Source Architecture - Next Steps

**Last Updated:** 2025-10-17
**Current Status:** Phase 1 Complete (Foundation Layer)

## Summary of Completed Work

We've successfully implemented the **foundation layer** for multi-source architecture:

✅ **Domain Models** - MessageSource enum, TelegramMessage, source_id tracking
✅ **Protocols** - MessageClientProtocol, updated RepositoryProtocol
✅ **Tests** - 28 new tests, 185/185 total tests passing
✅ **Backward Compatibility** - 100% verified, no breaking changes
✅ **Documentation** - Implementation summary and progress tracking

## Immediate Next Steps

To continue the implementation, follow these steps in order:

### Step 1: Repository Layer Tests (Phase 2.1)

Create `tests/test_repository_multi_source.py`:

```python
def test_raw_telegram_messages_table_created():
    """Test Telegram raw table is created."""
    repo = SQLiteRepository(":memory:")
    # Verify table exists

def test_table_routing_by_source_id():
    """Test _get_raw_table_name() routes correctly."""
    repo = SQLiteRepository(":memory:")
    assert repo._get_raw_table_name(MessageSource.SLACK) == "raw_slack_messages"
    assert repo._get_raw_table_name(MessageSource.TELEGRAM) == "raw_telegram_messages"

def test_source_specific_state_tables():
    """Test source-specific state tables."""
    repo = SQLiteRepository(":memory:")
    # Verify ingestion_state_slack and ingestion_state_telegram exist
```

### Step 2: Repository Implementation (Phase 2.2)

Update `src/adapters/sqlite_repository.py`:

1. **Add Telegram raw table schema:**
```python
cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_telegram_messages (
        message_id TEXT PRIMARY KEY,
        channel TEXT NOT NULL,
        message_date TEXT NOT NULL,
        sender_id TEXT,
        sender_name TEXT,
        text TEXT,
        text_norm TEXT,
        forward_from_channel TEXT,
        forward_from_message_id TEXT,
        media_type TEXT,
        links_raw TEXT,
        links_norm TEXT,
        anchors TEXT,
        views INTEGER DEFAULT 0,
        ingested_at TEXT
    )
""")
```

2. **Add source-specific state tables:**
```python
cursor.execute("""
    CREATE TABLE IF NOT EXISTS ingestion_state_slack (
        channel_id TEXT PRIMARY KEY,
        last_processed_ts REAL NOT NULL,
        updated_at TEXT NOT NULL
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS ingestion_state_telegram (
        channel_id TEXT PRIMARY KEY,
        last_processed_message_id INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    )
""")
```

3. **Add routing methods:**
```python
def _get_raw_table_name(self, source_id: MessageSource) -> str:
    """Get raw table name for source."""
    if source_id == MessageSource.SLACK:
        return "raw_slack_messages"
    elif source_id == MessageSource.TELEGRAM:
        return "raw_telegram_messages"
    else:
        raise ValueError(f"Unknown source: {source_id}")

def _get_state_table_name(self, source_id: MessageSource) -> str:
    """Get state table name for source."""
    if source_id == MessageSource.SLACK:
        return "ingestion_state_slack"
    elif source_id == MessageSource.TELEGRAM:
        return "ingestion_state_telegram"
    else:
        raise ValueError(f"Unknown source: {source_id}")
```

4. **Update get_last_processed_ts() and update_last_processed_ts():**
```python
def get_last_processed_ts(
    self, channel: str, source_id: MessageSource | None = None
) -> float | None:
    """Get last processed timestamp (source-specific)."""
    if source_id is None:
        source_id = MessageSource.SLACK  # Default for backward compat

    table_name = self._get_state_table_name(source_id)
    # Query from appropriate table
```

### Step 3: Telegram Client Stub (Phase 3.1-3.2)

Create `src/adapters/telegram_client.py`:

```python
"""Telegram client stub adapter.

TODO: Implement with telethon or python-telegram-bot library.
"""

from typing import Any

from src.domain.protocols import MessageClientProtocol


class TelegramClient:
    """Telegram client stub (returns empty results for now)."""

    def __init__(self, bot_token: str) -> None:
        """Initialize Telegram client.

        Args:
            bot_token: Telegram bot token

        TODO: Initialize actual Telegram client
        """
        self.bot_token = bot_token

    def fetch_messages(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Telegram channel (stub).

        Args:
            channel_id: Telegram channel username or ID
            oldest_ts: Oldest message ID to fetch
            latest_ts: Latest message ID to fetch
            limit: Maximum messages to fetch

        Returns:
            Empty list (stub implementation)

        TODO: Implement actual Telegram API calls
        """
        # Stub: return empty list
        return []

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        """Get user information (stub).

        Args:
            user_id: Telegram user ID

        Returns:
            Default user info dictionary

        TODO: Implement actual user lookup
        """
        return {"id": user_id, "username": "unknown", "first_name": "Unknown"}
```

### Step 4: Message Client Factory (Phase 3.3-3.4)

Create `src/adapters/message_client_factory.py`:

```python
"""Factory for creating message source clients."""

from src.adapters.slack_client import SlackClient
from src.adapters.telegram_client import TelegramClient
from src.config.settings import Settings
from src.domain.models import MessageSource
from src.domain.protocols import MessageClientProtocol


def create_message_client(
    source_id: MessageSource, settings: Settings
) -> MessageClientProtocol:
    """Create message client for specified source.

    Args:
        source_id: Message source identifier
        settings: Application settings

    Returns:
        Message client instance

    Raises:
        ValueError: If source_id is unknown

    Example:
        >>> settings = get_settings()
        >>> client = create_message_client(MessageSource.SLACK, settings)
        >>> messages = client.fetch_messages("C123")
    """
    if source_id == MessageSource.SLACK:
        return SlackClient(
            bot_token=settings.slack_bot_token.get_secret_value()
        )
    elif source_id == MessageSource.TELEGRAM:
        # Get Telegram token from settings (to be added)
        telegram_token = getattr(
            settings, "telegram_bot_token", None
        )
        if telegram_token:
            return TelegramClient(bot_token=telegram_token.get_secret_value())
        else:
            # Stub with dummy token
            return TelegramClient(bot_token="dummy_token")
    else:
        raise ValueError(f"Unknown message source: {source_id}")
```

### Step 5: Configuration Support (Phase 1.6)

Update `src/config/settings.py` to add:

```python
class MessageSourceConfig(BaseModel):
    """Configuration for a message source."""

    source_id: MessageSource
    enabled: bool
    raw_table: str
    state_table: str
    prompt_file: str
    llm_settings: dict[str, Any]
    channels: list[str]


class Settings(BaseSettings):
    # ... existing fields ...

    # Multi-source configuration
    message_sources: list[MessageSourceConfig] = Field(
        default_factory=list,
        description="Message source configurations"
    )

    def __init__(self, **data: Any):
        """Initialize with auto-migration from legacy config."""
        config = load_all_configs()

        # Auto-migrate from legacy slack_channels if no message_sources
        if "message_sources" not in config and "channels" in config:
            # Create default Slack source from existing channels
            slack_source = {
                "source_id": "slack",
                "enabled": True,
                "raw_table": "raw_slack_messages",
                "state_table": "ingestion_state_slack",
                "prompt_file": "config/prompts/slack.txt",
                "llm_settings": {
                    "temperature": config.get("llm", {}).get("temperature", 1.0),
                    "timeout_seconds": config.get("llm", {}).get("timeout_seconds", 30),
                },
                "channels": [ch["channel_id"] for ch in config.get("channels", [])],
            }
            data.setdefault("message_sources", [slack_source])
        elif "message_sources" in config:
            data.setdefault("message_sources", config["message_sources"])

        super().__init__(**data)
```

### Step 6: Create Example Configuration Files

Create `config/defaults/main.example.yaml` with message_sources section:

```yaml
# Multi-Source Configuration
message_sources:
  # Slack source (default)
  - source_id: slack
    enabled: true
    raw_table: raw_slack_messages
    state_table: ingestion_state_slack
    prompt_file: config/prompts/slack.txt
    llm_settings:
      temperature: 1.0
      timeout_seconds: 30
    channels:
      - C1234567890
      - C0987654321

  # Telegram source (stub, disabled by default)
  - source_id: telegram
    enabled: false
    raw_table: raw_telegram_messages
    state_table: ingestion_state_telegram
    prompt_file: config/prompts/telegram.txt
    llm_settings:
      temperature: 0.7
      timeout_seconds: 30
    channels:
      - "@crypto_news"
```

### Step 7: Create Prompt Files

Create `config/prompts/slack.txt`:
```
(Copy current LLM prompt from llm_client.py into this file)
```

Create `config/prompts/telegram.txt`:
```
You are an expert at extracting structured event information from Telegram channel messages.

(Adapt the prompt for cryptocurrency/external news context)
```

### Step 8: Migration Script

Create `scripts/migrate_multi_source.py`:

```python
"""Database migration script for multi-source support.

Creates new tables and migrates data from legacy ingestion_state.
"""

import sqlite3
from pathlib import Path
from datetime import datetime


def migrate_database(db_path: str) -> None:
    """Migrate database to multi-source schema.

    Args:
        db_path: Path to SQLite database

    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Create new tables
    # ... (code from Step 2)

    # 2. Migrate data from ingestion_state to ingestion_state_slack
    cursor.execute("""
        INSERT OR IGNORE INTO ingestion_state_slack (
            channel_id, last_processed_ts, updated_at
        )
        SELECT channel_id, last_processed_ts, ? as updated_at
        FROM ingestion_state
    """, (datetime.utcnow().isoformat(),))

    conn.commit()
    conn.close()

    print(f"✅ Migration complete: {db_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for db_file in sys.argv[1:]:
            migrate_database(db_file)
    else:
        # Migrate all databases in data/
        data_dir = Path("data")
        for db_file in data_dir.glob("*.db"):
            migrate_database(str(db_file))
```

## Testing Strategy

For each implementation step:

1. **Write tests first** (TDD)
2. **Run tests** (they should fail)
3. **Implement minimal code** to pass tests
4. **Run all tests** (new + existing)
5. **Verify backward compatibility**
6. **Refactor if needed**
7. **Update documentation**

## Success Metrics

Track these metrics after each phase:

- [ ] All new tests pass
- [ ] All existing tests pass (185+)
- [ ] No linter errors
- [ ] Test coverage ≥ 90% for new code
- [ ] Documentation updated
- [ ] Backward compatibility maintained

## Timeline Estimate

Based on TDD approach and current progress:

- **Phase 2** (Repository): 2-3 hours
- **Phase 3** (Adapters): 2-3 hours
- **Phase 4** (Use Cases): 3-4 hours
- **Phase 5** (CLI): 1-2 hours
- **Phase 6** (Documentation): 1-2 hours

**Total remaining: 9-14 hours**

## Questions to Answer

Before continuing, clarify:

1. **Telegram Bot Token:** Where should it be stored? (.env?)
2. **Prompt Templates:** Should they be in files or embedded in code?
3. **Cross-Source Deduplication:** Strict isolation or configurable?
4. **Migration Strategy:** Automatic on startup or manual script?

## References

- **Implementation Plan:** `/multi-source-architecture.plan.md`
- **Progress Tracking:** `docs/MULTI_SOURCE_PROGRESS.md`
- **Implementation Summary:** `docs/MULTI_SOURCE_IMPLEMENTATION_SUMMARY.md`
- **Current Code:** `src/domain/models.py`, `src/domain/protocols.py`
- **Tests:** `tests/test_domain_models_multi_source.py`, `tests/test_protocols_multi_source.py`

---

**Ready to continue? Start with Step 1: Repository Layer Tests**
