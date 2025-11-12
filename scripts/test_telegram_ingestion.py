"""Test Telegram ingestion with real API.

Manual test script to verify Telegram integration works end-to-end.
Requires:
- TELEGRAM_API_ID and TELEGRAM_API_HASH in .env
- Session file created via scripts/telegram_auth.py
- At least one channel configured in config/telegram_channels.yaml

Usage:
    python scripts/test_telegram_ingestion.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytz


def _get_channel_config_value(channel: Any, field: str, default: Any) -> Any:
    """Safely access Telegram channel configuration values.

    Args:
        channel: Configuration object or mapping describing the channel.
        field: Attribute or key to retrieve.
        default: Fallback value if the field is missing.

    Returns:
        Value for the requested field if available, otherwise ``default``.
    """

    if hasattr(channel, field):
        return getattr(channel, field)
    if isinstance(channel, dict):
        return channel.get(field, default)
    return default


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.message_client_factory import get_message_client
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import get_settings
from src.domain.models import MessageSource
from src.use_cases.ingest_telegram_messages import ingest_telegram_messages_use_case


def main() -> int:
    """Run Telegram ingestion test.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    print("=" * 80)
    print("Telegram Ingestion Test")
    print("=" * 80)
    print()

    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return 1

    # Check Telegram credentials
    if not settings.telegram_api_id or not settings.telegram_api_hash:
        print("❌ Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        print("\nTo configure:")
        print("1. Add TELEGRAM_API_ID and TELEGRAM_API_HASH to .env")
        print("2. Run: python scripts/telegram_auth.py")
        print("3. Configure channels in config/telegram_channels.yaml")
        return 1

    # Check session file
    session_path = Path(f"{settings.telegram_session_path}.session")
    if not session_path.exists():
        print(f"❌ Error: Session file not found: {session_path}")
        print("\nTo create session:")
        print("  python scripts/telegram_auth.py")
        return 1

    print(f"✓ Session file found: {session_path}")
    print(f"✓ API ID: {settings.telegram_api_id}")
    print()

    # Check Telegram channels configuration
    if not settings.telegram_channels:
        print("❌ Error: No Telegram channels configured")
        print("\nTo configure:")
        print("  Edit config/telegram_channels.yaml")
        print("  Add at least one channel with channel_id and enabled: true")
        return 1

    print(f"✓ Found {len(settings.telegram_channels)} configured channel(s):")
    for channel in settings.telegram_channels:
        username = _get_channel_config_value(channel, "username", "unknown")
        channel_name = _get_channel_config_value(channel, "channel_name", "unknown")
        enabled = bool(_get_channel_config_value(channel, "enabled", False))
        status = "✓ enabled" if enabled else "⏭ disabled"
        print(f"  - {username} — {channel_name} ({status})")
    print()

    # Create test database
    test_db_path = "data/test_telegram_ingestion.db"
    print(f"Using test database: {test_db_path}")
    repository = SQLiteRepository(db_path=test_db_path)
    print()

    # Create Telegram client
    print("Creating Telegram client...")
    try:
        telegram_client = get_message_client(
            source_id=MessageSource.TELEGRAM,
            bot_token="",  # Not used for Telegram
        )
        print("✓ Telegram client created")
    except Exception as e:
        print(f"❌ Failed to create Telegram client: {e}")
        return 1
    print()

    # Set backfill date (1 day ago as per requirements)
    backfill_from_date = datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(days=1)
    print(f"Backfill from: {backfill_from_date.isoformat()}")
    print()

    # Run ingestion
    print("=" * 80)
    print("Running Telegram ingestion...")
    print("=" * 80)
    print()

    try:
        result = ingest_telegram_messages_use_case(
            telegram_client=telegram_client,
            repository=repository,
            settings=settings,
            backfill_from_date=backfill_from_date,
        )

        print()
        print("=" * 80)
        print("Ingestion Results")
        print("=" * 80)
        print(f"✓ Messages fetched: {result.messages_fetched}")
        print(f"✓ Messages saved: {result.messages_saved}")
        print(f"✓ Channels processed: {', '.join(result.channels_processed)}")

        if result.errors:
            print(f"\n⚠️  Errors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  - {error}")

        # Query database to verify
        print()
        print("=" * 80)
        print("Database Verification")
        print("=" * 80)

        telegram_messages = repository.get_telegram_messages(limit=10)
        print(f"✓ Total Telegram messages in DB: {len(telegram_messages)}")

        if telegram_messages:
            print("\nSample messages:")
            for i, msg in enumerate(telegram_messages[:5], 1):
                print(f"\n{i}. Message ID: {msg.message_id}")
                print(f"   Channel: {msg.channel}")
                print(f"   Date: {msg.message_date.isoformat()}")
                print(
                    f"   Text: {msg.text[:100]}..."
                    if len(msg.text) > 100
                    else f"   Text: {msg.text}"
                )
                print(f"   Links: {len(msg.links_norm)}")
                print(f"   Views: {msg.views}")

        print()
        print("=" * 80)
        print("✅ Test completed successfully!")
        print("=" * 80)
        print()
        print("Next steps:")
        print(
            "  1. Run full pipeline: python scripts/run_multi_source_pipeline.py --source telegram"
        )
        print("  2. Check database: sqlite3 data/test_telegram_ingestion.db")
        print("  3. View messages: SELECT * FROM raw_telegram_messages LIMIT 10;")

        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ Ingestion failed: {e}")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
