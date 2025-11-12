"""Migration script for multi-source architecture.

This script migrates existing databases to support multi-source architecture:
1. Creates new tables (raw_telegram_messages, slack_ingestion_state, ingestion_state_telegram)
2. Adds source_id column to event_candidates and events tables
3. Migrates existing ingestion_state data to slack_ingestion_state
4. Sets default source_id='slack' for existing events and candidates

Safe to run multiple times (idempotent).
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def migrate_database(db_path: str, dry_run: bool = False) -> None:
    """Migrate database to multi-source schema.

    Args:
        db_path: Path to SQLite database
        dry_run: If True, only check what would be done without making changes
    """
    print(f"\n{'=' * 80}")
    print(f"Migrating database: {db_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'=' * 80}\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check current schema
        print("📋 Checking current schema...")
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}
        print(f"   Existing tables: {', '.join(sorted(existing_tables))}\n")

        # Step 1: Create raw_telegram_messages table
        print("1️⃣  Creating raw_telegram_messages table...")
        if "raw_telegram_messages" in existing_tables:
            print("   ✓ Table already exists, skipping\n")
        elif not dry_run:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_telegram_messages (
                        message_id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        message_thread_id TEXT,
                        ts TEXT NOT NULL,
                        ts_dt TIMESTAMP NOT NULL,
                        sender_id TEXT,
                        sender_name TEXT,
                        sender_username TEXT,
                        is_bot INTEGER NOT NULL DEFAULT 0,
                        text TEXT,
                        text_norm TEXT,
                        links_raw TEXT,
                        links_norm TEXT,
                        anchors TEXT,
                        reactions TEXT,
                        reply_count INTEGER DEFAULT 0,
                        forward_from TEXT,
                        edit_date TEXT,
                        ingested_at TIMESTAMP NOT NULL,
                        source_id TEXT NOT NULL DEFAULT 'telegram'
                )
            """)
            conn.commit()
            print("   ✓ Created raw_telegram_messages table\n")
        else:
            print("   → Would create raw_telegram_messages table\n")

        # Step 2: Create slack_ingestion_state table
        print("2️⃣  Creating slack_ingestion_state table...")
        if "slack_ingestion_state" in existing_tables:
            print("   ✓ Table already exists, skipping\n")
        elif not dry_run:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS slack_ingestion_state (
                    channel_id TEXT PRIMARY KEY,
                    max_processed_ts REAL NOT NULL DEFAULT 0,
                    resume_cursor TEXT,
                    resume_min_ts REAL,
                    updated_at TEXT
                )
            """)
            conn.commit()
            print("   ✓ Created slack_ingestion_state table\n")
        else:
            print("   → Would create slack_ingestion_state table\n")

        # Step 3: Create ingestion_state_telegram table
        print("3️⃣  Creating ingestion_state_telegram table...")
        if "ingestion_state_telegram" in existing_tables:
            print("   ✓ Table already exists, skipping\n")
        elif not dry_run:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_state_telegram (
                    chat_id TEXT PRIMARY KEY,
                    last_processed_message_id INTEGER NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            conn.commit()
            print("   ✓ Created ingestion_state_telegram table\n")
        else:
            print("   → Would create ingestion_state_telegram table\n")

        # Step 4: Migrate ingestion_state data to slack_ingestion_state
        print("4️⃣  Migrating ingestion_state data...")
        if "ingestion_state" in existing_tables:
            cursor.execute("SELECT COUNT(*) FROM ingestion_state")
            state_count = cursor.fetchone()[0]

            if state_count > 0:
                if "slack_ingestion_state" not in existing_tables or dry_run:
                    print(
                        f"   → Would migrate {state_count} records from ingestion_state to slack_ingestion_state\n"
                    )
                else:
                    # Check if already migrated
                    cursor.execute("SELECT COUNT(*) FROM slack_ingestion_state")
                    slack_state_count = cursor.fetchone()[0]

                    if slack_state_count > 0:
                        print(
                            f"   ✓ Already migrated ({slack_state_count} records in slack_ingestion_state)\n"
                        )
                    else:
                        cursor.execute("""
                            INSERT INTO slack_ingestion_state (
                                channel_id,
                                max_processed_ts,
                                resume_cursor,
                                resume_min_ts,
                                updated_at
                            )
                            SELECT channel_id, last_processed_ts, NULL, NULL, updated_at
                            FROM ingestion_state
                        """)
                        conn.commit()
                        print(
                            f"   ✓ Migrated {state_count} records to slack_ingestion_state\n"
                        )
            else:
                print("   ✓ No data to migrate\n")
        else:
            print("   ✓ No legacy ingestion_state table found\n")

        # Step 5: Add source_id column to event_candidates
        print("5️⃣  Adding source_id column to event_candidates...")
        cursor.execute("PRAGMA table_info(event_candidates)")
        columns = {row[1] for row in cursor.fetchall()}

        if "source_id" in columns:
            print("   ✓ Column already exists\n")
        elif not dry_run:
            cursor.execute("""
                    ALTER TABLE event_candidates ADD COLUMN source_id TEXT NOT NULL DEFAULT 'slack'
                """)
            conn.commit()
            print("   ✓ Added source_id column to event_candidates\n")
        else:
            print("   → Would add source_id column to event_candidates\n")

        # Step 6: Add source_id column to events
        print("6️⃣  Adding source_id column to events...")
        cursor.execute("PRAGMA table_info(events)")
        columns = {row[1] for row in cursor.fetchall()}

        if "source_id" in columns:
            print("   ✓ Column already exists\n")
        elif not dry_run:
            cursor.execute("""
                    ALTER TABLE events ADD COLUMN source_id TEXT NOT NULL DEFAULT 'slack'
                """)
            conn.commit()
            print("   ✓ Added source_id column to events\n")
        else:
            print("   → Would add source_id column to events\n")

        # Step 7: Verify migration
        print("7️⃣  Verifying migration...")
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        final_tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "raw_telegram_messages",
            "slack_ingestion_state",
            "ingestion_state_telegram",
        }

        missing_tables = expected_tables - final_tables
        if missing_tables:
            print(f"   ⚠️  Missing tables: {', '.join(missing_tables)}")
        else:
            print("   ✓ All required tables present")

        # Check columns
        cursor.execute("PRAGMA table_info(event_candidates)")
        candidate_columns = {row[1] for row in cursor.fetchall()}
        cursor.execute("PRAGMA table_info(events)")
        event_columns = {row[1] for row in cursor.fetchall()}

        if "source_id" in candidate_columns and "source_id" in event_columns:
            print("   ✓ All required columns present\n")
        else:
            if "source_id" not in candidate_columns:
                print("   ⚠️  Missing source_id column in event_candidates")
            if "source_id" not in event_columns:
                print("   ⚠️  Missing source_id column in events")
            print()

        # Print summary
        print(f"{'=' * 80}")
        if dry_run:
            print("✅ DRY RUN COMPLETE - No changes made")
        else:
            print("✅ MIGRATION COMPLETE")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}\n")
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> int:
    """Run migration script.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    parser = argparse.ArgumentParser(
        description="Migrate database to multi-source schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (check what would be done)
  python scripts/migrate_multi_source.py --dry-run

  # Migrate default database
  python scripts/migrate_multi_source.py

  # Migrate specific database
  python scripts/migrate_multi_source.py --db-path data/custom.db

  # Migrate all databases in data/ directory
  python scripts/migrate_multi_source.py --all
        """,
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/slack_events.db",
        help="Path to database file (default: data/slack_events.db)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate all .db files in data/ directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (check what would be done without making changes)",
    )

    args = parser.parse_args()

    try:
        if args.all:
            # Migrate all databases in data/ directory
            data_dir = Path("data")
            if not data_dir.exists():
                print(f"❌ Data directory not found: {data_dir}")
                return 1

            db_files = list(data_dir.glob("*.db"))
            if not db_files:
                print(f"❌ No .db files found in {data_dir}")
                return 1

            print(f"\n📦 Found {len(db_files)} database(s) to migrate:")
            for db_file in db_files:
                print(f"   - {db_file}")
            print()

            for db_file in db_files:
                migrate_database(str(db_file), dry_run=args.dry_run)

        else:
            # Migrate single database
            db_path = Path(args.db_path)
            if not db_path.exists():
                print(f"❌ Database not found: {db_path}")
                return 1

            migrate_database(str(db_path), dry_run=args.dry_run)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
