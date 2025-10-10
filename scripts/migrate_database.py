"""Database migration script for adding enhanced Slack message fields.

This script adds new columns to existing databases to support enhanced
Slack message data extraction.

Usage:
    python scripts/migrate_database.py [database_path]

Examples:
    python scripts/migrate_database.py data/slack_events.db
    python scripts/migrate_database.py data/streamlit_demo.db
    python scripts/migrate_database.py  # Migrates all databases in data/
"""

import sqlite3
import sys
from pathlib import Path


def migrate_database(db_path: str) -> bool:
    """Migrate database to add new columns.

    Args:
        db_path: Path to SQLite database

    Returns:
        True if migration successful, False otherwise
    """
    print(f"\n🔧 Migrating database: {db_path}")

    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='raw_slack_messages'
        """
        )

        if not cursor.fetchone():
            print("⚠️  Table 'raw_slack_messages' not found. Skipping migration.")
            conn.close()
            return False

        # List of new columns to add
        new_columns = [
            ("user_real_name", "TEXT"),
            ("user_display_name", "TEXT"),
            ("user_email", "TEXT"),
            ("user_profile_image", "TEXT"),
            ("attachments_count", "INTEGER DEFAULT 0"),
            ("files_count", "INTEGER DEFAULT 0"),
            ("total_reactions", "INTEGER DEFAULT 0"),
            ("permalink", "TEXT"),
            ("edited_ts", "TEXT"),
            ("edited_user", "TEXT"),
        ]

        columns_added = 0
        columns_skipped = 0

        for column_name, column_type in new_columns:
            try:
                # Check if column already exists
                cursor.execute(f"PRAGMA table_info(raw_slack_messages)")
                existing_columns = [row[1] for row in cursor.fetchall()]

                if column_name in existing_columns:
                    print(f"  ⏭️  Column '{column_name}' already exists, skipping")
                    columns_skipped += 1
                    continue

                # Add column
                cursor.execute(
                    f"""
                    ALTER TABLE raw_slack_messages 
                    ADD COLUMN {column_name} {column_type}
                """
                )
                print(f"  ✅ Added column: {column_name} ({column_type})")
                columns_added += 1

            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"  ⏭️  Column '{column_name}' already exists, skipping")
                    columns_skipped += 1
                else:
                    print(f"  ❌ Error adding column '{column_name}': {e}")
                    conn.rollback()
                    conn.close()
                    return False

        # Commit changes
        conn.commit()
        conn.close()

        print(f"\n✅ Migration completed successfully!")
        print(f"   • Added: {columns_added} columns")
        print(f"   • Skipped: {columns_skipped} columns (already exist)")

        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


def find_databases(data_dir: str = "data") -> list[str]:
    """Find all SQLite databases in data directory.

    Args:
        data_dir: Directory to search for databases

    Returns:
        List of database paths
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        return []

    return [str(db) for db in data_path.glob("*.db")]


def main():
    """Main migration function."""
    print("=" * 80)
    print("Database Migration Tool - Enhanced Slack Fields")
    print("=" * 80)

    # Get database path from command line or find all databases
    if len(sys.argv) > 1:
        databases = [sys.argv[1]]
    else:
        print(
            "\nNo database specified. Searching for databases in 'data/' directory..."
        )
        databases = find_databases()

        if not databases:
            print("\n❌ No databases found in 'data/' directory.")
            print("\nUsage:")
            print("  python scripts/migrate_database.py [database_path]")
            print("\nExamples:")
            print("  python scripts/migrate_database.py data/slack_events.db")
            print("  python scripts/migrate_database.py data/streamlit_demo.db")
            sys.exit(1)

        print(f"\nFound {len(databases)} database(s):")
        for db in databases:
            print(f"  • {db}")

        # Ask for confirmation
        response = input("\nMigrate all databases? (y/n): ")
        if response.lower() != "y":
            print("❌ Migration cancelled.")
            sys.exit(0)

    # Migrate each database
    print("\n" + "=" * 80)
    print("Starting Migration")
    print("=" * 80)

    success_count = 0
    failed_count = 0

    for db_path in databases:
        if migrate_database(db_path):
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Migration Summary")
    print("=" * 80)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📊 Total: {len(databases)}")

    if failed_count == 0:
        print("\n🎉 All migrations completed successfully!")
    else:
        print(f"\n⚠️  {failed_count} migration(s) failed. Please check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
