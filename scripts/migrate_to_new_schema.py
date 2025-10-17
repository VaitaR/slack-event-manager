"""Migration script for new event structure.

This script:
1. Backs up existing databases
2. Drops old events table
3. Creates new schema with comprehensive event structure

WARNING: This is a clean-slate migration. All existing event data will be lost.
Raw messages and candidates are preserved.
"""

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


def backup_database(db_path: Path) -> Path:
    """Create backup of database.

    Args:
        db_path: Path to database

    Returns:
        Path to backup file
    """
    if not db_path.exists():
        print(f"⚠️  Database not found: {db_path}")
        return db_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}.db"

    shutil.copy2(db_path, backup_path)
    print(f"✅ Backed up {db_path.name} -> {backup_path.name}")

    return backup_path


def drop_old_events_table(db_path: Path) -> None:
    """Drop old events table.

    Args:
        db_path: Path to database
    """
    if not db_path.exists():
        print(f"⚠️  Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if events table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='events'
            """
        )
        if cursor.fetchone():
            # Drop old events table
            cursor.execute("DROP TABLE IF EXISTS events")
            print(f"✅ Dropped old events table from {db_path.name}")
        else:
            print(f"ℹ️  No events table found in {db_path.name}")

        conn.commit()
    except sqlite3.Error as e:
        print(f"❌ Error dropping events table: {e}")
    finally:
        conn.close()


def migrate_database(db_path: Path, create_backup: bool = True) -> None:
    """Migrate database to new schema.

    Args:
        db_path: Path to database file
        create_backup: Whether to create backup (default: True)
    """
    print(f"\n🔄 Migrating {db_path.name}...")

    # Create backup
    if create_backup:
        backup_database(db_path)

    # Drop old events table
    drop_old_events_table(db_path)

    # New schema will be created automatically on next run
    print(f"✅ Migration complete for {db_path.name}")
    print("   New schema will be created automatically on next run")


def main() -> None:
    """Migrate all databases to new schema."""
    print("=" * 70)
    print("EVENT STRUCTURE MIGRATION SCRIPT")
    print("=" * 70)
    print()
    print("⚠️  WARNING: This is a clean-slate migration!")
    print("   - All existing event data will be lost")
    print("   - Raw messages and candidates are preserved")
    print("   - Backups will be created before migration")
    print()

    # Find databases
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    db_files = list(data_dir.glob("*.db"))
    if not db_files:
        print(f"ℹ️  No databases found in {data_dir}")
        return

    print(f"Found {len(db_files)} database(s) to migrate:")
    for db_file in db_files:
        print(f"  - {db_file.name}")
    print()

    # Confirm migration
    response = input("Proceed with migration? (yes/no): ").strip().lower()
    if response not in ("yes", "y"):
        print("❌ Migration cancelled")
        return

    print()
    print("Starting migration...")
    print()

    # Migrate each database
    for db_file in db_files:
        try:
            migrate_database(db_file, create_backup=True)
        except Exception as e:
            print(f"❌ Error migrating {db_file.name}: {e}")

    print()
    print("=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Run your pipeline to extract events with new structure")
    print("2. Check that titles are rendered correctly")
    print("3. Verify importance scores are calculated")
    print()


if __name__ == "__main__":
    main()
