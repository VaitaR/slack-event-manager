#!/usr/bin/env python3
"""Fix PostgreSQL schema issues for multi-source support."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2


def main():
    """Fix PostgreSQL schema issues."""
    # Get connection details from environment
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DATABASE", "slack_events")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        print("❌ POSTGRES_PASSWORD not set")
        sys.exit(1)

    print(f"🔌 Connecting to PostgreSQL at {host}:{port}/{database}")

    try:
        conn = psycopg2.connect(
            host=host, port=port, database=database, user=user, password=password
        )
        conn.autocommit = True
        cur = conn.cursor()

        print("\n📋 Current tables:")
        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )
        for row in cur.fetchall():
            print(f"  - {row[0]}")

        print("\n🔧 Applying fixes...")

        # Fix 1: Rename ingestion_state to slack_ingestion_state if it exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename = 'ingestion_state'
            )
        """)
        if cur.fetchone()[0]:
            print("  ✓ Renaming ingestion_state → slack_ingestion_state")
            cur.execute("ALTER TABLE ingestion_state RENAME TO slack_ingestion_state")
        else:
            print(
                "  ℹ️  ingestion_state table doesn't exist (already renamed or never created)"
            )

        # Fix 2: Ensure slack_ingestion_state has correct columns
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename = 'slack_ingestion_state'
            )
        """)
        if cur.fetchone()[0]:
            print("  ✓ Checking slack_ingestion_state columns...")

            # Check if columns exist
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'slack_ingestion_state'
            """)
            existing_columns = {row[0] for row in cur.fetchall()}
            print(f"    Existing columns: {existing_columns}")

            # Rename last_processed_ts to max_processed_ts if needed
            if (
                "last_processed_ts" in existing_columns
                and "max_processed_ts" not in existing_columns
            ):
                print("    ✓ Renaming last_processed_ts → max_processed_ts")
                cur.execute(
                    "ALTER TABLE slack_ingestion_state RENAME COLUMN last_processed_ts TO max_processed_ts"
                )

            # Add missing columns
            if "resume_cursor" not in existing_columns:
                print("    ✓ Adding resume_cursor column")
                cur.execute(
                    "ALTER TABLE slack_ingestion_state ADD COLUMN resume_cursor VARCHAR(50)"
                )

            if "resume_min_ts" not in existing_columns:
                print("    ✓ Adding resume_min_ts column")
                cur.execute(
                    "ALTER TABLE slack_ingestion_state ADD COLUMN resume_min_ts VARCHAR(50)"
                )

        # Fix 3: Ensure event_candidates has source_id column
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'event_candidates'
        """)
        existing_columns = {row[0] for row in cur.fetchall()}

        if "source_id" not in existing_columns:
            print("  ✓ Adding source_id column to event_candidates")
            cur.execute(
                "ALTER TABLE event_candidates ADD COLUMN source_id VARCHAR(20) DEFAULT 'slack' NOT NULL"
            )
        else:
            print("  ℹ️  event_candidates.source_id already exists")

        print("\n✅ Schema fixes applied successfully!")

        print("\n📋 Final tables:")
        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )
        for row in cur.fetchall():
            print(f"  - {row[0]}")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
