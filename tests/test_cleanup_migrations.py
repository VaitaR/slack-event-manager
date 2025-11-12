import sqlite3
from pathlib import Path

from src.adapters.sqlite_repository import SQLiteRepository
from src.domain.models import MessageSource


def test_channel_watermarks_table_restored(tmp_path: Path) -> None:
    """Ensure legacy state tables remain available when required by repositories."""

    db_path = tmp_path / "state.db"
    repository = SQLiteRepository(str(db_path))

    # Simulate a migration that drops state tables.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS channel_watermarks")
        conn.execute("DROP VIEW IF EXISTS ingestion_state_slack")
        conn.execute("DROP TABLE IF EXISTS slack_ingestion_state")
        conn.commit()
    finally:
        conn.close()

    # Rollback by re-running schema creation.
    repository._create_schema()

    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='channel_watermarks'"
        ).fetchone()
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='slack_ingestion_state'"
        ).fetchone()
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='ingestion_state_slack'"
        ).fetchone()
    finally:
        conn.close()

    repository.update_watermark("C123", "1700000000.000")
    assert repository.get_watermark("C123") == "1700000000.000"

    repository.update_last_processed_ts(
        "C123", 1700000000.0, source_id=MessageSource.SLACK
    )
    assert (
        repository.get_last_processed_ts("C123", source_id=MessageSource.SLACK)
        == 1700000000.0
    )
