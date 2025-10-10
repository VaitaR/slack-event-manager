"""Tests for ingestion state functionality."""

import tempfile
from pathlib import Path

import pytest

from src.adapters.sqlite_repository import SQLiteRepository


def test_get_last_processed_ts_returns_none_for_new_channel():
    """Test that get_last_processed_ts returns None for new channel."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        repo = SQLiteRepository(db_path)

        # Act
        result = repo.get_last_processed_ts("C123456")

        # Assert
        assert result is None


def test_update_and_get_last_processed_ts():
    """Test updating and retrieving last processed timestamp."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        repo = SQLiteRepository(db_path)
        channel_id = "C123456"
        timestamp = 1234567890.123456

        # Act
        repo.update_last_processed_ts(channel_id, timestamp)
        result = repo.get_last_processed_ts(channel_id)

        # Assert
        assert result is not None
        assert abs(result - timestamp) < 0.000001  # Float comparison


def test_update_last_processed_ts_upserts():
    """Test that update_last_processed_ts performs upsert."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        repo = SQLiteRepository(db_path)
        channel_id = "C123456"
        old_ts = 1234567890.0
        new_ts = 1234567900.0

        # Act
        repo.update_last_processed_ts(channel_id, old_ts)
        first_result = repo.get_last_processed_ts(channel_id)

        repo.update_last_processed_ts(channel_id, new_ts)
        second_result = repo.get_last_processed_ts(channel_id)

        # Assert
        assert abs(first_result - old_ts) < 0.000001
        assert abs(second_result - new_ts) < 0.000001


def test_multiple_channels_independent_state():
    """Test that different channels have independent state."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        repo = SQLiteRepository(db_path)
        channel1 = "C111111"
        channel2 = "C222222"
        ts1 = 1234567890.0
        ts2 = 1234567900.0

        # Act
        repo.update_last_processed_ts(channel1, ts1)
        repo.update_last_processed_ts(channel2, ts2)

        result1 = repo.get_last_processed_ts(channel1)
        result2 = repo.get_last_processed_ts(channel2)

        # Assert
        assert abs(result1 - ts1) < 0.000001
        assert abs(result2 - ts2) < 0.000001


def test_ingestion_state_persists_across_connections():
    """Test that ingestion state persists across repository instances."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        channel_id = "C123456"
        timestamp = 1234567890.123

        # Act - First repository instance
        repo1 = SQLiteRepository(db_path)
        repo1.update_last_processed_ts(channel_id, timestamp)

        # Act - Second repository instance (fresh connection)
        repo2 = SQLiteRepository(db_path)
        result = repo2.get_last_processed_ts(channel_id)

        # Assert
        assert result is not None
        assert abs(result - timestamp) < 0.000001

