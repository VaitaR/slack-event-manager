"""Tests for ingestion state functionality."""

from src.adapters.repository_factory import create_repository
from src.config.settings import Settings
from src.domain.protocols import RepositoryProtocol


def test_get_last_processed_ts_returns_none_for_new_channel(repo: RepositoryProtocol) -> None:
    """Test that get_last_processed_ts returns None for new channel."""
    result = repo.get_last_processed_ts("C123456")

    assert result is None


def test_update_and_get_last_processed_ts(repo: RepositoryProtocol) -> None:
    """Test updating and retrieving last processed timestamp."""
    channel_id = "C123456"
    timestamp = 1234567890.123456

    repo.update_last_processed_ts(channel_id, timestamp)
    result = repo.get_last_processed_ts(channel_id)

    assert result is not None
    assert abs(result - timestamp) < 0.000001


def test_update_last_processed_ts_upserts(repo: RepositoryProtocol) -> None:
    """Test that update_last_processed_ts performs upsert."""
    channel_id = "C123456"
    old_ts = 1234567890.0
    new_ts = 1234567900.0

    repo.update_last_processed_ts(channel_id, old_ts)
    first_result = repo.get_last_processed_ts(channel_id)

    repo.update_last_processed_ts(channel_id, new_ts)
    second_result = repo.get_last_processed_ts(channel_id)

    assert abs(first_result - old_ts) < 0.000001
    assert abs(second_result - new_ts) < 0.000001


def test_multiple_channels_independent_state(repo: RepositoryProtocol) -> None:
    """Test that different channels have independent state."""
    channel1 = "C111111"
    channel2 = "C222222"
    ts1 = 1234567890.0
    ts2 = 1234567900.0

    repo.update_last_processed_ts(channel1, ts1)
    repo.update_last_processed_ts(channel2, ts2)

    result1 = repo.get_last_processed_ts(channel1)
    result2 = repo.get_last_processed_ts(channel2)

    assert abs(result1 - ts1) < 0.000001
    assert abs(result2 - ts2) < 0.000001


def test_ingestion_state_persists_across_connections(
    repo: RepositoryProtocol, settings: Settings
) -> None:
    """Test that ingestion state persists across repository instances."""
    channel_id = "C123456"
    timestamp = 1234567890.123

    repo.update_last_processed_ts(channel_id, timestamp)

    repo2 = create_repository(settings)
    try:
        result = repo2.get_last_processed_ts(channel_id)
    finally:
        close = getattr(repo2, "close", None)
        if callable(close):
            close()

    assert result is not None
    assert abs(result - timestamp) < 0.000001
