"""Tests for multi-source protocols.

Tests MessageClientProtocol and updated RepositoryProtocol.
"""

from datetime import datetime
from unittest.mock import Mock

import pytz

from src.domain.models import (
    EventCandidate,
    MessageSource,
    SlackMessage,
    TelegramMessage,
)
from src.domain.protocols import MessageClientProtocol, RepositoryProtocol


class TestMessageClientProtocol:
    """Test MessageClientProtocol interface compliance."""

    def test_protocol_has_fetch_messages_method(self) -> None:
        """Test protocol defines fetch_messages method."""
        # Create a mock that satisfies the protocol
        mock_client = Mock(spec=MessageClientProtocol)
        mock_client.fetch_messages.return_value = []

        # Call the method
        result = mock_client.fetch_messages(
            channel_id="C123", oldest_ts="123456", latest_ts="789012"
        )

        assert result == []
        mock_client.fetch_messages.assert_called_once()

    def test_slack_client_mock_implements_protocol(self) -> None:
        """Test that a Slack-like client can implement the protocol."""
        mock_client = Mock(spec=MessageClientProtocol)
        mock_client.fetch_messages.return_value = [
            {"ts": "123", "text": "test", "user": "U123"}
        ]

        messages = mock_client.fetch_messages(channel_id="C123")
        assert len(messages) == 1
        assert messages[0]["text"] == "test"

    def test_telegram_client_mock_implements_protocol(self) -> None:
        """Test that a Telegram-like client can implement the protocol."""
        mock_client = Mock(spec=MessageClientProtocol)
        mock_client.fetch_messages.return_value = []  # Stub returns empty

        messages = mock_client.fetch_messages(channel_id="@test_channel")
        assert messages == []


class TestRepositoryProtocolMultiSource:
    """Test RepositoryProtocol supports multi-source operations."""

    def test_save_messages_accepts_slack_messages(self) -> None:
        """Test repository can save Slack messages."""
        mock_repo = Mock(spec=RepositoryProtocol)
        mock_repo.save_messages.return_value = 1

        messages = [
            SlackMessage(
                message_id="test123",
                channel="C123",
                ts="123456",
                ts_dt=datetime.now(pytz.UTC),
                text="test",
                source_id=MessageSource.SLACK,
            )
        ]

        count = mock_repo.save_messages(messages)
        assert count == 1

    def test_save_messages_accepts_telegram_messages(self) -> None:
        """Test repository can save Telegram messages."""
        mock_repo = Mock(spec=RepositoryProtocol)
        mock_repo.save_messages.return_value = 1

        messages = [
            TelegramMessage(
                message_id="tg123",
                channel="@test",
                message_date=datetime.now(pytz.UTC),
                text="test telegram",
                source_id=MessageSource.TELEGRAM,
            )
        ]

        # Repository should accept list of any message type
        count = mock_repo.save_messages(messages)
        assert count == 1

    def test_save_candidates_preserves_source_id(self) -> None:
        """Test saving candidates preserves source_id."""
        mock_repo = Mock(spec=RepositoryProtocol)
        mock_repo.save_candidates.return_value = 1

        from src.domain.models import CandidateStatus, ScoringFeatures

        candidate = EventCandidate(
            message_id="test123",
            channel="C123",
            ts_dt=datetime.now(pytz.UTC),
            text_norm="normalized",
            links_norm=[],
            anchors=[],
            score=50.0,
            status=CandidateStatus.NEW,
            features=ScoringFeatures(),
            source_id=MessageSource.TELEGRAM,
        )

        count = mock_repo.save_candidates([candidate])
        assert count == 1

    def test_get_last_processed_ts_per_source(self) -> None:
        """Test repository can track state per source."""
        mock_repo = Mock(spec=RepositoryProtocol)

        # Different timestamps for different sources
        def get_ts_side_effect(
            channel: str, source_id: MessageSource | None = None
        ) -> float | None:
            if source_id == MessageSource.SLACK:
                return 123456.0
            elif source_id == MessageSource.TELEGRAM:
                return 789012.0
            return None

        mock_repo.get_last_processed_ts.side_effect = get_ts_side_effect

        # Get Slack state
        slack_ts = mock_repo.get_last_processed_ts("C123", MessageSource.SLACK)
        assert slack_ts == 123456.0

        # Get Telegram state
        telegram_ts = mock_repo.get_last_processed_ts("@test", MessageSource.TELEGRAM)
        assert telegram_ts == 789012.0

    def test_update_last_processed_ts_per_source(self) -> None:
        """Test repository can update state per source."""
        mock_repo = Mock(spec=RepositoryProtocol)
        mock_repo.update_last_processed_ts.return_value = None

        # Update Slack state
        mock_repo.update_last_processed_ts("C123", 123456.0, MessageSource.SLACK)

        # Update Telegram state
        mock_repo.update_last_processed_ts("@test", 789012.0, MessageSource.TELEGRAM)

        assert mock_repo.update_last_processed_ts.call_count == 2
