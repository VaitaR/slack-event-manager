"""Tests for multi-source candidate building."""

from datetime import datetime
from pathlib import Path

import pytest
from datetime import datetime

import pytest
import pytz

from src.config.settings import Settings
from src.domain.models import MessageSource, TelegramMessage
from src.domain.protocols import RepositoryProtocol
from src.use_cases.build_candidates import build_candidates_use_case


def test_build_candidates_telegram_source(
    repo: RepositoryProtocol, settings: Settings
) -> None:
    """Test building candidates from Telegram messages."""
    repository = repo
    # Create and save Telegram messages
    now = datetime.now(tz=pytz.UTC)
    messages = [
        TelegramMessage(
            message_id="tg_001",
            channel="@test_channel",
            message_date=now,
            sender_id="user1",
            sender_name="Test User",
            text="🚀 Launching new feature tomorrow",
            text_norm="launching new feature tomorrow",
            links_raw=["https://example.com"],
            links_norm=["https://example.com"],
            anchors=["FEAT-123"],
            views=10,
            ingested_at=now,
            source_id=MessageSource.TELEGRAM,
        ),
        TelegramMessage(
            message_id="tg_002",
            channel="@test_channel",
            message_date=now,
            sender_id="user2",
            sender_name="Admin",
            text="Scheduled maintenance on database cluster",
            text_norm="scheduled maintenance on database cluster",
            links_raw=[],
            links_norm=[],
            anchors=["MAINT-456"],
            views=5,
            ingested_at=now,
            source_id=MessageSource.TELEGRAM,
        ),
    ]

    # Save messages
    saved = repository.save_telegram_messages(messages)
    assert saved == 2

    # Build candidates for Telegram source
    result = build_candidates_use_case(
        repository=repository, settings=settings, source_id=MessageSource.TELEGRAM
    )

    # Verify results
    print("\n📊 Candidate Building Results:")
    print(f"   Messages processed: {result.messages_processed}")
    print(f"   Candidates created: {result.candidates_created}")
    print(f"   Average score: {result.average_score:.2f}")

    # At least messages should be processed
    assert result.messages_processed == 2

    # Verify candidates have correct source_id
    if result.candidates_created > 0:
        candidates = repository.get_candidates_by_source(MessageSource.TELEGRAM)
        assert len(candidates) > 0
        assert all(c.source_id == MessageSource.TELEGRAM for c in candidates)
        print(f"   ✓ All {len(candidates)} candidates have source_id=telegram")


def test_build_candidates_source_isolation(
    repo: RepositoryProtocol, settings: Settings
) -> None:
    """Test that Telegram and Slack candidates are isolated."""
    repository = repo
    now = datetime.now(tz=pytz.UTC)

    # Save Telegram message
    tg_msg = TelegramMessage(
        message_id="tg_001",
        channel="@test_channel",
        message_date=now,
        sender_id="user1",
        sender_name="Test",
        text="Test telegram message with important event",
        text_norm="test telegram message with important event",
        links_raw=[],
        links_norm=[],
        anchors=["TG-123"],
        views=10,
        ingested_at=now,
        source_id=MessageSource.TELEGRAM,
    )
    repository.save_telegram_messages([tg_msg])

    # Build candidates for Telegram
    result = build_candidates_use_case(
        repository=repository, settings=settings, source_id=MessageSource.TELEGRAM
    )

    print("\n🔒 Source Isolation Test:")
    print(f"   Telegram messages processed: {result.messages_processed}")
    print(f"   Telegram candidates created: {result.candidates_created}")

    # Verify isolation
    telegram_candidates = repository.get_candidates_by_source(MessageSource.TELEGRAM)
    slack_candidates = repository.get_candidates_by_source(MessageSource.SLACK)

    print(f"   Telegram candidates: {len(telegram_candidates)}")
    print(f"   Slack candidates: {len(slack_candidates)}")
    print("   ✓ Source isolation verified")

    assert len(slack_candidates) == 0, "Slack candidates should be empty"


def test_build_candidates_backward_compatibility(
    repo: RepositoryProtocol, settings: Settings
) -> None:
    """Test that build_candidates_use_case still works without source_id (Slack default)."""
    repository = repo
    # Call without source_id (should default to Slack)
    result = build_candidates_use_case(repository=repository, settings=settings)

    # Should not crash and return valid result
    assert result.messages_processed >= 0
    assert result.candidates_created >= 0
    assert result.average_score >= 0.0

    print("\n✅ Backward compatibility test passed:")
    print(f"   Messages processed: {result.messages_processed}")
    print(f"   Candidates created: {result.candidates_created}")


def test_get_new_messages_by_source_telegram(repo: RepositoryProtocol) -> None:
    """Test get_new_messages_for_candidates_by_source for Telegram."""
    repository = repo
    now = datetime.now(tz=pytz.UTC)

    # Save Telegram messages
    messages = [
        TelegramMessage(
            message_id=f"tg_{i:03d}",
            channel="@test",
            message_date=now,
            sender_id=f"user{i}",
            sender_name=f"User {i}",
            text=f"Message {i}",
            text_norm=f"message {i}",
            links_raw=[],
            links_norm=[],
            anchors=[],
            views=i,
            ingested_at=now,
            source_id=MessageSource.TELEGRAM,
        )
        for i in range(5)
    ]
    repository.save_telegram_messages(messages)

    # Get new messages for Telegram source
    new_messages = repository.get_new_messages_for_candidates_by_source(
        MessageSource.TELEGRAM
    )

    assert len(new_messages) == 5
    assert all(msg.source_id == MessageSource.TELEGRAM for msg in new_messages)
    print(f"\n✅ Retrieved {len(new_messages)} Telegram messages for candidates")


def test_get_new_messages_by_source_slack(repo: RepositoryProtocol) -> None:
    """Test get_new_messages_for_candidates_by_source for Slack."""
    repository = repo
    # Get new messages for Slack source (should be empty)
    new_messages = repository.get_new_messages_for_candidates_by_source(
        MessageSource.SLACK
    )

    assert len(new_messages) == 0
    print(f"✅ Slack messages: {len(new_messages)} (expected 0)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
