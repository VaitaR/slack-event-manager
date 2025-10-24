"""Tests for Telegram scoring parity against Slack."""

from datetime import UTC, datetime

from src.domain.models import (
    ChannelConfig,
    SlackMessage,
    TelegramChannelConfig,
    TelegramMessage,
)
from src.services.scoring_engine import score_message


def test_telegram_scoring_matches_slack_signals() -> None:
    """Telegram messages with rich signals should align with Slack scoring."""

    now = datetime.now(tz=UTC)
    telegram_config = TelegramChannelConfig(
        username="@testchannel",
        channel_name="Test Telegram",
        whitelist_keywords=["urgent"],
        keyword_weight=1.5,
        mention_weight=2.0,
        reply_weight=4.0,
        reaction_weight=1.25,
        anchor_weight=0.5,
        link_weight=0.75,
        file_weight=2.5,
        bot_penalty=-7.0,
        trusted_bots=["trusted-bot"],
    )
    telegram_message = TelegramMessage(
        message_id="telegram-1",
        channel="@testchannel",
        message_date=now,
        sender_id="user-1",
        sender_name="Alice",
        text="Urgent update shipped today",
        text_norm="urgent update shipped today",
        links_raw=["https://example.com/update"],
        links_norm=["https://example.com/update"],
        anchors=["UPDATE-123"],
        reply_count=2,
        reactions={"👍": 3},
        reactions_count=3,
        has_file=True,
        file_mime="application/pdf",
        is_bot=False,
        bot_id=None,
    )

    telegram_score, telegram_features = score_message(telegram_message, telegram_config)

    assert telegram_score > 0
    assert telegram_features.reaction_count == 3
    assert telegram_features.has_files
    assert not telegram_features.is_bot

    slack_config = ChannelConfig(
        channel_id="C123",
        channel_name="Test Slack",
        whitelist_keywords=["urgent"],
        keyword_weight=1.5,
        mention_weight=2.0,
        reply_weight=4.0,
        reaction_weight=1.25,
        anchor_weight=0.5,
        link_weight=0.75,
        file_weight=2.5,
        bot_penalty=-7.0,
        trusted_bots=["trusted-bot"],
    )
    slack_message = SlackMessage(
        message_id="slack-1",
        channel="C123",
        ts="123.456",
        ts_dt=now,
        user="user-1",
        text="Urgent update shipped today",
        text_norm="urgent update shipped today",
        links_norm=["https://example.com/update"],
        anchors=["UPDATE-123"],
        attachments_count=1,
        files_count=0,
        reactions={"thumbs_up": 3},
        total_reactions=3,
        reply_count=2,
        is_bot=False,
        bot_id=None,
    )

    slack_score, slack_features = score_message(slack_message, slack_config)

    assert slack_score > 0
    assert slack_features.reaction_count == 3
    assert slack_features.has_files
    assert not slack_features.is_bot

    assert abs(slack_score - telegram_score) <= 0.25
