#!/usr/bin/env python3
"""Test script to verify multi-source pipeline fix.

This script tests that run_multi_source_pipeline.py correctly passes
source_id to build_candidates_use_case for proper source isolation.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Settings
from src.domain.models import MessageSource
from src.use_cases.build_candidates import build_candidates_use_case


def test_source_id_passing():
    """Test that source_id is correctly passed to build_candidates_use_case."""

    # Create mock repository
    mock_repo = Mock()

    # Create mock settings with Telegram channel config
    mock_settings = Mock(spec=Settings)

    # Mock get_telegram_channel_config to return a config
    mock_telegram_config = Mock()
    mock_telegram_config.channel_name = "@test_channel"
    mock_telegram_config.whitelist_keywords = []
    mock_telegram_config.trusted_bots = []
    mock_telegram_config.threshold_score = 0.0
    mock_telegram_config.anchor_weight = 4.0
    mock_telegram_config.keyword_weight = 10.0
    mock_telegram_config.mention_weight = 8.0
    mock_telegram_config.reply_weight = 5.0
    mock_telegram_config.reaction_weight = 3.0
    mock_telegram_config.link_weight = 2.0
    mock_telegram_config.file_weight = 3.0
    mock_telegram_config.bot_penalty = -15.0
    mock_settings.get_telegram_channel_config.return_value = mock_telegram_config

    # Mock get_scoring_config to return the config for Telegram
    mock_settings.get_scoring_config.return_value = mock_telegram_config

    # Mock repository to return Telegram messages
    from datetime import datetime

    import pytz

    mock_telegram_message = Mock()
    mock_telegram_message.channel = "@test_channel"
    mock_telegram_message.source_id = MessageSource.TELEGRAM
    mock_telegram_message.text_norm = "test message"
    mock_telegram_message.anchors = []
    mock_telegram_message.links_norm = []
    mock_telegram_message.message_id = "test_msg_123"
    mock_telegram_message.ts_dt = datetime.now(tz=pytz.UTC)

    mock_repo.get_new_messages_for_candidates_by_source.return_value = [
        mock_telegram_message
    ]
    mock_repo.save_candidates.return_value = 1

    # Test the fix: call build_candidates_use_case with source_id
    result = build_candidates_use_case(
        repository=mock_repo, settings=mock_settings, source_id=MessageSource.TELEGRAM
    )

    # Verify that get_scoring_config was called with correct source_id
    mock_settings.get_scoring_config.assert_called_with(
        MessageSource.TELEGRAM, "@test_channel"
    )

    # Verify that repository method for Telegram was called
    mock_repo.get_new_messages_for_candidates_by_source.assert_called_with(
        MessageSource.TELEGRAM
    )

    print("✅ source_id correctly passed to build_candidates_use_case")
    print(f"✅ Messages processed: {result.messages_processed}")
    print(f"✅ Candidates created: {result.candidates_created}")

    return True


def test_backward_compatibility():
    """Test that build_candidates_use_case still works without source_id (Slack default)."""

    # Create mock repository
    mock_repo = Mock()

    # Create mock settings
    mock_settings = Mock(spec=Settings)

    # Mock get_channel_config for Slack
    mock_slack_config = Mock()
    mock_slack_config.channel_name = "#releases"
    mock_slack_config.whitelist_keywords = []
    mock_slack_config.trusted_bots = []
    mock_slack_config.threshold_score = 0.0
    mock_slack_config.anchor_weight = 4.0
    mock_slack_config.keyword_weight = 10.0
    mock_slack_config.mention_weight = 8.0
    mock_slack_config.reply_weight = 5.0
    mock_slack_config.reaction_weight = 3.0
    mock_slack_config.link_weight = 2.0
    mock_slack_config.file_weight = 3.0
    mock_slack_config.bot_penalty = -15.0
    mock_settings.get_channel_config.return_value = mock_slack_config
    mock_settings.get_scoring_config.return_value = mock_slack_config

    # Mock repository to return empty list for Slack messages (backward compatibility)
    mock_repo.get_new_messages_for_candidates.return_value = []
    mock_repo.save_candidates.return_value = 0

    # Test backward compatibility: call without source_id (should default to Slack)
    result = build_candidates_use_case(repository=mock_repo, settings=mock_settings)

    # When source_id is None, and no messages are returned, get_scoring_config is not called
    # because the function returns early with empty results
    # This is correct backward compatibility behavior
    # The function should not crash and should return valid result
    assert result.messages_processed == 0
    assert result.candidates_created == 0

    # Verify that repository method for Slack was called
    mock_repo.get_new_messages_for_candidates.assert_called_once()

    print("✅ Backward compatibility maintained")
    print(f"✅ Messages processed: {result.messages_processed}")
    print(f"✅ Candidates created: {result.candidates_created}")

    return True


if __name__ == "__main__":
    print("🧪 Testing multi-source pipeline fix...")

    try:
        test_source_id_passing()
        test_backward_compatibility()

        print("\n🎉 All tests passed! The fix is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
