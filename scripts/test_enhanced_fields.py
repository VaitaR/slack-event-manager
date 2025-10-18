"""Test script for enhanced Slack message fields.

This script verifies that new fields are properly extracted and stored:
- User information (real_name, display_name, email, profile_image)
- Attachments and files count
- Total reactions count
- Message permalinks
- Edit information
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from src.adapters.repository_factory import create_repository
from src.config.settings import Settings, get_settings
from src.domain.protocols import RepositoryProtocol
from src.use_cases.ingest_messages import process_slack_message


def test_enhanced_fields_extraction() -> None:
    """Test extraction of enhanced Slack message fields."""
    print("=" * 80)
    print("Testing Enhanced Slack Message Field Extraction")
    print("=" * 80)

    # Mock raw Slack message with all fields
    raw_msg = {
        "ts": "1710345600.123456",
        "user": "U12345",
        "text": "Important update about new feature release!",
        "blocks": [],
        "attachments": [
            {"text": "attachment 1"},
            {"text": "attachment 2"},
        ],
        "files": [
            {"name": "document.pdf"},
        ],
        "reactions": [
            {"name": "thumbsup", "count": 5},
            {"name": "rocket", "count": 3},
            {"name": "eyes", "count": 2},
        ],
        "reply_count": 12,
        "edited": {
            "user": "U12345",
            "ts": "1710346000.123456",
        },
    }

    # Mock user info
    user_info = {
        "id": "U12345",
        "name": "john.doe",
        "real_name": "John Doe",
        "profile": {
            "display_name": "Johnny",
            "email": "john.doe@example.com",
            "image_512": "https://example.com/avatar_512.jpg",
            "image_192": "https://example.com/avatar_192.jpg",
        },
    }

    # Mock permalink
    permalink = "https://workspace.slack.com/archives/C123/p1710345600123456"

    # Process message
    processed_msg = process_slack_message(
        raw_msg=raw_msg,
        channel_id="C123456",
        user_info=user_info,
        permalink=permalink,
    )

    # Verify basic fields
    print("\n✅ Basic Fields:")
    print(f"  Message ID: {processed_msg.message_id}")
    print(f"  Channel: {processed_msg.channel}")
    print(f"  User ID: {processed_msg.user}")
    print(f"  Timestamp: {processed_msg.ts}")

    # Verify user information
    print("\n✅ User Information:")
    print(f"  Real Name: {processed_msg.user_real_name}")
    print(f"  Display Name: {processed_msg.user_display_name}")
    print(f"  Email: {processed_msg.user_email}")
    print(f"  Profile Image: {processed_msg.user_profile_image}")

    assert processed_msg.user_real_name == "John Doe"
    assert processed_msg.user_display_name == "Johnny"
    assert processed_msg.user_email == "john.doe@example.com"
    assert processed_msg.user_profile_image == "https://example.com/avatar_512.jpg"

    # Verify attachments and files
    print("\n✅ Attachments & Files:")
    print(f"  Attachments Count: {processed_msg.attachments_count}")
    print(f"  Files Count: {processed_msg.files_count}")

    assert processed_msg.attachments_count == 2
    assert processed_msg.files_count == 1

    # Verify reactions
    print("\n✅ Reactions:")
    print(f"  Reactions: {processed_msg.reactions}")
    print(f"  Total Reactions: {processed_msg.total_reactions}")

    assert processed_msg.reactions == {"thumbsup": 5, "rocket": 3, "eyes": 2}
    assert processed_msg.total_reactions == 10

    # Verify reply count
    print("\n✅ Replies:")
    print(f"  Reply Count: {processed_msg.reply_count}")

    assert processed_msg.reply_count == 12

    # Verify permalink
    print("\n✅ Permalink:")
    print(f"  Permalink: {processed_msg.permalink}")

    assert processed_msg.permalink == permalink

    # Verify edit information
    print("\n✅ Edit Information:")
    print(f"  Edited Timestamp: {processed_msg.edited_ts}")
    print(f"  Edited User: {processed_msg.edited_user}")

    assert processed_msg.edited_ts == "1710346000.123456"
    assert processed_msg.edited_user == "U12345"

    print("\n" + "=" * 80)
    print("✅ All Enhanced Fields Extracted Successfully!")
    print("=" * 80)


def test_database_schema() -> None:
    """Test that database schema supports new fields."""
    print("\n" + "=" * 80)
    print("Testing Database Schema")
    print("=" * 80)

    # Create test database
    test_db_path = "data/test_enhanced_fields.db"
    settings: Settings = get_settings()
    temp_settings: Settings = settings.model_copy(
        update={"db_path": test_db_path, "database_type": "sqlite"}
    )
    repo: RepositoryProtocol = create_repository(temp_settings)

    # Create a test message with enhanced fields
    from src.domain.models import SlackMessage

    test_msg = SlackMessage(
        message_id="test123",
        channel="C123",
        ts="1710345600.123456",
        ts_dt=datetime.utcnow(),
        user="U12345",
        user_real_name="Test User",
        user_display_name="Tester",
        user_email="test@example.com",
        user_profile_image="https://example.com/avatar.jpg",
        is_bot=False,
        text="Test message",
        attachments_count=2,
        files_count=1,
        reactions={"thumbsup": 5},
        total_reactions=5,
        reply_count=3,
        permalink="https://slack.com/archives/C123/p123",
        edited_ts="1710346000.123456",
        edited_user="U12345",
    )

    # Save to database
    saved_count = repo.save_messages([test_msg])
    print(f"\n✅ Saved {saved_count} message(s) to database")

    # Retrieve from database
    messages = repo.get_new_messages_for_candidates()
    print(f"✅ Retrieved {len(messages)} message(s) from database")

    if messages:
        msg = messages[0]
        print("\n✅ Verified Enhanced Fields in Database:")
        print(f"  User Real Name: {msg.user_real_name}")
        print(f"  User Display Name: {msg.user_display_name}")
        print(f"  User Email: {msg.user_email}")
        print(f"  User Profile Image: {msg.user_profile_image}")
        print(f"  Attachments Count: {msg.attachments_count}")
        print(f"  Files Count: {msg.files_count}")
        print(f"  Total Reactions: {msg.total_reactions}")
        print(f"  Permalink: {msg.permalink}")
        print(f"  Edited Timestamp: {msg.edited_ts}")
        print(f"  Edited User: {msg.edited_user}")

        # Verify values
        assert msg.user_real_name == "Test User"
        assert msg.user_display_name == "Tester"
        assert msg.user_email == "test@example.com"
        assert msg.attachments_count == 2
        assert msg.files_count == 1
        assert msg.total_reactions == 5
        assert msg.permalink == "https://slack.com/archives/C123/p123"
        assert msg.edited_ts == "1710346000.123456"

    print("\n" + "=" * 80)
    print("✅ Database Schema Test Passed!")
    print("=" * 80)

    # Cleanup
    import os

    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        print(f"\n🧹 Cleaned up test database: {test_db_path}")


if __name__ == "__main__":
    try:
        test_enhanced_fields_extraction()
        test_database_schema()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! Enhanced fields are working correctly.")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
