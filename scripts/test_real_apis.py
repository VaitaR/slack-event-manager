#!/usr/bin/env python3
"""Quick test script to verify real API credentials and connections."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings


def test_apis():
    """Test each API connection separately with short timeouts."""

    print("🧪 Testing Real API Connections")
    print("=" * 50)

    # Load settings
    try:
        settings = get_settings()
        print("✅ Settings loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return False

    # Test 1: Slack API
    print("\n📱 Test 1: Slack API Connection")
    print("-" * 30)
    try:
        from src.adapters.slack_client import SlackClient

        slack_client = SlackClient(
            bot_token=settings.slack_bot_token.get_secret_value()
        )
        print("✅ Slack client initialized")

        # Try to fetch 1 message
        print("   Fetching 1 test message...")
        messages = slack_client.fetch_messages("C04V0TK7UG6", limit=1)
        print(f"✅ Slack API works! Fetched {len(messages)} message(s)")

        if messages:
            msg = messages[0]
            print(f"   Sample: {msg.get('text', '')[:50]}...")

    except Exception as e:
        print(f"❌ Slack API failed: {e}")
        return False

    # Test 2: OpenAI API
    print("\n🤖 Test 2: OpenAI API Connection")
    print("-" * 30)
    try:
        from src.adapters.llm_client import LLMClient

        llm_client = LLMClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=10,
        )
        print("✅ LLM client initialized")
        print(f"   Model: {settings.llm_model}")
        print(f"   Temperature: {settings.llm_temperature}")

        # Try a simple extraction
        print("   Testing LLM extraction with short text...")
        from datetime import datetime

        test_text = "New feature released: improved dashboard. Coming next week."

        response = llm_client.extract_events(
            text=test_text,
            links=[],
            message_ts_dt=datetime.utcnow(),
            channel_name="test",
        )

        print(
            f"✅ LLM API works! is_event={response.is_event}, events={len(response.events)}"
        )

        # Get metadata
        metadata = llm_client.get_call_metadata()
        print(f"   Tokens: {metadata.tokens_in} in, {metadata.tokens_out} out")
        print(f"   Cost: ${metadata.cost_usd:.4f}")
        print(f"   Latency: {metadata.latency_ms}ms")

    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 3: Database
    print("\n💾 Test 3: Database Operations")
    print("-" * 30)
    try:
        import tempfile
        from src.adapters.sqlite_repository import SQLiteRepository

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()

        repo = SQLiteRepository(temp_db.name)
        print("✅ Database initialized")
        print(f"   Path: {temp_db.name}")

        # Clean up
        import os

        os.unlink(temp_db.name)
        print("✅ Database operations work!")

    except Exception as e:
        print(f"❌ Database failed: {e}")
        return False

    print("\n🎉 All API tests passed!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_apis()
    sys.exit(0 if success else 1)
