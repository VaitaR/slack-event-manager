#!/usr/bin/env python3
"""Diagnostic script to test each pipeline component separately.

Tests components one by one to identify issues before running full pipeline.
"""

import sys
import signal
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.adapters.slack_client import SlackClient
from src.adapters.llm_client import LLMClient


def log(msg: str) -> None:
    """Print with immediate flush."""
    print(msg)
    sys.stdout.flush()


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    log("\n❌ TIMEOUT: Operation took too long (>30s)")
    sys.exit(1)


def test_settings() -> bool:
    """Test if settings load correctly."""
    log("\n" + "=" * 70)
    log("TEST 1: Configuration Settings")
    log("=" * 70)

    try:
        log("⏳ Loading settings...")
        settings = get_settings()
        log(f"✅ Settings loaded successfully")
        log(f"   • Model: {settings.llm_model}")
        log(f"   • Temperature: {settings.llm_temperature}")
        log(f"   • DB Path: {settings.db_path}")
        log(f"   • Slack channels: {len(settings.slack_channels)} configured")
        return True
    except Exception as e:
        log(f"❌ Failed to load settings: {e}")
        return False


def test_slack_init(settings) -> tuple[bool, SlackClient | None]:
    """Test Slack client initialization."""
    log("\n" + "=" * 70)
    log("TEST 2: Slack Client Initialization")
    log("=" * 70)

    try:
        log("⏳ Initializing Slack client...")
        slack_client = SlackClient(
            bot_token=settings.slack_bot_token.get_secret_value()
        )
        log("✅ Slack client initialized")
        return True, slack_client
    except Exception as e:
        log(f"❌ Failed to initialize Slack client: {e}")
        return False, None


def test_slack_auth(slack_client: SlackClient) -> bool:
    """Test Slack authentication."""
    log("\n" + "=" * 70)
    log("TEST 3: Slack Authentication")
    log("=" * 70)

    try:
        log("⏳ Testing Slack auth (timeout: 10s)...")

        # Set alarm for 10 seconds
        signal.alarm(10)
        response = slack_client.client.auth_test()
        signal.alarm(0)  # Cancel alarm

        if response["ok"]:
            log(f"✅ Slack authentication successful")
            log(f"   • Bot user: {response.get('user')}")
            log(f"   • Team: {response.get('team')}")
            return True
        else:
            log(f"❌ Auth test failed: {response.get('error')}")
            return False
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        log(f"❌ Auth test error: {e}")
        return False


def test_slack_fetch_messages(slack_client: SlackClient, channel_id: str) -> bool:
    """Test fetching messages from Slack."""
    log("\n" + "=" * 70)
    log("TEST 4: Fetch Messages from Slack")
    log("=" * 70)

    try:
        log(f"⏳ Fetching 3 messages from channel {channel_id} (timeout: 15s)...")

        # Set alarm for 15 seconds
        signal.alarm(15)
        messages = slack_client.fetch_messages(channel_id=channel_id, limit=3)
        signal.alarm(0)  # Cancel alarm

        log(f"✅ Successfully fetched {len(messages)} messages")

        if messages:
            log("\n📨 Sample message:")
            msg = messages[0]
            text = (
                msg.get("text", "")[:100] + "..."
                if len(msg.get("text", "")) > 100
                else msg.get("text", "")
            )
            log(f"   • Timestamp: {msg.get('ts')}")
            log(f"   • User: {msg.get('user', 'unknown')}")
            log(f"   • Text: {text}")

        return True
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        log(f"❌ Failed to fetch messages: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_llm_init(settings) -> tuple[bool, LLMClient | None]:
    """Test LLM client initialization."""
    log("\n" + "=" * 70)
    log("TEST 5: LLM Client Initialization")
    log("=" * 70)

    try:
        log("⏳ Initializing LLM client...")
        llm_client = LLMClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=10,
        )
        log(f"✅ LLM client initialized")
        log(f"   • Model: {settings.llm_model}")
        log(f"   • Temperature: {settings.llm_temperature}")
        return True, llm_client
    except Exception as e:
        log(f"❌ Failed to initialize LLM client: {e}")
        return False, None


def test_llm_extraction(llm_client: LLMClient) -> bool:
    """Test LLM event extraction with a simple message."""
    log("\n" + "=" * 70)
    log("TEST 6: LLM Event Extraction")
    log("=" * 70)

    # Simple test message
    test_text = """We're releasing version 2.0 on October 15th.
This includes new features and bug fixes.
More details: https://example.com/release-notes"""

    test_links = ["https://example.com/release-notes"]
    test_timestamp = datetime.utcnow()

    try:
        log("⏳ Testing LLM extraction with sample message (timeout: 20s)...")
        log(f"   Message: {test_text[:80]}...")

        # Set alarm for 20 seconds
        signal.alarm(20)
        response = llm_client.extract_events(
            text=test_text,
            links=test_links,
            message_ts_dt=test_timestamp,
            channel_name="test-channel",
        )
        signal.alarm(0)  # Cancel alarm

        log(f"✅ LLM extraction successful")
        log(f"   • Is event: {response.is_event}")
        log(f"   • Events found: {len(response.events)}")

        if response.events:
            event = response.events[0]
            log(f"   • First event: {event.title}")
            log(f"   • Category: {event.category}")

        # Get metadata
        metadata = llm_client.get_call_metadata()
        log(f"   • Cost: ${metadata.cost_usd:.4f}")
        log(f"   • Latency: {metadata.latency_ms}ms")

        return True
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        log(f"❌ LLM extraction failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    # Setup timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)

    log("\n🔍 Slack Event Manager - Component Diagnostics")
    log("Testing each component separately to identify issues")
    log("⏱️ Each test has a timeout to prevent hanging\n")

    results = {}

    # Test 1: Settings
    results["settings"] = test_settings()
    if not results["settings"]:
        log("\n❌ EARLY EXIT: Cannot proceed without valid settings")
        return False

    settings = get_settings()

    # Test 2-4: Slack
    results["slack_init"], slack_client = test_slack_init(settings)
    if not results["slack_init"] or not slack_client:
        log("\n❌ EARLY EXIT: Cannot test Slack without client initialization")
        results["slack_auth"] = False
        results["slack_fetch"] = False
    else:
        results["slack_auth"] = test_slack_auth(slack_client)

        if not results["slack_auth"]:
            log("\n⚠️ Skipping message fetch test due to auth failure")
            results["slack_fetch"] = False
        else:
            # Use releases channel
            channel_id = "C04V0TK7UG6"
            results["slack_fetch"] = test_slack_fetch_messages(slack_client, channel_id)

    # Test 5-6: LLM
    results["llm_init"], llm_client = test_llm_init(settings)
    if not results["llm_init"] or not llm_client:
        log("\n⚠️ Skipping LLM extraction test due to initialization failure")
        results["llm_extract"] = False
    else:
        results["llm_extract"] = test_llm_extraction(llm_client)

    # Summary
    log("\n" + "=" * 70)
    log("DIAGNOSTIC SUMMARY")
    log("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        log(f"{status} - {test_name}")

    all_passed = all(results.values())

    log("")
    if all_passed:
        log("🎉 All tests passed! Pipeline should work correctly.")
    else:
        log("⚠️ Some tests failed. Fix these issues before running full pipeline.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
