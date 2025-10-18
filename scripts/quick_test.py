#!/usr/bin/env python3
"""Quick sanity check - tests only basic connectivity without hanging."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.repository_factory import create_repository
from src.config.settings import Settings, get_settings
from src.domain.protocols import RepositoryProtocol


def log(msg: str) -> None:
    """Print with immediate flush."""
    print(msg)
    sys.stdout.flush()


def main() -> bool:
    """Run quick tests."""
    log("\n🔍 Quick Sanity Check")
    log("=" * 70)

    # Test 1: Settings
    log("\n1️⃣ Testing settings...")
    try:
        settings = get_settings()
        log(f"   ✅ Model: {settings.llm_model}")
        log(f"   ✅ Temperature: {settings.llm_temperature}")
    except Exception as e:
        log(f"   ❌ Failed: {e}")
        return False

    # Test 2: Slack auth only (no messages)
    log("\n2️⃣ Testing Slack auth (no message fetch)...")
    try:
        from src.adapters.slack_client import SlackClient

        slack_client = SlackClient(
            bot_token=settings.slack_bot_token.get_secret_value()
        )

        # Only test auth, don't fetch messages
        response = slack_client.client.auth_test()
        if response["ok"]:
            log(f"   ✅ Slack auth OK: {response.get('user')}")
        else:
            log(f"   ❌ Slack auth failed: {response.get('error')}")
            return False
    except Exception as e:
        log(f"   ❌ Failed: {e}")
        return False

    # Test 3: LLM client init only (no actual call)
    log("\n3️⃣ Testing LLM client init...")
    try:
        from src.adapters.llm_client import LLMClient

        LLMClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=5,
        )
        log("   ✅ LLM client initialized")
    except Exception as e:
        log(f"   ❌ Failed: {e}")
        return False

    # Test 4: Database
    log("\n4️⃣ Testing database...")
    try:
        import tempfile

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()

        temp_settings: Settings = settings.model_copy(
            update={"db_path": temp_db.name, "database_type": "sqlite"}
        )
        repository: RepositoryProtocol = create_repository(temp_settings)
        # Explicitly close repository if supported
        close_method = getattr(repository, "close", None)
        if callable(close_method):
            close_method()
        log("   ✅ Database initialized")

        import os

        os.unlink(temp_db.name)
    except Exception as e:
        log(f"   ❌ Failed: {e}")
        return False

    log("\n" + "=" * 70)
    log("✅ All basic checks passed!")
    log("\n💡 Note: Skipped actual Slack message fetch due to rate limits")
    log("💡 Run full pipeline with: python scripts/run_releases_pipeline_real.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
