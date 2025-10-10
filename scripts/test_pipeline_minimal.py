#!/usr/bin/env python3
"""Minimal pipeline test - processes only 5 messages to test full flow quickly."""

import sys
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


def log(msg: str) -> None:
    """Print with immediate flush."""
    print(msg)
    sys.stdout.flush()


from src.adapters.slack_client import SlackClient
from src.adapters.sqlite_repository import SQLiteRepository
from src.adapters.llm_client import LLMClient
from src.config.settings import get_settings
from src.use_cases.ingest_messages import process_slack_message
from src.use_cases.build_candidates import build_candidates_use_case
from src.use_cases.extract_events import extract_events_use_case
from src.use_cases.deduplicate_events import deduplicate_events_use_case


def main():
    """Run minimal pipeline test."""
    log("\n🚀 Minimal Pipeline Test (5 messages only)")
    log("=" * 70)
    log("")

    # Initialize
    log("⏳ Step 0: Initializing...")
    settings = get_settings()
    slack_client = SlackClient(bot_token=settings.slack_bot_token.get_secret_value())
    llm_client = LLMClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=10,
    )

    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()

    try:
        repo = SQLiteRepository(temp_db.name)
        log("✅ Components initialized")
        log("")

        # Step 1: Fetch minimal messages
        log("⏳ Step 1: Fetching 5 messages from releases channel...")
        try:
            # Add small delay to avoid rate limit
            import time

            time.sleep(2)

            raw_messages = slack_client.fetch_messages(
                channel_id="C04V0TK7UG6", limit=5
            )
            log(f"✅ Fetched {len(raw_messages)} messages")
        except Exception as e:
            log(f"❌ Failed to fetch messages: {e}")
            log("💡 Slack API might be rate limited. Try again in 1 minute.")
            return False

        if not raw_messages:
            log("❌ No messages returned")
            return False

        # Step 2: Ingest
        log("")
        log("⏳ Step 2: Ingesting messages...")
        processed_messages = [
            process_slack_message(msg, "C04V0TK7UG6") for msg in raw_messages
        ]
        saved_count = repo.save_messages(processed_messages)
        log(f"✅ Saved {saved_count} messages")

        # Step 3: Build candidates
        log("")
        log("⏳ Step 3: Building candidates...")
        candidate_result = build_candidates_use_case(
            repository=repo,
            settings=settings,
        )
        log(f"✅ Created {candidate_result.candidates_created} candidates")

        if candidate_result.candidates_created == 0:
            log("ℹ️ No candidates - messages don't meet scoring criteria")
            log("✅ Pipeline test completed (no events to extract)")
            return True

        # Step 4: Extract with LLM
        log("")
        log("⏳ Step 4: Extracting events with LLM...")
        extraction_result = extract_events_use_case(
            llm_client=llm_client,
            repository=repo,
            settings=settings,
            batch_size=10,
            check_budget=False,
        )
        log(f"✅ Extracted {extraction_result.events_extracted} events")
        log(f"   LLM calls: {extraction_result.llm_calls}")
        log(f"   Cost: ${extraction_result.total_cost_usd:.4f}")

        # Step 5: Deduplicate
        log("")
        log("⏳ Step 5: Deduplicating...")
        dedup_result = deduplicate_events_use_case(
            repository=repo,
            settings=settings,
            lookback_days=7,
        )
        log(f"✅ Unique events: {dedup_result.total_events}")

        # Summary
        log("")
        log("=" * 70)
        log("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
        log("=" * 70)
        log(f"   📊 Messages: {saved_count}")
        log(f"   🎯 Candidates: {candidate_result.candidates_created}")
        log(f"   📝 Events: {extraction_result.events_extracted}")
        log(f"   💰 Cost: ${extraction_result.total_cost_usd:.4f}")

        return True

    except Exception as e:
        log(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            os.unlink(temp_db.name)
        except:
            pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
