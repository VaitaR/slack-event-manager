#!/usr/bin/env python3
"""Production script to fetch latest 50 messages from releases channel and run through full pipeline.

This script loads the last 50 messages from the releases channel and processes them
through the complete Slack Event Manager pipeline using ONLY real API credentials.
No mock data is used - all operations require valid Slack and OpenAI API keys.
"""

import sys
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def log(msg: str) -> None:
    """Print with immediate flush to avoid hanging output."""
    print(msg)
    sys.stdout.flush()


from src.adapters.slack_client import SlackClient
from src.adapters.sqlite_repository import SQLiteRepository
from src.adapters.llm_client import LLMClient
from src.config.settings import get_settings
from src.use_cases.ingest_messages import ingest_messages_use_case
from src.use_cases.build_candidates import build_candidates_use_case
from src.use_cases.extract_events import extract_events_use_case
from src.use_cases.deduplicate_events import deduplicate_events_use_case
from src.use_cases.publish_digest import publish_digest_use_case


def run_releases_pipeline_real():
    """Run the complete pipeline on the latest 50 messages from releases channel using real APIs."""

    log("🚀 Slack Event Manager - Releases Channel Pipeline (REAL DATA ONLY)")
    log("=" * 70)
    log("")

    # Initialize components with real APIs
    log("🔧 Step 0: Initializing components with real API credentials...")
    log("-" * 70)

    settings = get_settings()

    # Initialize real Slack client
    try:
        log("⏳ Initializing Slack client...")
        slack_client = SlackClient(
            bot_token=settings.slack_bot_token.get_secret_value()
        )
        log("✅ Real Slack client initialized successfully")
    except Exception as e:
        log(f"❌ Failed to initialize Slack client: {e}")
        log("💡 Make sure SLACK_BOT_TOKEN is set correctly in .env file")
        return False

    # Test Slack API connection
    try:
        log("⏳ Testing Slack API connection...")
        test_result = slack_client.fetch_messages("C04V0TK7UG6", limit=1)
        log("✅ Slack API connection verified")
    except Exception as e:
        log(f"❌ Slack API connection failed: {e}")
        log(
            "💡 Check that the bot is a member of the releases channel and has proper permissions"
        )
        return False

    # Initialize real LLM client
    try:
        log("⏳ Initializing LLM client...")
        llm_client = LLMClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=10,  # Shorter timeout to avoid hanging
        )
        log(f"✅ Real LLM client initialized with model: {settings.llm_model}")
        log(f"   Temperature: {settings.llm_temperature}, Timeout: 10s")
    except Exception as e:
        log(f"❌ Failed to initialize LLM client: {e}")
        log("💡 Make sure OPENAI_API_KEY is set correctly in .env file")
        return False

    # Create temporary database for this run
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()

    try:
        repo = SQLiteRepository(temp_db.name)
        log("✅ Database initialized")
        log(f"📊 Database: {temp_db.name}")
        log("")

        # Step 1: Fetch latest 50 messages from releases channel
        log("📥 STEP 1: Fetching latest 50 messages from releases channel")
        log("-" * 60)

        releases_channel_id = "C04V0TK7UG6"  # releases channel

        try:
            # Fetch latest 10 messages (reduced to avoid rate limits)
            raw_messages = slack_client.fetch_messages(
                channel_id=releases_channel_id, oldest_ts=None, latest_ts=None, limit=10
            )

            log(
                f"✅ Successfully fetched {len(raw_messages)} real messages from releases channel"
            )

            if not raw_messages:
                log("❌ No messages found in releases channel")
                log("💡 Check that the channel exists and the bot has access to it")
                return False

            # Show first few messages for context
            log("\n📨 Recent messages (first 3):")
            for i, msg in enumerate(raw_messages[:3], 1):
                text = (
                    msg.get("text", "")[:100] + "..."
                    if len(msg.get("text", "")) > 100
                    else msg.get("text", "")
                )
                ts = msg.get("ts", "")
                user = msg.get("user", "unknown")
                log(f"  {i}. [{ts}] User {user}: {text}")

        except Exception as e:
            log(f"❌ Error fetching messages from Slack: {e}")
            log("💡 Check your SLACK_BOT_TOKEN and channel permissions")
            return False

        # Step 2: Ingest messages
        log("")
        log("📥 STEP 2: Ingesting and processing messages")
        log("-" * 50)

        try:
            log("⏳ Fetching messages from releases channel...")
            # Process only the releases channel
            releases_channel_id = "C04V0TK7UG6"

            # Get messages directly from the specific channel
            raw_messages = slack_client.fetch_messages(
                channel_id=releases_channel_id, oldest_ts=None, latest_ts=None, limit=50
            )

            log(
                f"📨 Processing {len(raw_messages)} real messages from releases channel"
            )

            # Process and save messages
            from src.use_cases.ingest_messages import process_slack_message

            processed_messages = [
                process_slack_message(raw_msg, releases_channel_id)
                for raw_msg in raw_messages
            ]

            saved_count = repo.save_messages(processed_messages)

            log(f"✅ Successfully saved {saved_count} messages to database")
            log(f"📋 Channel processed: {releases_channel_id}")
            log(f"📊 Messages fetched: {len(raw_messages)}")
            log(f"💾 Messages saved: {saved_count}")

        except Exception as e:
            log(f"❌ Error during message ingestion: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 3: Build candidates
        log("")
        log("🎯 STEP 3: Building event candidates")
        log("-" * 40)

        try:
            log("⏳ Building candidates from messages...")
            # Get messages for candidate building
            messages = repo.get_new_messages_for_candidates()
            log(f"📨 Found {len(messages)} messages for candidate building")

            if not messages:
                log("❌ No messages found for candidate building")
                return False

            candidate_result = build_candidates_use_case(
                repository=repo,
                settings=settings,
            )

            log(f"✅ Messages processed: {candidate_result.messages_processed}")
            log(f"✅ Candidates created: {candidate_result.candidates_created}")
            log(f"✅ Average score: {candidate_result.average_score:.2f}")

            if candidate_result.candidates_created == 0:
                log("ℹ️ No candidates created - messages may not meet scoring criteria")
                return True  # This is not necessarily an error

        except Exception as e:
            log(f"❌ Error during candidate building: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 4: Extract events with LLM
        log("")
        log("🤖 STEP 4: Extracting events with LLM")
        log("-" * 40)

        try:
            log("⏳ Starting LLM extraction...")

            extraction_result = extract_events_use_case(
                llm_client=llm_client,
                repository=repo,
                settings=settings,
                batch_size=50,
                check_budget=False,  # Disable budget check for this run
            )

            log(f"✅ Candidates processed: {extraction_result.candidates_processed}")
            log(f"✅ Events extracted: {extraction_result.events_extracted}")
            log(f"✅ LLM calls made: {extraction_result.llm_calls}")
            log(f"✅ Cache hits: {extraction_result.cache_hits}")
            log(f"✅ Total cost: ${extraction_result.total_cost_usd:.4f}")

            if extraction_result.errors:
                log(f"⚠️ Errors during extraction: {len(extraction_result.errors)}")
                for error in extraction_result.errors[:3]:
                    log(f"  - {error}")

        except Exception as e:
            log(f"❌ Error during event extraction: {e}")
            log("💡 Check your OPENAI_API_KEY and LLM model settings")
            import traceback

            traceback.print_exc()
            return False

        # Step 5: Deduplicate events
        log("")
        log("🔗 STEP 5: Deduplicating events")
        log("-" * 40)

        try:
            log("⏳ Deduplicating events...")
            dedup_result = deduplicate_events_use_case(
                repository=repo,
                settings=settings,
                lookback_days=7,
            )

            log(f"✅ New events: {dedup_result.new_events}")
            log(f"✅ Merged events: {dedup_result.merged_events}")
            log(f"✅ Total unique events: {dedup_result.total_events}")

        except Exception as e:
            log(f"❌ Error during deduplication: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 6: Show results summary
        log("")
        log("📋 STEP 6: Results summary")
        log("-" * 40)

        # Get events for display
        events = repo.get_events_in_window(
            datetime.utcnow() - timedelta(days=30), datetime.utcnow()  # Last 30 days
        )

        if events:
            log(f"✅ Found {len(events)} events in the last 30 days")

            # Group by category
            events_by_category = {}
            for event in events:
                category = (
                    event.category.value
                    if hasattr(event.category, "value")
                    else str(event.category)
                )
                if category not in events_by_category:
                    events_by_category[category] = []
                events_by_category[category].append(event)

            # Display summary
            log("\n📊 Events by category:")
            for category, cat_events in events_by_category.items():
                log(f"  {category}: {len(cat_events)} events")

            # Show recent events (last 5)
            log("\n🕐 Most recent events:")
            recent_events = sorted(events, key=lambda x: x.event_date, reverse=True)[:5]
            for event in recent_events:
                date_str = event.event_date.strftime("%d.%m.%Y %H:%M UTC")
                log(f"  • {event.title} ({date_str}) - {event.category}")

        else:
            log("📭 No events found")

        # Step 7: Generate digest (optional)
        log("")
        log("📋 STEP 7: Generating digest")
        log("-" * 40)

        try:
            log("⏳ Generating digest...")
            digest_result = publish_digest_use_case(
                slack_client=slack_client,
                repository=repo,
                settings=settings,
                lookback_hours=24 * 7,  # Last 7 days
                target_channel=settings.slack_digest_channel_id,
                dry_run=True,  # Don't actually post for safety
            )

            log(f"✅ Digest generated: {digest_result.messages_posted} messages")
            log(f"✅ Events included: {digest_result.events_included}")
            log(f"✅ Target channel: {digest_result.channel}")
            log("ℹ️ Digest was generated in dry-run mode (not posted to Slack)")

        except Exception as e:
            log(f"❌ Error during digest generation: {e}")
            import traceback

            traceback.print_exc()

        log("")
        log("🎉 Pipeline completed successfully with real data!")
        log("=" * 70)

        # Show final statistics
        log("📈 Final Statistics:")
        log(
            f"   • Messages processed: {len(messages) if 'messages' in locals() else 0}"
        )
        log(
            f"   • Candidates created: {candidate_result.candidates_created if 'candidate_result' in locals() else 0}"
        )
        log(
            f"   • Events extracted: {extraction_result.events_extracted if 'extraction_result' in locals() else 0}"
        )
        log(
            f"   • LLM API calls: {extraction_result.llm_calls if 'extraction_result' in locals() else 0}"
        )
        log(
            f"   • Total cost: ${extraction_result.total_cost_usd:.4f}"
            if "extraction_result" in locals()
            else "   • Total cost: $0.0000"
        )

        return True

    except Exception as e:
        log(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up temporary database
        try:
            os.unlink(temp_db.name)
            log(f"🗑️ Cleaned up temporary database: {temp_db.name}")
        except:
            pass


if __name__ == "__main__":
    success = run_releases_pipeline_real()
    sys.exit(0 if success else 1)
