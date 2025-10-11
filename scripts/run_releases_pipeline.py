#!/usr/bin/env python3
"""Script to fetch latest 50 messages from releases channel and run through full pipeline.

This script loads the last 50 messages from the releases channel and processes them
through the complete Slack Event Manager pipeline without any pre-filtering.

If real Slack API is not available, it will use mock data for demonstration.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.llm_client import LLMClient
from src.adapters.slack_client import SlackClient
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import get_settings
from src.use_cases.build_candidates import build_candidates_use_case
from src.use_cases.deduplicate_events import deduplicate_events_use_case
from src.use_cases.extract_events import extract_events_use_case
from src.use_cases.publish_digest import publish_digest_use_case


def create_mock_slack_client_for_releases():
    """Create a mock Slack client that simulates the releases channel with realistic data."""

    class MockSlackClient(SlackClient):
        def __init__(self):
            # Don't call super().__init__ to avoid needing real token
            pass

        def fetch_messages(
            self,
            channel_id: str,
            oldest_ts: str | None = None,
            latest_ts: str | None = None,
            limit: int = 100,
        ) -> list[dict[str, any]]:
            """Return mock messages simulating the releases channel."""
            print(f"📡 [MOCK] Fetching messages from channel {channel_id}")

            # Mock messages simulating real releases channel content
            mock_messages = [
                {
                    "type": "message",
                    "ts": "1759761669.628109",
                    "user": "U03CQE1JJ1M",
                    "text": "<!here> Hi everyone! :wave:\nWe’ve rolled out the *new Deposit screen* in Wallet! :rocket:\n This update includes improved UX, faster processing, and better error handling. Users can now deposit with 50+ currencies! Check out the updated docs at https://docs.company.com/wallet/deposit-v2",
                    "reactions": [
                        {"name": "rocket", "count": 12},
                        {"name": "tada", "count": 8},
                    ],
                    "reply_count": 5,
                    "thread_ts": "1759761669.628109",
                },
                {
                    "type": "message",
                    "ts": "1759745592.370999",
                    "user": "U03439ZFUQ3",
                    "text": "*<!here> Hi everyone!*\nIt’s finally happened! :tada: We’ve rolled out the island with beautiful animations and improved performance. The new design system makes everything feel smoother and more responsive. Major kudos to the design team! 🎨",
                    "reactions": [
                        {"name": "tada", "count": 15},
                        {"name": "art", "count": 6},
                    ],
                    "reply_count": 8,
                    "thread_ts": "1759745592.370999",
                },
                {
                    "type": "message",
                    "ts": "1759424187.804379",
                    "user": "U09CTU7UFTK",
                    "text": "Hey everyone :wave:\n\nExciting news: *three new tokens are now available in Wallet* (trade-only mode)! We've added SOL, MATIC, and AVAX support. This expands our ecosystem significantly. Trading pairs will be available next week. 📈\n\nDocs: https://docs.company.com/wallet/supported-tokens",
                    "reactions": [
                        {"name": "chart_with_upwards_trend", "count": 20},
                        {"name": "moneybag", "count": 12},
                    ],
                    "reply_count": 12,
                    "thread_ts": "1759424187.804379",
                },
                {
                    "type": "message",
                    "ts": "1759400000.000000",
                    "user": "U01ABC123DE",
                    "text": "🔧 *Scheduled maintenance window* for tomorrow 2-4 AM UTC. This will affect wallet services and P2P transfers. Expected downtime: 30 minutes. Status updates in #infrastructure",
                    "reactions": [{"name": "warning", "count": 8}],
                    "reply_count": 3,
                    "thread_ts": "1759400000.000000",
                },
                {
                    "type": "message",
                    "ts": "1759350000.000000",
                    "user": "U02DEF456GH",
                    "text": "📊 *Weekly product metrics*: P2P volume +15%, new user registrations +12%, wallet adoption rate steady at 89%. KYC completion rate improved to 94%. Details in dashboard.company.com/metrics",
                    "reactions": [{"name": "chart_with_upwards_trend", "count": 18}],
                    "reply_count": 4,
                    "thread_ts": "1759350000.000000",
                },
                {
                    "type": "message",
                    "ts": "1759300000.000000",
                    "user": "U03GHI789IJ",
                    "text": "🚨 *Security incident resolved*: Investigation completed for unusual login attempts. No data breach detected, but we've enhanced monitoring. All systems operational. Thanks for your patience!",
                    "reactions": [
                        {"name": "white_check_mark", "count": 22},
                        {"name": "shield", "count": 15},
                    ],
                    "reply_count": 6,
                    "thread_ts": "1759300000.000000",
                },
                {
                    "type": "message",
                    "ts": "1759250000.000000",
                    "user": "U04JKL012KL",
                    "text": "🎯 *Q4 OKRs update*: On track for 95% completion. Key achievements: 200k+ new users, 50% reduction in support tickets, 99.9% uptime maintained. Full report: company.okr.com/q4-2024",
                    "reactions": [
                        {"name": "target", "count": 25},
                        {"name": "trophy", "count": 18},
                    ],
                    "reply_count": 7,
                    "thread_ts": "1759250000.000000",
                },
            ]

            print(f"📨 [MOCK] Returning {len(mock_messages)} messages")
            return mock_messages

        def get_user_info(self, user_id: str) -> dict[str, any]:
            """Return mock user info."""
            return {"real_name": f"User {user_id}", "name": f"user_{user_id}"}

        def post_message(
            self, channel_id: str, blocks: list[dict[str, any]], text: str = ""
        ) -> str:
            """Mock posting message."""
            print(f"📤 [MOCK] Would post digest to channel {channel_id}")
            return "1759761669.628109"

    return MockSlackClient()


def run_releases_pipeline():
    """Run the complete pipeline on the latest 50 messages from releases channel."""

    print("🚀 Slack Event Manager - Releases Channel Pipeline")
    print("=" * 60)

    # Initialize components
    print("🔧 Initializing components...")

    settings = get_settings()

    # Try to use real Slack client, fallback to mock if not available
    slack_client = None
    try:
        slack_client = SlackClient(
            bot_token=settings.slack_bot_token.get_secret_value()
        )
        print("✅ Real Slack client initialized")

        # Test the connection by trying to fetch a small number of messages
        slack_client.fetch_messages("C04V0TK7UG6", limit=1)
        print("✅ Slack API connection verified")
    except Exception as e:
        print(f"⚠️ Real Slack client failed: {e}")
        print("🎭 Using mock data for demonstration")
        slack_client = create_mock_slack_client_for_releases()

    # Try to use real LLM client, fallback to mock if not available
    llm_client = None
    try:
        llm_client = LLMClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
        print(f"✅ Real LLM client initialized with model: {settings.llm_model}")
    except Exception as e:
        print(f"⚠️ Real LLM client failed: {e}")
        print("🤖 Using demo LLM client for demonstration")
        # Import demo LLM client from demo_e2e script
        sys.path.append(str(Path(__file__).parent))
        from demo_e2e import create_demo_llm_client

        llm_client = create_demo_llm_client()

    # Create temporary database for this run
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()

    try:
        repo = SQLiteRepository(temp_db.name)
        print("✅ Database initialized")
        print(f"📊 Database: {temp_db.name}")

        # Step 1: Fetch latest 50 messages from releases channel
        print("\n📥 STEP 1: Fetching latest 50 messages from releases channel")
        print("-" * 60)

        releases_channel_id = "C04V0TK7UG6"  # releases channel

        try:
            # Fetch latest 50 messages (no timestamp filter to get most recent)
            raw_messages = slack_client.fetch_messages(
                channel_id=releases_channel_id, oldest_ts=None, latest_ts=None, limit=50
            )

            print(f"✅ Fetched {len(raw_messages)} messages from releases channel")

            if not raw_messages:
                print("❌ No messages found in releases channel")
                return False

            # Show first few messages for context
            print("\n📨 Recent messages (first 3):")
            for i, msg in enumerate(raw_messages[:3], 1):
                text = (
                    msg.get("text", "")[:100] + "..."
                    if len(msg.get("text", "")) > 100
                    else msg.get("text", "")
                )
                ts = msg.get("ts", "")
                user = msg.get("user", "unknown")
                print(f"  {i}. [{ts}] User {user}: {text}")

        except Exception as e:
            print(f"❌ Error fetching messages: {e}")
            return False

        # Step 2: Ingest messages
        print("\n📥 STEP 2: Ingesting messages")
        print("-" * 40)

        try:
            # For this demo, we'll only process the releases channel
            releases_channel_id = "C04V0TK7UG6"

            # Get messages directly from the specific channel
            raw_messages = slack_client.fetch_messages(
                channel_id=releases_channel_id, oldest_ts=None, latest_ts=None, limit=50
            )

            print(f"📨 Fetched {len(raw_messages)} messages from releases channel")

            # Process and save messages manually
            from src.use_cases.ingest_messages import process_slack_message

            processed_messages = [
                process_slack_message(raw_msg, releases_channel_id)
                for raw_msg in raw_messages
            ]

            saved_count = repo.save_messages(processed_messages)

            ingest_result = type(
                "IngestResult",
                (),
                {
                    "messages_fetched": len(raw_messages),
                    "messages_saved": saved_count,
                    "channels_processed": [releases_channel_id],
                    "errors": [],
                },
            )()

            print(f"📋 Channels processed: {ingest_result.channels_processed}")
            print(f"📊 Total messages fetched: {ingest_result.messages_fetched}")
            print(f"💾 Messages saved: {ingest_result.messages_saved}")

            if ingest_result.errors:
                print(f"⚠️ Errors: {len(ingest_result.errors)}")
                for error in ingest_result.errors[:3]:
                    print(f"  - {error}")

        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 3: Build candidates
        print("\n🎯 STEP 3: Building candidates")
        print("-" * 40)

        try:
            # Debug: Check what messages are in the database
            print("🔍 Debug: Checking messages in database...")
            messages = repo.get_new_messages_for_candidates()
            print(f"📨 Found {len(messages)} messages for candidate building")

            for i, msg in enumerate(messages[:3], 1):
                print(
                    f"  {i}. {msg.message_id[:8]}...: {msg.text_norm[:50]}... (score will be calculated)"
                )

            candidate_result = build_candidates_use_case(
                repository=repo,
                settings=settings,
            )

            print(f"✅ Messages processed: {candidate_result.messages_processed}")
            print(f"✅ Candidates created: {candidate_result.candidates_created}")
            print(f"✅ Average score: {candidate_result.average_score:.2f}")

            if candidate_result.candidates_created == 0:
                print(
                    "ℹ️ No candidates created - checking if messages meet scoring criteria..."
                )

                # Show scoring details for first message
                if messages:
                    print("🔍 Debugging first message scoring...")
                    msg = messages[0]
                    print(f"   Message channel: {msg.channel}")
                    channel_config = settings.get_channel_config(msg.channel)
                    print(f"   Channel config found: {channel_config is not None}")

                    if channel_config:
                        print(f"   Channel name: {channel_config.channel_name}")
                        print(f"   Threshold: {channel_config.threshold_score}")
                        print(f"   Keyword weight: {channel_config.keyword_weight}")

                        try:
                            from src.services import scoring_engine

                            score, features = scoring_engine.score_message(
                                msg, channel_config
                            )
                            print(
                                f"🔍 First message scoring: {score:.2f} (threshold: {channel_config.threshold_score})"
                            )
                            print(
                                f"   Features: keywords={features.has_keywords}, mentions={features.has_mention}, reactions={features.reaction_count}, replies={features.reply_count}"
                            )

                            # Show actual text content for debugging
                            print(f"   Text sample: {msg.text_norm[:100]}...")
                            print(
                                f"   Blocks text: {msg.blocks_text[:100] if msg.blocks_text else 'None'}..."
                            )
                        except Exception as e:
                            print(f"❌ Error in scoring: {e}")
                            import traceback

                            traceback.print_exc()
                    else:
                        print("❌ No channel config found for message channel")

        except Exception as e:
            print(f"❌ Error during candidate building: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 4: Extract events
        print("\n🤖 STEP 4: Extracting events with LLM")
        print("-" * 40)

        try:
            extraction_result = extract_events_use_case(
                llm_client=llm_client,
                repository=repo,
                settings=settings,
                batch_size=50,
                check_budget=False,  # Disable budget check for this run
            )

            print(f"✅ Candidates processed: {extraction_result.candidates_processed}")
            print(f"✅ Events extracted: {extraction_result.events_extracted}")
            print(f"✅ LLM calls made: {extraction_result.llm_calls}")
            print(f"✅ Cache hits: {extraction_result.cache_hits}")
            print(f"✅ Total cost: ${extraction_result.total_cost_usd:.4f}")

            if extraction_result.errors:
                print(f"⚠️ Errors: {len(extraction_result.errors)}")
                for error in extraction_result.errors[:3]:
                    print(f"  - {error}")

        except Exception as e:
            print(f"❌ Error during event extraction: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 5: Deduplicate events
        print("\n🔗 STEP 5: Deduplicating events")
        print("-" * 40)

        try:
            dedup_result = deduplicate_events_use_case(
                repository=repo,
                settings=settings,
                lookback_days=7,
            )

            print(f"✅ New events: {dedup_result.new_events}")
            print(f"✅ Merged events: {dedup_result.merged_events}")
            print(f"✅ Total events: {dedup_result.total_events}")

        except Exception as e:
            print(f"❌ Error during deduplication: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 6: Show results
        print("\n📋 STEP 6: Results summary")
        print("-" * 40)

        # Get events for display
        events = repo.get_events_in_window(
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow(),  # Last 30 days
        )

        if events:
            print(f"✅ Found {len(events)} events in the last 30 days")

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
            print("\n📊 Events by category:")
            for category, cat_events in events_by_category.items():
                print(f"  {category}: {len(cat_events)} events")

            # Show recent events (last 5)
            print("\n🕐 Most recent events:")
            recent_events = sorted(events, key=lambda x: x.event_date, reverse=True)[:5]
            for event in recent_events:
                date_str = event.event_date.strftime("%d.%m.%Y %H:%M UTC")
                print(f"  • {event.title} ({date_str}) - {event.category}")

        else:
            print("📭 No events found")

        # Step 7: Generate digest
        print("\n📋 STEP 7: Generating digest")
        print("-" * 40)

        try:
            digest_result = publish_digest_use_case(
                slack_client=slack_client,
                repository=repo,
                settings=settings,
                lookback_hours=24 * 7,  # Last 7 days
                target_channel=settings.slack_digest_channel_id,
                dry_run=True,  # Don't actually post
            )

            print(f"✅ Digest generated: {digest_result.messages_posted} messages")
            print(f"✅ Events included: {digest_result.events_included}")
            print(f"✅ Target channel: {digest_result.channel}")

        except Exception as e:
            print(f"❌ Error during digest generation: {e}")
            import traceback

            traceback.print_exc()

        print("\n🎉 Pipeline completed successfully!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up temporary database
        try:
            os.unlink(temp_db.name)
            print(f"🗑️ Cleaned up temporary database: {temp_db.name}")
        except:
            pass


if __name__ == "__main__":
    success = run_releases_pipeline()
    sys.exit(0 if success else 1)
