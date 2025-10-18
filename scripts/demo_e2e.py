#!/usr/bin/env python3
"""End-to-End Demo Script for Slack Event Manager.

Demonstrates the complete pipeline by:
1. Fetching real Slack messages (or using mock data)
2. Processing through the full pipeline
3. Generating and displaying a digest in the terminal

Usage:
    python scripts/demo_e2e.py [--real] [--channel CHANNEL_ID] [--hours HOURS]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.llm_client import LLMClient
from src.adapters.repository_factory import create_repository
from src.adapters.slack_client import SlackClient
from src.config.settings import Settings, get_settings
from src.domain.models import EventCategory, LLMEvent, LLMResponse
from src.domain.protocols import RepositoryProtocol
from src.use_cases.build_candidates import build_candidates_use_case
from src.use_cases.extract_events import extract_events_use_case
from src.use_cases.ingest_messages import ingest_messages_use_case

# Mock messages for demo when no real Slack access
MOCK_MESSAGES = [
    {
        "type": "message",
        "ts": "1728000000.123456",
        "user": "U123456",
        "text": "🚀 Release v2.1.0 is now live! Major updates include improved P2P transfers, enhanced KYC verification, and bug fixes for wallet connectivity. ETA for full rollout: Oct 15, 2024. Check docs at https://docs.company.com/v2.1",
        "reactions": [{"name": "rocket", "count": 15}, {"name": "eyes", "count": 8}],
        "reply_count": 5,
        "thread_ts": "1728000000.123456",  # Root message
    },
    {
        "type": "message",
        "ts": "1728003600.654321",
        "user": "U789012",
        "text": "🔧 Scheduled maintenance window for tomorrow 2-4 AM UTC. This will affect wallet services and P2P transfers. Downtime expected: 30 minutes. Status updates in #infrastructure",
        "reactions": [{"name": "warning", "count": 12}],
        "reply_count": 2,
        "thread_ts": "1728003600.654321",
    },
    {
        "type": "message",
        "ts": "1728007200.987654",
        "user": "U345678",
        "text": "📢 New feature announcement: Enhanced AML screening for all transactions over $1000. This improves compliance and reduces false positives. Go-live: next Monday. Training materials at confluence.company.com/aml-screening-v2",
        "reactions": [{"name": "shield", "count": 18}, {"name": "book", "count": 7}],
        "reply_count": 8,
        "thread_ts": "1728007200.987654",
    },
    {
        "type": "message",
        "ts": "1728010800.111111",
        "user": "U901234",
        "text": "⚠️ Security incident detected: Unusual login attempts from multiple IPs. Investigation underway. Affected services: wallet authentication. ETA for resolution: 2 hours.",
        "reactions": [{"name": "rotating_light", "count": 25}],
        "reply_count": 12,
        "thread_ts": "1728010800.111111",
    },
    {
        "type": "message",
        "ts": "1728014400.222222",
        "user": "U567890",
        "text": "🎯 Weekly product metrics: P2P volume +12%, new user registrations +8%, wallet adoption rate steady at 87%. Details in dashboard.company.com/metrics",
        "reactions": [{"name": "chart_with_upwards_trend", "count": 14}],
        "reply_count": 3,
        "thread_ts": "1728014400.222222",
    },
]


def create_mock_slack_client() -> SlackClient:
    """Create a mock Slack client that returns demo messages."""

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
        ) -> list[dict[str, Any]]:
            """Return mock messages instead of real API call."""
            print(f"📡 [MOCK] Fetching messages from channel {channel_id}")

            # Filter messages by timestamp if provided
            filtered_messages = MOCK_MESSAGES
            if oldest_ts:
                oldest_timestamp = float(oldest_ts)
                # For demo, return all messages if timestamp filter would exclude them
                filtered_messages = [
                    msg
                    for msg in MOCK_MESSAGES
                    if float(msg["ts"]) >= oldest_timestamp
                    or oldest_timestamp > 1728000000  # Demo mode
                ]

            print(f"📨 [MOCK] Returning {len(filtered_messages)} messages")
            return filtered_messages

        def get_user_info(self, user_id: str) -> dict[str, Any]:
            """Return mock user info."""
            return {"real_name": f"User {user_id}", "name": f"user_{user_id}"}

        def post_message(
            self, channel_id: str, blocks: list[dict[str, Any]], text: str = ""
        ) -> str:
            """Mock posting message."""
            print(f"📤 [MOCK] Would post digest to channel {channel_id}")
            return "1728000000.000000"

    return MockSlackClient()


def create_demo_llm_client() -> LLMClient:
    """Create a mock LLM client that returns realistic responses."""

    class DemoLLMClient(LLMClient):
        def __init__(self):
            # Don't call super().__init__ to avoid needing real API key
            pass

        def extract_events(
            self,
            text: str,
            links: list[str],
            message_ts_dt: datetime,
            channel_name: str = "",
        ) -> Any:
            """Return mock LLM response based on message content."""

            events = []

            # Analyze text for events
            # Create unique events based on message content
            if "release" in text.lower() or "v2.1" in text:
                events.append(
                    LLMEvent(
                        title="🚀 Product Release v2.1.0",
                        summary="Major release with P2P improvements, KYC enhancements, and bug fixes",
                        category=EventCategory.PRODUCT,
                        event_date="2024-10-15T10:00:00Z",
                        event_end=None,
                        impact_area=["p2p", "kyc", "wallet"],
                        tags=["release", "v2.1", "improvement"],
                        links=["https://docs.company.com/v2.1"],
                        confidence=0.95,
                        source_channels=[f"#{channel_name}"] if channel_name else [],
                    )
                )

            if "maintenance" in text.lower() or "downtime" in text.lower():
                events.append(
                    LLMEvent(
                        title="🔧 Scheduled Maintenance Window",
                        summary="30-minute downtime for wallet services and P2P transfers",
                        category=EventCategory.PROCESS,
                        event_date="2024-10-11T02:00:00Z",
                        event_end="2024-10-11T04:00:00Z",
                        impact_area=["wallet", "p2p"],
                        tags=["maintenance", "downtime"],
                        links=[],
                        confidence=0.90,
                        source_channels=[f"#{channel_name}"] if channel_name else [],
                    )
                )

            if "security incident" in text.lower() or "unusual login" in text.lower():
                events.append(
                    LLMEvent(
                        title="⚠️ Security Incident Investigation",
                        summary="Investigation of unusual login attempts affecting wallet authentication",
                        category=EventCategory.RISK,
                        event_date="2024-10-10T15:00:00Z",
                        event_end="2024-10-10T17:00:00Z",
                        impact_area=["security", "wallet"],
                        tags=["security", "incident"],
                        links=[],
                        confidence=0.85,
                        source_channels=[f"#{channel_name}"] if channel_name else [],
                    )
                )

            if "metrics" in text.lower() and (
                "p2p" in text.lower() or "volume" in text.lower()
            ):
                events.append(
                    LLMEvent(
                        title="📊 Weekly Product Metrics Update",
                        summary="P2P volume increased 12%, new user registrations up 8%",
                        category=EventCategory.MARKETING,
                        event_date="2024-10-10T12:00:00Z",
                        event_end=None,
                        impact_area=["p2p", "user_growth"],
                        tags=["metrics", "growth"],
                        links=["https://dashboard.company.com/metrics"],
                        confidence=0.80,
                        source_channels=[f"#{channel_name}"] if channel_name else [],
                    )
                )

            return LLMResponse(
                is_event=len(events) > 0,
                overflow_note=(
                    None
                    if len(events) <= 5
                    else f"Found {len(events)} events, showing top 5"
                ),
                events=events[:5],  # Limit to 5 events
            )

        def extract_events_with_retry(self, *args, **kwargs) -> Any:
            """Mock retry wrapper."""
            return self.extract_events(*args, **kwargs)

        def get_call_metadata(self) -> Any:
            """Return mock metadata."""
            from src.domain.models import LLMCallMetadata

            return LLMCallMetadata(
                message_id="demo",
                prompt_hash="demo_hash",
                model="gpt-5-nano-demo",
                tokens_in=150,
                tokens_out=200,
                cost_usd=0.015,
                latency_ms=500,
                cached=False,
                ts=datetime.utcnow(),
            )

    return DemoLLMClient()


def format_digest_for_terminal(events: list[Any], channel_name: str = "demo") -> str:
    """Format digest for terminal display."""
    if not events:
        return "📭 No events found in the selected time period."

    output = []
    output.append("📅" + "=" * 60)
    output.append(f"🎯 SLACK EVENT DIGEST - {channel_name.upper()}")
    output.append("=" * 60)
    output.append("")

    # Group events by category for better organization
    events_by_category = {}
    for event in events:
        category = event.category.value.upper()
        if category not in events_by_category:
            events_by_category[category] = []
        events_by_category[category].append(event)

    # Display events by category
    category_order = ["PRODUCT", "PROCESS", "MARKETING", "RISK", "ORG", "UNKNOWN"]

    for category in category_order:
        if category in events_by_category:
            events_list = events_by_category[category]
            output.append(f"🔸 {category}")
            output.append("-" * 20)

            for event in events_list:
                # Format date
                date_str = event.event_date.strftime("%d.%m.%Y %H:%M UTC")
                if event.event_end:
                    end_str = event.event_end.strftime("%H:%M UTC")
                    date_str = f"{date_str} - {end_str}"

                # Format confidence
                confidence_icon = (
                    "✅"
                    if event.confidence >= 0.8
                    else "⚠️"
                    if event.confidence >= 0.6
                    else "❓"
                )

                output.append(f"  {confidence_icon} {event.title}")
                output.append(f"    📅 {date_str}")
                if event.summary:
                    output.append(f"    💬 {event.summary}")
                if event.impact_area:
                    output.append(f"    🎯 Areas: {', '.join(event.impact_area)}")
                if event.tags:
                    output.append(f"    🏷️ Tags: {', '.join(event.tags)}")
                if event.links:
                    output.append(f"    🔗 Links: {', '.join(event.links)}")
                output.append("")

    output.append("=" * 60)
    output.append(f"📊 Summary: {len(events)} events extracted from demo data")
    output.append("=" * 60)

    return "\n".join(output)


def run_e2e_demo(
    use_real_slack: bool = False, channel_id: str | None = None, hours_back: int = 24
) -> None:
    """Run end-to-end demo of the Slack Event Manager pipeline."""

    print("🚀 Starting Slack Event Manager E2E Demo")
    print("=" * 60)

    # Initialize components
    print("🔧 Initializing components...")

    settings: Settings = get_settings()
    slack_client = None

    if use_real_slack:
        print("📡 Using REAL Slack API (requires valid tokens)")
        try:
            slack_client = SlackClient(
                bot_token=settings.slack_bot_token.get_secret_value()
            )
            print("✅ Real Slack client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize real Slack client: {e}")
            print("💡 Falling back to mock data for demo")
            use_real_slack = False

    if not use_real_slack or slack_client is None:
        print("🎭 Using MOCK data for demonstration")
        slack_client = create_mock_slack_client()

    if use_real_slack:
        # Use real LLM client for real data processing
        from src.adapters.llm_client import LLMClient

        llm_client = LLMClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=1.0,  # Default temperature for gpt-5-nano
            timeout=settings.llm_timeout_seconds,
        )
        print("🤖 Using REAL LLM client for event extraction")
    else:
        # Use demo LLM client (doesn't need real API key)
        llm_client = create_demo_llm_client()

    # Use temporary file for demo
    import tempfile

    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()

    repository: RepositoryProtocol | None = None
    try:
        temp_settings: Settings = settings.model_copy(
            update={"db_path": temp_db.name, "database_type": "sqlite"}
        )
        repository = create_repository(temp_settings)
        print("✅ Components initialized")

        # For real Slack demo, we'll fetch real messages instead of mock data
        print("📡 Ready to fetch real messages from Slack...")

        # For real demo, messages will be fetched in the next step

    except Exception as e:
        print(f"❌ Failed to create demo database: {e}")
        return

    print("\n📥 STEP 1: Ingesting messages")
    print("-" * 40)

    # Ingest messages from Slack
    print("📥 Ingesting messages from Slack...")

    # Get channels to process
    channels_to_process = (
        [channel_id]
        if channel_id
        else [ch.channel_id for ch in settings.slack_channels]
    )
    print(f"📋 Processing channels: {', '.join(channels_to_process)}")

    # Run ingestion use case
    try:
        ingest_result = ingest_messages_use_case(
            slack_client=slack_client,
            repository=repository,
            settings=settings,
            lookback_hours=(
                hours_back if not use_real_slack else 24
            ),  # Use shorter window for demo
        )

        print(f"✅ Messages fetched: {ingest_result.messages_fetched}")
        print(f"✅ Messages saved: {ingest_result.messages_saved}")
        print(f"✅ Channels processed: {len(ingest_result.channels_processed)}")

        if ingest_result.errors:
            print(f"⚠️ Errors: {len(ingest_result.errors)}")
            for error in ingest_result.errors[:3]:
                print(f"  - {error}")

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n🎯 STEP 2: Building candidates")
    print("-" * 40)

    # Build candidates
    try:
        candidate_result = build_candidates_use_case(
            repository=repository,
            settings=settings,
        )

        print(f"✅ Messages processed: {candidate_result.messages_processed}")
        print(f"✅ Candidates created: {candidate_result.candidates_created}")
        print(f"✅ Average score: {candidate_result.average_score:.2f}")

        # Show candidate creation summary
        if candidate_result.candidates_created == 0:
            print(
                "ℹ️ No candidates created - this might be normal if no messages were fetched or they don't meet scoring criteria"
            )
        else:
            print(
                f"✅ Successfully created {candidate_result.candidates_created} candidates for processing"
            )

    except Exception as e:
        print(f"❌ Error during candidate building: {e}")
        return

    print("\n🤖 STEP 3: Extracting events with LLM")
    print("-" * 40)

    # Extract events
    try:
        extraction_result = extract_events_use_case(
            llm_client=llm_client,
            repository=repository,
            settings=settings,
            batch_size=50,
            check_budget=False,  # Disable budget check for demo
        )

        print(f"✅ Candidates processed: {extraction_result.candidates_processed}")
        print(f"✅ Events extracted: {extraction_result.events_extracted}")
        print(f"✅ LLM calls made: {extraction_result.llm_calls}")
        print(f"✅ Total cost: ${extraction_result.total_cost_usd:.4f}")

        if extraction_result.errors:
            print(f"⚠️ Errors: {len(extraction_result.errors)}")
            for error in extraction_result.errors[:3]:
                print(f"  - {error}")

    except Exception as e:
        print(f"❌ Error during event extraction: {e}")
        return

    print("\n🔗 STEP 4: Deduplicating events")
    print("-" * 40)

    # Deduplicate events (skip for demo to preserve all events)
    print("📝 Skipping deduplication for demo (preserving all extracted events)")

    print("\n📋 STEP 5: Generating digest")
    print("-" * 40)

    # Generate digest
    try:
        # Get events from a wider time window to include our mock events (Oct 10-15, 2024)
        from datetime import datetime

        import pytz

        start_time = datetime(2024, 10, 9, tzinfo=pytz.UTC)  # Start from Oct 9
        end_time = datetime(2024, 10, 16, tzinfo=pytz.UTC)  # End at Oct 16

        events = repository.get_events_in_window(start_time, end_time)

        if events:
            print(f"✅ Found {len(events)} events in time window")

            # Format and display digest
            digest_text = format_digest_for_terminal(events, "demo-channel")
            print("\n" + digest_text)

        else:
            print("📭 No events found in the time window")

    except Exception as e:
        print(f"❌ Error during digest generation: {e}")
        return

    print("\n🎉 E2E Demo completed successfully!")
    print("=" * 60)

    # Clean up temporary database file
    try:
        if repository is not None:
            close_method = getattr(repository, "close", None)
            if callable(close_method):
                close_method()
        os.unlink(temp_db.name)
    except:
        pass


def main() -> None:
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end demo of Slack Event Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_e2e.py                    # Run with mock data
  python scripts/demo_e2e.py --real            # Try to use real Slack API
  python scripts/demo_e2e.py --hours 72        # Process last 72 hours
        """,
    )

    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real Slack API (requires valid tokens in .env)",
    )

    parser.add_argument(
        "--channel",
        type=str,
        help="Specific channel ID to process (for real Slack mode)",
    )

    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours to look back for messages (default: 24)",
    )

    args = parser.parse_args()

    try:
        run_e2e_demo(
            use_real_slack=args.real, channel_id=args.channel, hours_back=args.hours
        )
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
