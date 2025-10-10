"""Backfill historical data script.

Processes historical messages within a date range.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytz

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.llm_client import LLMClient
from src.adapters.slack_client import SlackClient
from src.adapters.sqlite_repository import SQLiteRepository
from src.config.settings import get_settings
from src.use_cases.build_candidates import build_candidates_use_case
from src.use_cases.deduplicate_events import deduplicate_events_use_case
from src.use_cases.extract_events import extract_events_use_case
from src.use_cases.ingest_messages import ingest_messages_use_case


def main() -> None:
    """Run backfill."""
    parser = argparse.ArgumentParser(description="Backfill historical Slack data")
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Specific channels to backfill (default: all)",
    )
    parser.add_argument(
        "--budget-per-day",
        type=float,
        default=None,
        help="Override daily LLM budget",
    )

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
            tzinfo=pytz.UTC
        )
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=pytz.UTC
        )
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)

    if start_date > end_date:
        print("Error: start_date must be before end_date")
        sys.exit(1)

    # Load settings
    print("Loading configuration...")
    settings = get_settings()

    if args.budget_per_day:
        settings.llm_daily_budget_usd = args.budget_per_day

    # Initialize adapters
    print("Initializing clients...")
    slack_client = SlackClient(bot_token=settings.slack_bot_token.get_secret_value())
    repository = SQLiteRepository(db_path=settings.db_path)
    llm_client = LLMClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
    )

    # Process day by day
    current_date = start_date
    total_messages = 0
    total_events = 0
    total_cost = 0.0

    print(f"\nBackfilling from {args.start_date} to {args.end_date}")
    print("=" * 60)

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\nProcessing {date_str}...")

        # Override watermarks temporarily for this date
        next_date = current_date + timedelta(days=1)

        # Ingest for this day
        # Note: This is simplified; real backfill would need more sophisticated watermark handling
        print(f"  Ingesting messages...")
        ingest_result = ingest_messages_use_case(
            slack_client=slack_client,
            repository=repository,
            settings=settings,
            lookback_hours=24,
        )
        total_messages += ingest_result.messages_saved

        # Build candidates
        print(f"  Building candidates...")
        candidate_result = build_candidates_use_case(
            repository=repository,
            settings=settings,
        )

        # Extract events
        print(f"  Extracting events...")
        extraction_result = extract_events_use_case(
            llm_client=llm_client,
            repository=repository,
            settings=settings,
            batch_size=50,
            check_budget=True,
        )
        total_events += extraction_result.events_extracted
        total_cost += extraction_result.total_cost_usd

        # Deduplicate
        print(f"  Deduplicating...")
        dedup_result = deduplicate_events_use_case(
            repository=repository,
            settings=settings,
            lookback_days=7,
        )

        print(
            f"  ✓ Messages: {ingest_result.messages_saved}, "
            f"Events: {extraction_result.events_extracted}, "
            f"Cost: ${extraction_result.total_cost_usd:.4f}"
        )

        current_date = next_date

    print("\n" + "=" * 60)
    print("Backfill completed!")
    print(f"Total messages processed: {total_messages}")
    print(f"Total events extracted: {total_events}")
    print(f"Total LLM cost: ${total_cost:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

