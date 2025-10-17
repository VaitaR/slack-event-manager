"""Standalone digest generation script.

Generates and optionally posts event digest for a specific date.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytz

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.repository_factory import create_repository
from src.adapters.slack_client import SlackClient
from src.config.settings import get_settings
from src.use_cases.publish_digest import publish_digest_use_case


def main() -> None:
    """Generate digest."""
    parser = argparse.ArgumentParser(description="Generate event digest")
    parser.add_argument(
        "--date",
        default="yesterday",
        help='Date for digest (YYYY-MM-DD or "yesterday", "today")',
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=None,
        help="Hours to look back from date (default: from config)",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="Override target channel ID",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum confidence score 0.0-1.0 (default: from config)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum events to include (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build digest but don't post to Slack",
    )

    args = parser.parse_args()

    # Parse date
    if args.date == "yesterday":
        target_date = datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(days=1)
    elif args.date == "today":
        target_date = datetime.utcnow().replace(tzinfo=pytz.UTC)
    else:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").replace(
                tzinfo=pytz.UTC
            )
        except ValueError as e:
            print(f"Error parsing date: {e}")
            print('Use YYYY-MM-DD format or "yesterday"/"today"')
            sys.exit(1)

    # Load settings
    print("Loading configuration...")
    settings = get_settings()

    # Initialize adapters
    print("Initializing clients...")
    slack_client = SlackClient(bot_token=settings.slack_bot_token.get_secret_value())
    repository = create_repository(settings)

    date_str = target_date.strftime("%Y-%m-%d")
    print(f"\nGenerating digest for {date_str}")
    print(f"Lookback: {args.lookback_hours or settings.digest_lookback_hours} hours")
    print(
        f"Min confidence: {args.min_confidence or settings.digest_min_confidence:.2f}"
    )
    print(f"Max events: {args.max_events or settings.digest_max_events or 'unlimited'}")
    print("=" * 60)

    # Generate and publish digest
    digest_result = publish_digest_use_case(
        slack_client=slack_client,
        repository=repository,
        settings=settings,
        lookback_hours=args.lookback_hours,
        target_channel=args.channel,
        dry_run=args.dry_run,
        min_confidence=args.min_confidence,
        max_events=args.max_events,
    )

    print(f"\n✓ Events included: {digest_result.events_included}")
    print(f"✓ Messages posted: {digest_result.messages_posted}")
    print(f"✓ Channel: {digest_result.channel}")

    if args.dry_run:
        print("\n⚠ DRY RUN - Digest was NOT posted to Slack")
    else:
        print("\n✅ Digest posted successfully!")


if __name__ == "__main__":
    main()
