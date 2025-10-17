"""Main pipeline runner script.

Runs complete ETL pipeline locally:
1. Ingest messages from Slack
2. Build candidates
3. Extract events (LLM)
4. Deduplicate events
5. (Optional) Publish digest

Can run once or continuously with configurable interval.
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import pytz

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.llm_client import LLMClient
from src.adapters.repository_factory import create_repository
from src.adapters.slack_client import SlackClient
from src.config.settings import get_settings
from src.domain.protocols import RepositoryProtocol
from src.use_cases.build_candidates import build_candidates_use_case
from src.use_cases.deduplicate_events import deduplicate_events_use_case
from src.use_cases.extract_events import extract_events_use_case
from src.use_cases.ingest_messages import ingest_messages_use_case
from src.use_cases.publish_digest import publish_digest_use_case

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum: int, frame: object) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    print(f"\n⚠️  Received {sig_name}, shutting down gracefully...")
    _shutdown_requested = True


def setup_logging(log_dir: Path) -> None:
    """Setup logging to file and stdout.

    Args:
        log_dir: Directory for log files
    """
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def run_single_iteration(
    slack_client: SlackClient,
    repository: RepositoryProtocol,
    llm_client: LLMClient,
    settings: object,
    args: argparse.Namespace,
    backfill_from_date: datetime | None = None,
) -> None:
    """Run single pipeline iteration.

    Args:
        slack_client: Slack API client
        repository: Data repository
        llm_client: LLM API client
        settings: Application settings
        args: CLI arguments
        backfill_from_date: Optional backfill start date
    """
    print("\n" + "=" * 60)
    print("STEP 1: Ingesting messages from Slack")
    print("=" * 60)
    ingest_result = ingest_messages_use_case(
        slack_client=slack_client,
        repository=repository,
        settings=settings,
        lookback_hours=args.lookback_hours,
        backfill_from_date=backfill_from_date,
    )
    print(f"✓ Fetched: {ingest_result.messages_fetched} messages")
    print(f"✓ Saved: {ingest_result.messages_saved} messages")
    print(f"✓ Channels: {', '.join(ingest_result.channels_processed)}")
    if ingest_result.errors:
        print(f"⚠ Errors: {len(ingest_result.errors)}")
        for error in ingest_result.errors:
            print(f"  - {error}")

    print("\n" + "=" * 60)
    print("STEP 2: Building event candidates")
    print("=" * 60)
    candidate_result = build_candidates_use_case(
        repository=repository,
        settings=settings,
    )
    print(f"✓ Messages processed: {candidate_result.messages_processed}")
    print(f"✓ Candidates created: {candidate_result.candidates_created}")
    print(f"✓ Average score: {candidate_result.average_score:.2f}")

    if not args.skip_llm:
        print("\n" + "=" * 60)
        print("STEP 3: Extracting events with LLM")
        print("=" * 60)
        extraction_result = extract_events_use_case(
            llm_client=llm_client,
            repository=repository,
            settings=settings,
            batch_size=50,
            check_budget=True,
        )
        print(f"✓ Candidates processed: {extraction_result.candidates_processed}")
        print(f"✓ Events extracted: {extraction_result.events_extracted}")
        print(f"✓ LLM calls: {extraction_result.llm_calls}")
        print(f"✓ Cache hits: {extraction_result.cache_hits}")
        print(f"✓ Total cost: ${extraction_result.total_cost_usd:.4f}")
        if extraction_result.errors:
            print(f"⚠ Errors: {len(extraction_result.errors)}")
            for error in extraction_result.errors[:5]:  # Show first 5
                print(f"  - {error}")

        print("\n" + "=" * 60)
        print("STEP 4: Deduplicating events")
        print("=" * 60)
        dedup_result = deduplicate_events_use_case(
            repository=repository,
            settings=settings,
            lookback_days=7,
        )
        print(f"✓ New events: {dedup_result.new_events}")
        print(f"✓ Merged events: {dedup_result.merged_events}")
        print(f"✓ Total events: {dedup_result.total_events}")

    if args.publish:
        print("\n" + "=" * 60)
        print("STEP 5: Publishing digest")
        print("=" * 60)
        digest_result = publish_digest_use_case(
            slack_client=slack_client,
            repository=repository,
            settings=settings,
            lookback_hours=48,
            dry_run=args.dry_run,
        )
        print(f"✓ Events included: {digest_result.events_included}")
        print(f"✓ Messages posted: {digest_result.messages_posted}")
        print(f"✓ Channel: {digest_result.channel}")
        if args.dry_run:
            print("  (DRY RUN - not actually posted)")


def main() -> int:
    """Run pipeline (once or continuously).

    Returns:
        Exit code (0 = success, 1 = error)
    """
    global _shutdown_requested

    parser = argparse.ArgumentParser(
        description="Run Slack Event Manager pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run once
  python scripts/run_pipeline.py

  # Run continuously every hour
  python scripts/run_pipeline.py --interval-seconds 3600

  # Backfill from specific date (first run only)
  python scripts/run_pipeline.py --backfill-from 2025-09-01

  # Run with publish
  python scripts/run_pipeline.py --interval-seconds 3600 --publish
        """,
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=0,
        help="Run interval in seconds (0 = run once and exit, >0 = run continuously)",
    )
    parser.add_argument(
        "--backfill-from",
        type=str,
        default=None,
        help="Backfill from date YYYY-MM-DD (first run only, default: 30 days ago)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish digest after processing",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=None,
        help="Hours to look back for messages (deprecated, use --backfill-from)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM extraction step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no digest posting)",
    )

    args = parser.parse_args()

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Setup logging
    log_dir = Path("logs")
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    # Parse backfill date if provided
    backfill_from_date: datetime | None = None
    if args.backfill_from:
        try:
            backfill_from_date = datetime.strptime(
                args.backfill_from, "%Y-%m-%d"
            ).replace(tzinfo=pytz.UTC)
            logger.info(f"Backfill from date: {backfill_from_date.isoformat()}")
        except ValueError as e:
            logger.error(f"Invalid date format for --backfill-from: {e}")
            return 1

    # Load settings
    logger.info("Loading configuration...")
    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return 1

    # Initialize adapters
    logger.info("Initializing clients...")
    try:
        slack_client = SlackClient(
            bot_token=settings.slack_bot_token.get_secret_value()
        )
        repository = create_repository(settings)
        llm_client = LLMClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        return 1

    # Determine run mode
    if args.interval_seconds > 0:
        logger.info(f"🔄 Running continuously with {args.interval_seconds}s interval")
        logger.info("Press Ctrl+C to stop gracefully")
    else:
        logger.info("▶️  Running once")

    iteration = 0
    while not _shutdown_requested:
        iteration += 1
        start_time = time.time()

        try:
            logger.info("=" * 80)
            logger.info(
                f"Pipeline iteration #{iteration} started at {datetime.now().isoformat()}"
            )
            logger.info("=" * 80)

            run_single_iteration(
                slack_client=slack_client,
                repository=repository,
                llm_client=llm_client,
                settings=settings,
                args=args,
                backfill_from_date=backfill_from_date,
            )

            elapsed = time.time() - start_time
            logger.info("=" * 80)
            logger.info(
                f"✅ Pipeline iteration #{iteration} completed in {elapsed:.1f}s"
            )
            logger.info("=" * 80)

        except Exception as e:
            logger.error(
                f"❌ Pipeline iteration #{iteration} failed: {e}", exc_info=True
            )
            if args.interval_seconds == 0:
                # Single run mode: exit with error
                return 1
            # Continuous mode: log error and continue

        # Exit if single-run mode
        if args.interval_seconds == 0:
            logger.info("Single run completed, exiting")
            break

        # Sleep until next iteration (check shutdown flag frequently)
        if not _shutdown_requested:
            logger.info(
                f"💤 Sleeping for {args.interval_seconds}s until next iteration..."
            )
            sleep_start = time.time()
            while time.time() - sleep_start < args.interval_seconds:
                if _shutdown_requested:
                    break
                time.sleep(min(1, args.interval_seconds))  # Check every second

    if _shutdown_requested:
        logger.info("🛑 Shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
