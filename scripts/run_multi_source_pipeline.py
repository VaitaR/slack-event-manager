"""Multi-source pipeline orchestrator.

Runs the complete ETL pipeline for all enabled message sources (Slack, Telegram, etc.):
1. Loop through enabled sources from config
2. For each source:
   a. Create source-specific clients (message client, LLM client)
   b. Ingest messages
   c. Build candidates
   d. Extract events with source-specific LLM prompt
   e. Deduplicate events (with strict source isolation)
3. (Optional) Publish digest across all sources

Can run once or continuously with configurable interval.
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pytz

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.llm_client import LLMClient
from src.adapters.message_client_factory import get_message_client
from src.adapters.repository_factory import create_repository
from src.adapters.slack_client import SlackClient
from src.adapters.sqlite_repository import SQLiteRepository
from src.adapters.telegram_client import TelegramClient
from src.config.settings import Settings, get_settings
from src.domain.models import MessageSource
from src.domain.protocols import RepositoryProtocol
from src.use_cases.build_candidates import build_candidates_use_case
from src.use_cases.deduplicate_events import deduplicate_events_use_case
from src.use_cases.extract_events import extract_events_use_case
from src.use_cases.ingest_messages import ingest_messages_use_case
from src.use_cases.ingest_telegram_messages import ingest_telegram_messages_use_case
from src.use_cases.publish_digest import publish_digest_use_case

# Global flag for graceful shutdown
_shutdown_requested = False


class PipelineStats(TypedDict):
    """Pipeline statistics structure."""

    messages_fetched: int
    messages_saved: int
    candidates_created: int
    events_extracted: int
    events_merged: int
    llm_calls: int
    total_cost_usd: float
    channels_processed: list[str]
    errors: list[str]


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


def run_source_pipeline(
    source_id: MessageSource,
    repository: RepositoryProtocol,
    settings: Settings,
    args: argparse.Namespace,
    backfill_from_date: datetime | None = None,
) -> PipelineStats:
    """Run pipeline for a single message source.

    Args:
        source_id: Message source identifier (SLACK, TELEGRAM, etc.)
        repository: Data repository
        settings: Application settings
        args: CLI arguments
        backfill_from_date: Optional backfill start date

    Returns:
        Dictionary with pipeline statistics
    """
    logger = logging.getLogger(__name__)
    stats: PipelineStats = {
        "messages_fetched": 0,
        "messages_saved": 0,
        "candidates_created": 0,
        "events_extracted": 0,
        "events_merged": 0,
        "llm_calls": 0,
        "total_cost_usd": 0.0,
        "channels_processed": [],
        "errors": [],
    }

    # Get source configuration
    source_config = settings.get_source_config(source_id)
    if not source_config:
        logger.warning(f"⚠️  No configuration found for source: {source_id.value}")
        return stats

    if not source_config.enabled:
        logger.info(f"⏭️  Source {source_id.value} is disabled, skipping")
        return stats

    print("\n" + "=" * 80)
    print(f"🔄 PROCESSING SOURCE: {source_id.value.upper()}")
    print("=" * 80)

    # Get bot token from environment
    bot_token_env = (
        source_config.bot_token_env or f"{source_id.value.upper()}_BOT_TOKEN"
    )
    bot_token = os.getenv(bot_token_env)
    if not bot_token:
        # Fallback to default token for backward compatibility
        if source_id == MessageSource.SLACK:
            bot_token = settings.slack_bot_token.get_secret_value()
        else:
            logger.warning(
                f"⚠️  No bot token found for {source_id.value} (env: {bot_token_env})"
            )
            return stats

    # Create source-specific message client
    try:
        message_client = get_message_client(source_id, bot_token)
        logger.info(f"✓ Created {source_id.value} message client")
    except Exception as e:
        logger.error(f"❌ Failed to create message client for {source_id.value}: {e}")
        return stats

    if isinstance(message_client, SlackClient):
        slack_client = message_client
    elif isinstance(message_client, TelegramClient):
        # Process Telegram source
        logger.info(f"📥 Processing Telegram source: {source_id.value}")
        telegram_client = message_client
    else:
        logger.warning(
            f"⚠️  Message client type {type(message_client)} not supported yet; skipping source"
        )
        return stats

    # Create source-specific LLM client
    try:
        llm_settings = source_config.llm_settings or {}
        llm_temperature = llm_settings.get("temperature", settings.llm_temperature)
        llm_timeout = llm_settings.get("timeout", settings.llm_timeout_seconds)
        prompt_file = source_config.prompt_file

        llm_client = LLMClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=llm_temperature,
            timeout=llm_timeout,
            prompt_file=prompt_file,
        )
        logger.info(
            f"✓ Created LLM client (model={settings.llm_model}, temp={llm_temperature}, prompt={prompt_file})"
        )
    except Exception as e:
        logger.error(f"❌ Failed to create LLM client: {e}")
        return stats

    # STEP 1: Ingest messages
    print("\n" + "=" * 60)
    print(f"STEP 1: Ingesting messages from {source_id.value}")
    print("=" * 60)
    try:
        if isinstance(message_client, SlackClient):
            # Slack ingestion
            ingest_result = ingest_messages_use_case(
                slack_client=slack_client,
                repository=repository,
                settings=settings,
                lookback_hours=args.lookback_hours,
                backfill_from_date=backfill_from_date,
            )
        elif isinstance(message_client, TelegramClient):
            # Telegram ingestion - requires SQLiteRepository
            sqlite_repo = (
                repository if isinstance(repository, SQLiteRepository) else None
            )
            if sqlite_repo is None:
                logger.error("❌ Telegram ingestion requires SQLiteRepository")
                stats["errors"].append("Telegram ingestion requires SQLiteRepository")
                return stats

            ingest_result = ingest_telegram_messages_use_case(
                telegram_client=telegram_client,
                repository=sqlite_repo,
                settings=settings,
                backfill_from_date=backfill_from_date,
            )
        else:
            raise ValueError(f"Unsupported message client type: {type(message_client)}")

        stats["messages_fetched"] = ingest_result.messages_fetched
        stats["messages_saved"] = ingest_result.messages_saved
        print(f"✓ Fetched: {ingest_result.messages_fetched} messages")
        print(f"✓ Saved: {ingest_result.messages_saved} messages")
        print(f"✓ Channels: {', '.join(ingest_result.channels_processed)}")
        if ingest_result.errors:
            print(f"⚠ Errors: {len(ingest_result.errors)}")
            for error in ingest_result.errors:
                print(f"  - {error}")
    except Exception as e:
        logger.error(f"❌ Ingestion failed for {source_id.value}: {e}", exc_info=True)
        return stats

    # STEP 2: Build candidates
    print("\n" + "=" * 60)
    print("STEP 2: Building event candidates")
    print("=" * 60)
    try:
        candidate_result = build_candidates_use_case(
            repository=repository,
            settings=settings,
        )
        stats["candidates_created"] = candidate_result.candidates_created
        print(f"✓ Messages processed: {candidate_result.messages_processed}")
        print(f"✓ Candidates created: {candidate_result.candidates_created}")
        print(f"✓ Average score: {candidate_result.average_score:.2f}")
    except Exception as e:
        logger.error(
            f"❌ Candidate building failed for {source_id.value}: {e}", exc_info=True
        )
        return stats

    # STEP 3: Extract events with LLM (source-specific prompt)
    if not args.skip_llm:
        print("\n" + "=" * 60)
        print(f"STEP 3: Extracting events with LLM (source: {source_id.value})")
        print("=" * 60)
        try:
            extraction_result = extract_events_use_case(
                llm_client=llm_client,  # Source-specific LLM client with custom prompt
                repository=repository,
                settings=settings,
                batch_size=50,
                check_budget=True,
            )
            stats["events_extracted"] = extraction_result.events_extracted
            stats["llm_calls"] = extraction_result.llm_calls
            stats["total_cost_usd"] = extraction_result.total_cost_usd
            print(f"✓ Candidates processed: {extraction_result.candidates_processed}")
            print(f"✓ Events extracted: {extraction_result.events_extracted}")
            print(f"✓ LLM calls: {extraction_result.llm_calls}")
            print(f"✓ Cache hits: {extraction_result.cache_hits}")
            print(f"✓ Total cost: ${extraction_result.total_cost_usd:.4f}")
            if extraction_result.errors:
                print(f"⚠ Errors: {len(extraction_result.errors)}")
                for error in extraction_result.errors[:5]:  # Show first 5
                    print(f"  - {error}")
        except Exception as e:
            logger.error(
                f"❌ Event extraction failed for {source_id.value}: {e}", exc_info=True
            )
            return stats

        # STEP 4: Deduplicate events (strict source isolation)
        print("\n" + "=" * 60)
        print(f"STEP 4: Deduplicating events (source: {source_id.value})")
        print("=" * 60)
        try:
            dedup_result = deduplicate_events_use_case(
                repository=repository,
                settings=settings,
                lookback_days=7,
                source_id=source_id,  # Strict source isolation
            )
            stats["events_merged"] = dedup_result.merged_events
            print(f"✓ New events: {dedup_result.new_events}")
            print(f"✓ Merged events: {dedup_result.merged_events}")
            print(f"✓ Total events: {dedup_result.total_events}")
        except Exception as e:
            logger.error(
                f"❌ Deduplication failed for {source_id.value}: {e}", exc_info=True
            )
            return stats

    return stats


def run_single_iteration(
    repository: RepositoryProtocol,
    settings: Settings,
    args: argparse.Namespace,
    backfill_from_date: datetime | None = None,
) -> None:
    """Run single pipeline iteration across all enabled sources.

    Args:
        repository: Data repository
        settings: Application settings
        args: CLI arguments
        backfill_from_date: Optional backfill start date
    """
    logger = logging.getLogger(__name__)

    # Get enabled sources
    enabled_sources = settings.get_enabled_sources()
    if not enabled_sources:
        logger.warning("⚠️  No enabled sources found in configuration")
        return

    # Filter by --source flag if specified
    if args.source:
        source_filter = args.source.lower()
        try:
            filtered_source = MessageSource(source_filter)
            # Check if filtered source is in enabled configs
            enabled_source_ids = [s.source_id for s in enabled_sources]
            if filtered_source in enabled_source_ids:
                # Keep only the matching config
                enabled_sources = [
                    s for s in enabled_sources if s.source_id == filtered_source
                ]
                logger.info(f"🎯 Filtering to single source: {source_filter}")
            else:
                logger.warning(
                    f"⚠️  Source '{source_filter}' is not enabled in configuration"
                )
                return
        except ValueError:
            logger.error(f"❌ Invalid source: {source_filter}")
            logger.error(
                f"   Valid sources: {', '.join(s.value for s in MessageSource)}"
            )
            return

    print(
        f"\n📋 Processing sources: {', '.join(s.source_id.value for s in enabled_sources)}"
    )

    # Aggregate statistics across all sources
    total_stats: PipelineStats = {
        "messages_fetched": 0,
        "messages_saved": 0,
        "candidates_created": 0,
        "events_extracted": 0,
        "events_merged": 0,
        "llm_calls": 0,
        "total_cost_usd": 0.0,
        "channels_processed": [],
        "errors": [],
    }

    # Process each source independently
    for source_config in enabled_sources:
        if _shutdown_requested:
            logger.info("🛑 Shutdown requested, stopping source processing")
            break

        source_stats = run_source_pipeline(
            source_id=source_config.source_id,  # Extract source_id from config
            repository=repository,
            settings=settings,
            args=args,
            backfill_from_date=backfill_from_date,
        )

        # Aggregate statistics
        total_stats["messages_fetched"] += source_stats["messages_fetched"]
        total_stats["messages_saved"] += source_stats["messages_saved"]
        total_stats["candidates_created"] += source_stats["candidates_created"]
        total_stats["events_extracted"] += source_stats["events_extracted"]
        total_stats["events_merged"] += source_stats["events_merged"]
        total_stats["llm_calls"] += source_stats["llm_calls"]
        total_stats["total_cost_usd"] += source_stats["total_cost_usd"]
        total_stats["channels_processed"].extend(source_stats["channels_processed"])
        total_stats["errors"].extend(source_stats["errors"])

    # Print aggregate statistics
    print("\n" + "=" * 80)
    print("📊 AGGREGATE STATISTICS (ALL SOURCES)")
    print("=" * 80)
    print(f"✓ Messages fetched: {total_stats['messages_fetched']}")
    print(f"✓ Messages saved: {total_stats['messages_saved']}")
    print(f"✓ Candidates created: {total_stats['candidates_created']}")
    print(f"✓ Events extracted: {total_stats['events_extracted']}")
    print(f"✓ Events merged: {total_stats['events_merged']}")
    print(f"✓ LLM calls: {total_stats['llm_calls']}")
    print(f"✓ Total cost: ${total_stats['total_cost_usd']:.4f}")

    # STEP 5: Publish digest (optional, across all sources)
    if args.publish:
        print("\n" + "=" * 60)
        print("STEP 5: Publishing digest (all sources)")
        print("=" * 60)
        try:
            # Create Slack client for digest posting
            slack_client = get_message_client(
                MessageSource.SLACK, settings.slack_bot_token.get_secret_value()
            )
            if not isinstance(slack_client, SlackClient):
                raise TypeError(
                    "Slack message client factory did not return SlackClient"
                )
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
        except Exception as e:
            logger.error(f"❌ Digest publishing failed: {e}", exc_info=True)


def main() -> int:
    """Run multi-source pipeline (once or continuously).

    Returns:
        Exit code (0 = success, 1 = error)
    """
    global _shutdown_requested

    parser = argparse.ArgumentParser(
        description="Run Slack Event Manager multi-source pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run once for all enabled sources
  python scripts/run_multi_source_pipeline.py

  # Run only for Slack
  python scripts/run_multi_source_pipeline.py --source slack

  # Run only for Telegram
  python scripts/run_multi_source_pipeline.py --source telegram

  # Run continuously every hour
  python scripts/run_multi_source_pipeline.py --interval-seconds 3600

  # Backfill from specific date (first run only)
  python scripts/run_multi_source_pipeline.py --backfill-from 2025-09-01

  # Run with publish
  python scripts/run_multi_source_pipeline.py --interval-seconds 3600 --publish
        """,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Process only specific source (slack, telegram, etc.). If not specified, processes all enabled sources.",
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
        settings: Settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return 1

    # Initialize repository
    logger.info("Initializing repository...")
    try:
        repository: RepositoryProtocol = create_repository(settings)
    except Exception as e:
        logger.error(f"Failed to initialize repository: {e}")
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
                repository=repository,
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
