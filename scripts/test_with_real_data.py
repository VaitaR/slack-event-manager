#!/usr/bin/env python3
"""Test pipeline with real data - keep database for inspection."""

import os
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Final, TextIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.llm_client import VERBOSE_ENV_FLAG, LLMClient
from src.adapters.repository_factory import create_repository
from src.adapters.slack_client import SlackClient
from src.config.settings import Settings, get_settings
from src.domain.models import MessageSource
from src.domain.protocols import RepositoryProtocol
from src.services.importance_scorer import ImportanceScorer
from src.use_cases.build_candidates import build_candidates_use_case
from src.use_cases.deduplicate_events import deduplicate_events_use_case
from src.use_cases.extract_events import build_object_registry, extract_events_use_case
from src.use_cases.ingest_messages import process_slack_message

# Setup logging to file and console
LOG_FILE: Final[str] = "pipeline_detailed.log"
DEFAULT_MESSAGE_LIMIT: Final[int] = 5
DEFAULT_BATCH_LIMIT: Final[int] = 5

log_handle: TextIO | None = None


def parse_args() -> Namespace:
    """Parse command line arguments for configuring the test run."""

    parser = ArgumentParser(
        description="Run the real data pipeline test with optional limits."
    )
    parser.add_argument(
        "--message-limit",
        type=int,
        default=DEFAULT_MESSAGE_LIMIT,
        help=(
            "Maximum number of Slack messages to fetch. "
            "Use --full-run to process the historical 20-message batch."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_LIMIT,
        help=(
            "Maximum number of candidates to process during LLM extraction. "
            "Set to a small number to avoid exceeding CI timeouts."
        ),
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help=(
            "Process the full historical batch (20 messages, all candidates). "
            "This matches legacy behaviour and may take >10 minutes."
        ),
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Setup logging to file."""
    global log_handle
    log_handle = open(LOG_FILE, "w", encoding="utf-8")
    log(f"📝 Logging to: {LOG_FILE}")
    log(f"⏰ Started at: {datetime.now().isoformat()}")
    log("=" * 80)
    log("")


def log(msg: str) -> None:
    """Print with immediate flush to console and file."""
    print(msg)
    sys.stdout.flush()
    if log_handle:
        log_handle.write(msg + "\n")
        log_handle.flush()


def inspect_database(db_path: str, stage: str = "") -> None:
    """Show what's in the database with detailed stats."""
    import sqlite3

    if stage:
        log(f"\n{'=' * 80}")
        log(f"📊 DATABASE INSPECTION - {stage}")
        log("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get message count
    cursor.execute("SELECT COUNT(*) FROM raw_slack_messages")
    msg_count = cursor.fetchone()[0]
    log(f"\n📊 Database contains {msg_count} messages")

    # Show first 3 messages with full details
    cursor.execute(
        """
        SELECT message_id, text, ts, text_norm, links_norm, anchors
        FROM raw_slack_messages
        ORDER BY ts DESC
        LIMIT 3
    """
    )

    log("\n📨 Sample messages:")
    log("=" * 80)
    for i, row in enumerate(cursor.fetchall(), 1):
        msg_id, text, ts, text_norm, links_norm, anchors = row
        text_preview = (text[:150] + "...") if len(text) > 150 else text
        norm_preview = (
            (text_norm[:150] + "...")
            if text_norm and len(text_norm) > 150
            else text_norm
        )
        log(f"\n{i}. Message ID: {msg_id[:8]}...")
        log(f"   Timestamp: {ts}")
        log(f"   Text: {text_preview}")
        log(f"   Text norm: {norm_preview}")
        log(f"   Links: {links_norm}")
        log(f"   Anchors: {anchors}")

    # Get candidate count and distribution
    cursor.execute("SELECT COUNT(*) FROM event_candidates")
    cand_count = cursor.fetchone()[0]
    log(f"\n🎯 Database contains {cand_count} candidates")

    # Show ALL candidates with scores
    if cand_count > 0:
        cursor.execute(
            """
            SELECT message_id, text_norm, score, status, features_json
            FROM event_candidates
            ORDER BY score DESC
        """
        )

        log("\n🎯 ALL Candidates (sorted by score):")
        log("=" * 80)
        for i, row in enumerate(cursor.fetchall(), 1):
            msg_id, text, score, status, features = row
            text_preview = (text[:200] + "...") if len(text) > 200 else text
            log(f"\n{i}. Message ID: {msg_id[:8]}...")
            log(f"   Score: {score:.2f}")
            log(f"   Status: {status}")
            log(f"   Text: {text_preview}")
            if features:
                log(f"   Features: {features[:200]}")

    # Get event count and details
    cursor.execute("SELECT COUNT(*) FROM events")
    event_count = cursor.fetchone()[0]
    log(f"\n📝 Database contains {event_count} events")

    if event_count > 0:
        cursor.execute(
            """
            SELECT event_id, message_id, source_msg_event_idx, title, category,
                   event_date, confidence, dedup_key, version
            FROM events
            ORDER BY event_date DESC
        """
        )

        log("\n📝 ALL Events:")
        log("=" * 80)
        for i, row in enumerate(cursor.fetchall(), 1):
            (
                event_id,
                message_id,
                idx,
                title,
                category,
                event_date,
                confidence,
                dedup_key,
                version,
            ) = row
            log(f"\n{i}. {title}")
            log(f"   Event ID: {event_id[:16]}...")
            log(f"   Message ID: {message_id[:8]}... (index: {idx})")
            log(f"   Category: {category}")
            log(f"   Date: {event_date}")
            log(f"   Confidence: {confidence}")
            log(f"   Dedup key: {dedup_key[:16]}...")
            log(f"   Version: {version}")

    # LLM call statistics
    cursor.execute(
        "SELECT COUNT(*), SUM(cost_usd), SUM(tokens_in), SUM(tokens_out) FROM llm_calls"
    )
    llm_row = cursor.fetchone()
    if llm_row[0]:
        calls, total_cost, total_in, total_out = llm_row
        log("\n💰 LLM Statistics:")
        log(f"   Total calls: {calls}")
        log(f"   Total cost: ${total_cost:.6f}")
        log(f"   Total tokens IN: {total_in}")
        log(f"   Total tokens OUT: {total_out}")

    conn.close()


def resolve_limits(args: Namespace) -> tuple[int, int | None]:
    """Resolve message and batch size limits based on CLI arguments."""

    if args.full_run:
        return 20, None

    if args.message_limit <= 0:
        raise ValueError("--message-limit must be positive")

    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive when provided")

    return args.message_limit, args.batch_size


def main(args: Namespace) -> bool:
    """Run pipeline test with real data."""
    setup_logging()

    message_limit, batch_size = resolve_limits(args)

    log("\n🚀 Pipeline Test with Real Data")
    log("=" * 70)
    log("")

    # Initialize
    log("⏳ Step 0: Initializing...")
    settings: Settings = get_settings()
    log("   Settings loaded:")
    log(f"   - LLM model: {settings.llm_model}")
    log(f"   - LLM temperature: {settings.llm_temperature}")
    log(f"   - Threshold score: {settings.threshold_score_default}")
    log(f"   - Dedup date window: {settings.dedup_date_window_hours}h")
    log(f"   - Dedup title similarity: {settings.dedup_title_similarity}")

    slack_client = SlackClient(bot_token=settings.slack_bot_token.get_secret_value())
    verbose_enabled = os.getenv(VERBOSE_ENV_FLAG, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if verbose_enabled:
        log("🔍 Verbose LLM logging enabled (redacted previews)")
    else:
        log(
            "🔒 Verbose LLM logging disabled; set "
            f"{VERBOSE_ENV_FLAG}=1 for additional diagnostics"
        )

    llm_client = LLMClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=30,  # Increased for complex messages
        verbose=verbose_enabled,
    )

    # Use persistent database
    db_path = "data/test_real_pipeline.db"

    # Remove old test db if exists
    if Path(db_path).exists():
        os.unlink(db_path)
        log("🗑️ Removed old test database")

    try:
        temp_settings: Settings = settings.model_copy(
            update={"db_path": db_path, "database_type": "sqlite"}
        )
        repo: RepositoryProtocol = create_repository(temp_settings)
        object_registry = build_object_registry(settings)
        importance_scorer = ImportanceScorer()
        log("✅ Components initialized")
        log(f"📊 Database: {db_path}")
        log("")

        # Step 1: Fetch messages
        log(f"⏳ Step 1: Fetching {message_limit} messages from releases channel...")
        try:
            import time

            time.sleep(2)  # Avoid rate limit

            raw_messages = slack_client.fetch_messages(
                channel_id="C04V0TK7UG6",
                limit=message_limit,
            )
            log(f"✅ Fetched {len(raw_messages)} messages")
        except Exception as e:
            log(f"❌ Failed to fetch messages: {e}")
            return False

        if not raw_messages:
            log("❌ No messages returned")
            return False

        # Show sample messages
        log("\n📨 First 3 raw messages (for analysis):")
        for i, msg in enumerate(raw_messages[:3], 1):
            msg_text = msg.get("text", "")[:300]
            log(f"\n{i}. TS: {msg.get('ts')}")
            log(f"   Text: {msg_text}")

        # Step 2: Ingest
        log("")
        log("⏳ Step 2: Ingesting messages...")
        processed_messages = [
            process_slack_message(msg, "C04V0TK7UG6") for msg in raw_messages
        ]
        saved_count = repo.save_messages(processed_messages)
        log(f"✅ Saved {saved_count} messages")

        # Inspect after ingestion
        inspect_database(db_path, "After Ingestion")

        # Step 3: Build candidates
        log("")
        log("⏳ Step 3: Building candidates...")
        candidate_result = build_candidates_use_case(
            repository=repo,
            settings=settings,
        )
        log(f"✅ Created {candidate_result.candidates_created} candidates")
        if candidate_result.candidates_created > 0:
            log(f"   Average score: {candidate_result.average_score:.2f}")
            log(f"   Max score: {candidate_result.max_score:.2f}")
            log(f"   Min score: {candidate_result.min_score:.2f}")

        if candidate_result.candidates_created == 0:
            log("ℹ️ No candidates - messages don't meet scoring criteria")
            log("✅ Pipeline test completed")
            log(f"📊 Database saved at: {db_path}")
            log(f"📝 Log saved at: {LOG_FILE}")
            return True

        # Inspect after candidates
        inspect_database(db_path, "After Candidate Building")

        # Step 4: Extract with LLM (process ALL candidates)
        log("")
        llm_scope = (
            "all candidates" if batch_size is None else f"batch size {batch_size}"
        )
        log(f"⏳ Step 4: Extracting events with LLM ({llm_scope})...")
        log("   Note: Will show full LLM prompts and responses in verbose mode")
        log("")

        extraction_result = extract_events_use_case(
            llm_client=llm_client,
            repository=repo,
            settings=settings,
            source_id=MessageSource.SLACK,  # Real data test - Slack only
            batch_size=batch_size,
            check_budget=False,
            object_registry=object_registry,
            importance_scorer=importance_scorer,
        )
        log("")
        log("✅ Extraction completed:")
        log(f"   Events extracted: {extraction_result.events_extracted}")
        log(f"   Candidates processed: {extraction_result.candidates_processed}")
        log(f"   LLM calls: {extraction_result.llm_calls}")
        log(f"   Cost: ${extraction_result.total_cost_usd:.6f}")

        if extraction_result.errors:
            log(f"⚠️ Errors encountered: {len(extraction_result.errors)}")
            for err in extraction_result.errors:
                log(f"   - {err}")

        # Inspect after extraction
        inspect_database(db_path, "After LLM Extraction")

        # Step 5: Deduplicate
        log("")
        log("⏳ Step 5: Deduplicating...")
        dedup_result = deduplicate_events_use_case(
            repository=repo,
            settings=settings,
            lookback_days=7,
        )
        log("✅ Deduplication completed:")
        log(f"   Total unique events: {dedup_result.total_events}")
        log(f"   New events: {dedup_result.new_events}")
        log(f"   Merged events: {dedup_result.merged_events}")

        # Final inspection
        inspect_database(db_path, "Final State")

        # Summary
        log("")
        log("=" * 70)
        log("✅ PIPELINE TEST COMPLETED!")
        log("=" * 70)
        log(f"   📊 Messages fetched: {saved_count}")
        log(f"   🎯 Candidates created: {candidate_result.candidates_created}")
        log(f"   📝 Events extracted: {extraction_result.events_extracted}")
        log(f"   🔄 Events after dedup: {dedup_result.total_events}")
        log(f"   💰 Total cost: ${extraction_result.total_cost_usd:.6f}")
        log(f"   💾 Database: {db_path}")
        log(f"   📝 Detailed log: {LOG_FILE}")

        if log_handle:
            log_handle.close()

        return True

    except Exception as e:
        log(f"❌ Pipeline failed: {e}")
        import traceback

        tb = traceback.format_exc()
        log(f"\n{tb}")
        if log_handle:
            log_handle.close()
        return False


if __name__ == "__main__":
    cli_args = parse_args()
    success = main(cli_args)
    sys.exit(0 if success else 1)
