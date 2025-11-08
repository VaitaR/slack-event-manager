from __future__ import annotations

"""Entry point for the deduplication worker."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import pipeline_runtime
from src.adapters.repository_factory import create_repository
from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.services.task_queue_factory import (
    TaskQueueUnavailableError,
    resolve_task_queue,
)
from src.use_cases.pipeline_factories import create_deduplication_handler
from src.workers.pipeline import DedupWorker

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deduplication worker")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait between lease attempts when the queue is idle",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Process a single task and exit",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Emit logs in JSON format",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    settings = get_settings()
    pipeline_runtime.initialize_logging(settings, json_logs=args.json_logs)

    controller = pipeline_runtime.create_shutdown_controller()
    pipeline_runtime.install_signal_handlers(controller)

    repository = create_repository(settings)
    try:
        task_queue = resolve_task_queue(repository)
    except TaskQueueUnavailableError as exc:
        logger.error("task_queue_unavailable", error=str(exc))
        return 1

    deduplicate_events = create_deduplication_handler(
        settings=settings,
        repository=repository,
    )

    worker = DedupWorker(
        task_queue=task_queue,
        deduplicate_events=deduplicate_events,
    )

    pipeline_runtime.run_worker_loop(
        worker,
        controller,
        poll_interval=args.poll_interval_seconds,
        run_once=args.run_once,
        idle_backoff_seconds=1.0,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
