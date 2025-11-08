from __future__ import annotations

"""Entry point for the candidate scheduling worker."""

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
from src.use_cases.pipeline_factories import create_llm_scheduler
from src.workers.pipeline import ExtractionWorker

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the extraction scheduler worker")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait between queue lease attempts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Default batch size when leasing candidates for LLM extraction",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Process a single batch of tasks and exit",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Emit logs in JSON format",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    return args


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

    scheduler = create_llm_scheduler(
        settings=settings,
        repository=repository,
        task_queue=task_queue,
        default_batch_size=args.batch_size,
    )

    worker = ExtractionWorker(
        task_queue=task_queue,
        schedule_llm_tasks=scheduler,
    )

    pipeline_runtime.run_worker_loop(
        worker,
        controller,
        poll_interval=args.poll_interval_seconds,
        run_once=args.run_once,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
