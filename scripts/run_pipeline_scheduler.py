from __future__ import annotations

"""Periodic scheduler that enqueues pipeline tasks."""

import argparse
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import pipeline_runtime
from src.adapters.repository_factory import create_repository
from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.services.task_queue_factory import (
    TaskQueueUnavailableError,
    resolve_task_queue,
)
from src.use_cases.pipeline_scheduler import enqueue_pipeline_iteration

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pipeline task scheduler")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=300.0,
        help="Interval between scheduling iterations",
    )
    parser.add_argument(
        "--include-dedup/--skip-dedup",
        dest="include_dedup",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to enqueue deduplication tasks on each iteration",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Emit logs in JSON format",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single scheduling iteration and exit",
    )
    args = parser.parse_args(argv)
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than 0")
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

    def _schedule() -> None:
        correlation_id = str(uuid4())
        enqueue_pipeline_iteration(
            task_queue,
            correlation_id=correlation_id,
            include_dedup=args.include_dedup,
        )

    pipeline_runtime.run_scheduler_loop(
        controller=controller,
        interval_seconds=args.interval_seconds,
        run_once=args.run_once,
        action=_schedule,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
