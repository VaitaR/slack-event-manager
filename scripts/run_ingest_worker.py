from __future__ import annotations

"""Entry point for the asynchronous ingestion worker."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import SecretStr

from scripts import pipeline_runtime
from src.adapters.repository_factory import create_repository
from src.adapters.slack_client import SlackClient
from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.services.task_queue_factory import (
    TaskQueueUnavailableError,
    resolve_task_queue,
)
from src.use_cases.pipeline_factories import create_slack_ingestion_handlers
from src.workers.pipeline import IngestWorker

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ingestion worker")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait between lease attempts when the queue is idle",
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

    try:
        slack_client = SlackClient(bot_token=_extract_secret(settings.slack_bot_token))
    except Exception:  # noqa: BLE001
        logger.exception("slack_client_initialization_failed")
        return 1

    ingest_messages, build_candidates = create_slack_ingestion_handlers(
        settings=settings,
        repository=repository,
        slack_client=slack_client,
    )

    worker = IngestWorker(
        task_queue=task_queue,
        ingest_messages=ingest_messages,
        build_candidates=build_candidates,
    )

    pipeline_runtime.run_worker_loop(
        worker,
        controller,
        poll_interval=args.poll_interval_seconds,
        run_once=args.run_once,
    )

    return 0


def _extract_secret(secret: SecretStr) -> str:
    value = secret.get_secret_value()
    if not value:
        msg = "Secret value is empty"
        raise ValueError(msg)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
