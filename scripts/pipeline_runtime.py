from __future__ import annotations

"""Common runtime helpers for pipeline scripts."""

import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType
from typing import Protocol

from src.config.logging_config import get_logger, setup_logging
from src.config.settings import Settings

logger = get_logger(__name__)


class ShutdownSignal(Protocol):
    def is_set(self) -> bool: ...

    def wait(self, timeout: float) -> bool: ...


@dataclass
class _ShutdownController:
    """Mutable shutdown state shared across signal handlers and loops."""

    _event: threading.Event

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)

    def request(self, signum: int, frame: FrameType | None) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("shutdown_signal_received", signal=sig_name)
        self._event.set()


def create_shutdown_controller() -> _ShutdownController:
    return _ShutdownController(threading.Event())


def install_signal_handlers(controller: _ShutdownController) -> None:
    """Register SIGTERM/SIGINT handlers for graceful shutdown."""

    def _handler(signum: int, frame: FrameType | None) -> None:
        controller.request(signum, frame)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def initialize_logging(settings: Settings, *, json_logs: bool = False) -> None:
    """Initialize structlog-based logging for scripts."""

    setup_logging(log_level=settings.log_level, json_logs=json_logs)
    logger.info("logging_initialized", level=settings.log_level, json_logs=json_logs)


class WorkerProtocol(Protocol):
    def process_available_tasks(self) -> int: ...


def run_worker_loop(
    worker: WorkerProtocol,
    controller: ShutdownSignal,
    *,
    poll_interval: float,
    run_once: bool = False,
    idle_backoff_seconds: float = 1.0,
) -> None:
    """Run a worker until shutdown is requested."""

    poll_interval = max(0.1, poll_interval)
    idle_backoff_seconds = max(0.1, idle_backoff_seconds)

    logger.info(
        "worker_loop_started",
        poll_interval=poll_interval,
        run_once=run_once,
        idle_backoff=idle_backoff_seconds,
    )

    iteration = 0
    while not controller.is_set():
        iteration += 1
        try:
            processed = worker.process_available_tasks()
        except Exception:  # noqa: BLE001
            logger.exception("worker_iteration_failed", iteration=iteration)
            if run_once:
                raise
            controller.wait(idle_backoff_seconds)
            continue

        if run_once:
            break

        if processed <= 0:
            controller.wait(min(poll_interval, idle_backoff_seconds))
        else:
            time.sleep(idle_backoff_seconds)

    logger.info("worker_loop_stopped", iterations=iteration)


def run_scheduler_loop(
    *,
    controller: ShutdownSignal,
    interval_seconds: float,
    run_once: bool,
    action: Callable[[], object],
) -> None:
    """Execute a scheduler callback at a fixed interval."""

    interval_seconds = max(0.1, interval_seconds)
    logger.info("scheduler_loop_started", interval=interval_seconds, run_once=run_once)
    iteration = 0
    while not controller.is_set():
        iteration += 1
        try:
            action()
        except Exception:  # noqa: BLE001
            logger.exception("scheduler_iteration_failed", iteration=iteration)
            if run_once:
                raise
        if run_once:
            break
        controller.wait(interval_seconds)

    logger.info("scheduler_loop_stopped", iterations=iteration)


__all__ = [
    "ShutdownSignal",
    "WorkerProtocol",
    "create_shutdown_controller",
    "initialize_logging",
    "install_signal_handlers",
    "run_scheduler_loop",
    "run_worker_loop",
]
