"""Helpers for resolving task queue adapters from repositories."""

from __future__ import annotations

from collections.abc import Callable

from src.config.logging_config import get_logger
from src.domain.protocols import RepositoryProtocol
from src.ports.task_queue import TaskQueuePort

logger = get_logger(__name__)


class TaskQueueUnavailableError(RuntimeError):
    """Raised when the repository does not expose a task queue."""


def resolve_task_queue(repository: RepositoryProtocol) -> TaskQueuePort:
    """Obtain the task queue adapter from the configured repository."""

    provider_raw = getattr(repository, "task_queue", None)
    provider: Callable[[], TaskQueuePort] | None
    if callable(provider_raw):
        provider = provider_raw
    else:
        provider = None

    if provider is None:
        msg = "Repository does not expose a task_queue method"
        raise TaskQueueUnavailableError(msg)

    try:
        queue = provider()
    except NotImplementedError as exc:  # pragma: no cover - defensive branch
        logger.error("task_queue_not_supported", repository=type(repository).__name__)
        raise TaskQueueUnavailableError(
            "Task queue is not supported by this repository"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "task_queue_resolution_failed", repository=type(repository).__name__
        )
        raise TaskQueueUnavailableError("Failed to resolve task queue") from exc

    required_methods = ("enqueue", "enqueue_many", "lease", "complete", "fail")
    if not all(hasattr(queue, name) for name in required_methods):
        msg = "Resolved task queue does not provide required task queue methods"
        raise TaskQueueUnavailableError(msg)

    return queue


__all__ = ["TaskQueueUnavailableError", "resolve_task_queue"]
