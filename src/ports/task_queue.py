"""Port definition for task queue backends."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from src.domain.task_queue import Task, TaskCreate, TaskType


@runtime_checkable
class TaskQueuePort(Protocol):
    """Abstract interface implemented by task queue adapters."""

    def enqueue(self, task: TaskCreate) -> Task:
        """Enqueue a single task and return its persisted representation."""

    def enqueue_many(self, tasks: list[TaskCreate]) -> list[Task]:
        """Enqueue multiple tasks atomically."""

    def lease(self, task_type: TaskType, limit: int) -> list[Task]:
        """Lease up to ``limit`` tasks for processing."""

    def complete(self, task_id: UUID) -> None:
        """Mark task as completed successfully."""

    def fail(self, task_id: UUID, *, error: str, retry_at: datetime | None) -> None:
        """Record task failure and optionally schedule a retry."""


__all__ = ["TaskQueuePort"]
