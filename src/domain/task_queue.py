"""Domain models and helpers for task queue operations."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

DEFAULT_PRIORITY_NORMAL: Final[int] = 50
DEFAULT_MAX_ATTEMPTS: Final[int] = 5


class TaskType(StrEnum):
    """Logical task categories supported by the pipeline."""

    INGEST = "ingest"
    EXTRACTION = "extraction"
    LLM_EXTRACTION = "llm_extraction"
    DEDUP = "dedup"
    DIGEST = "digest"


class TaskStatus(StrEnum):
    """Processing lifecycle states for queued tasks."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class TaskCreate(BaseModel):
    """Schema used when enqueuing a new task."""

    task_type: TaskType
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = DEFAULT_PRIORITY_NORMAL
    run_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    idempotency_key: str
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    @field_validator("run_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, value: int) -> int:
        if value < 0:
            msg = "priority must be non-negative"
            raise ValueError(msg)
        return value

    @field_validator("max_attempts")
    @classmethod
    def _validate_attempts(cls, value: int) -> int:
        if value <= 0:
            msg = "max_attempts must be positive"
            raise ValueError(msg)
        return value


class Task(BaseModel):
    """Persisted task representation."""

    task_id: UUID = Field(default_factory=uuid4)
    task_type: TaskType
    payload: dict[str, Any]
    priority: int
    run_at: datetime
    status: TaskStatus
    attempts: int
    max_attempts: int
    idempotency_key: str
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime
    locked_at: datetime | None = None

    @field_validator("run_at", "created_at", "updated_at", "locked_at")
    @classmethod
    def _ensure_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_PRIORITY_NORMAL",
    "Task",
    "TaskCreate",
    "TaskStatus",
    "TaskType",
]
