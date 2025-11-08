"""Utilities for scheduling pipeline tasks via the task queue."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from src.config.logging_config import get_logger
from src.domain.task_queue import TaskCreate, TaskType
from src.ports.task_queue import TaskQueuePort
from src.use_cases.pipeline_priorities import (
    DEDUP_TASK_PRIORITY,
    EXTRACTION_TASK_PRIORITY,
    INGEST_TASK_PRIORITY,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineScheduleResult:
    """Summary of tasks produced by a scheduler iteration."""

    ingest_enqueued: int
    extraction_enqueued: int
    dedup_enqueued: int
    correlation_id: str


def enqueue_pipeline_iteration(
    task_queue: TaskQueuePort,
    *,
    correlation_id: str | None = None,
    include_dedup: bool = True,
) -> PipelineScheduleResult:
    """Enqueue core pipeline tasks for a scheduler iteration."""

    iteration_id = correlation_id or str(uuid4())
    scheduled_at = datetime.now(tz=UTC).isoformat()

    tasks = [
        TaskCreate(
            task_type=TaskType.INGEST,
            payload={
                "source": "scheduler",
                "scheduled_at": scheduled_at,
                "correlation_id": iteration_id,
            },
            priority=INGEST_TASK_PRIORITY,
            idempotency_key=f"ingest:{iteration_id}:{scheduled_at}",
        ),
        TaskCreate(
            task_type=TaskType.EXTRACTION,
            payload={
                "source": "scheduler",
                "scheduled_at": scheduled_at,
                "correlation_id": iteration_id,
            },
            priority=EXTRACTION_TASK_PRIORITY,
            idempotency_key=f"extraction:{iteration_id}:{scheduled_at}",
        ),
    ]

    if include_dedup:
        tasks.append(
            TaskCreate(
                task_type=TaskType.DEDUP,
                payload={
                    "source": "scheduler",
                    "scheduled_at": scheduled_at,
                    "correlation_id": iteration_id,
                },
                priority=DEDUP_TASK_PRIORITY,
                idempotency_key=f"dedup:{iteration_id}:{scheduled_at}",
            )
        )

    results = task_queue.enqueue_many(tasks)

    ingest_count = sum(1 for task in results if task.task_type is TaskType.INGEST)
    extraction_count = sum(
        1 for task in results if task.task_type is TaskType.EXTRACTION
    )
    dedup_count = sum(1 for task in results if task.task_type is TaskType.DEDUP)

    logger.info(
        "pipeline_iteration_enqueued",
        correlation_id=iteration_id,
        ingest=ingest_count,
        extraction=extraction_count,
        dedup=dedup_count,
    )

    return PipelineScheduleResult(
        ingest_enqueued=ingest_count,
        extraction_enqueued=extraction_count,
        dedup_enqueued=dedup_count,
        correlation_id=iteration_id,
    )


__all__ = ["PipelineScheduleResult", "enqueue_pipeline_iteration"]
