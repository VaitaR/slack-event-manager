"""Pipeline workers backed by the task queue."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol

from src.config.logging_config import get_logger
from src.domain.models import (
    CandidateResult,
    DeduplicationResult,
    DigestResult,
    IngestResult,
)
from src.domain.task_queue import Task, TaskCreate, TaskType
from src.ports.task_queue import TaskQueuePort
from src.use_cases.extract_events import (
    CandidateExtractionMetrics,
    LLMTaskScheduleResult,
)
from src.use_cases.pipeline_priorities import (
    DEDUP_TASK_PRIORITY,
    DIGEST_TASK_PRIORITY,
    EXTRACTION_TASK_PRIORITY,
)

logger = get_logger(__name__)


class LLMScheduler(Protocol):
    """Protocol for scheduling LLM extraction tasks."""

    def __call__(
        self, *, correlation_id: str | None, batch_hint: int | None
    ) -> LLMTaskScheduleResult: ...


class LLMCandidateProcessor(Protocol):
    """Protocol for processing a single LLM candidate."""

    def __call__(
        self, *, message_id: str, correlation_id: str | None
    ) -> CandidateExtractionMetrics: ...


_DEFAULT_RETRY_MAX_SECONDS: Final[float] = 300.0
_DEFAULT_BATCH_SIZE: Final[int] = 8


class _BaseWorker:
    """Common worker functionality (leasing, retries, logging)."""

    def __init__(
        self,
        *,
        task_queue: TaskQueuePort,
        task_type: TaskType,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        jitter_provider: Callable[[float], float] | None = None,
    ) -> None:
        if batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)

        self._task_queue = task_queue
        self._task_type = task_type
        self._batch_size = batch_size
        self._jitter_provider = jitter_provider or _default_jitter

    def process_available_tasks(self) -> int:
        """Lease and process pending tasks."""

        tasks = self._task_queue.lease(self._task_type, self._batch_size)
        for task in tasks:
            try:
                logger.info(
                    "worker_task_started",
                    task_type=self._task_type.value,
                    task_id=str(task.task_id),
                    attempts=task.attempts,
                )
                self._handle_task(task)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "worker_task_failed",
                    task_type=self._task_type.value,
                    task_id=str(task.task_id),
                )
                retry_at = self._compute_retry_at(task)
                self._task_queue.fail(
                    task.task_id,
                    error=f"{type(exc).__name__}: {exc}",
                    retry_at=retry_at,
                )
            else:
                self._task_queue.complete(task.task_id)
                logger.info(
                    "worker_task_completed",
                    task_type=self._task_type.value,
                    task_id=str(task.task_id),
                )
        return len(tasks)

    def _compute_retry_at(self, task: Task) -> datetime | None:
        if task.attempts >= task.max_attempts:
            return None

        base_delay = min(
            _DEFAULT_RETRY_MAX_SECONDS, math.pow(2.0, max(task.attempts - 1, 0))
        )
        jitter = max(0.0, self._jitter_provider(base_delay))
        delay = max(1.0, base_delay + jitter)
        return datetime.now(tz=UTC) + timedelta(seconds=delay)

    def _handle_task(self, task: Task) -> None:
        raise NotImplementedError


def _default_jitter(base: float) -> float:
    return random.uniform(0.0, base * 0.25)


class IngestWorker(_BaseWorker):
    """Worker responsible for message ingestion and candidate creation."""

    def __init__(
        self,
        *,
        task_queue: TaskQueuePort,
        ingest_messages: Callable[..., IngestResult],
        build_candidates: Callable[[], CandidateResult],
        jitter_provider: Callable[[float], float] | None = None,
    ) -> None:
        super().__init__(
            task_queue=task_queue,
            task_type=TaskType.INGEST,
            batch_size=1,
            jitter_provider=jitter_provider,
        )
        self._ingest_messages = ingest_messages
        self._build_candidates = build_candidates

    def _handle_task(self, task: Task) -> None:
        ingest_result = self._ingest_messages(correlation_id=str(task.task_id))
        candidate_result = self._build_candidates()

        logger.info(
            "ingest_worker_summary",
            fetched=ingest_result.messages_fetched,
            saved=ingest_result.messages_saved,
            candidates=candidate_result.candidates_created,
        )

        if candidate_result.candidates_created <= 0:
            return

        enqueue = TaskCreate(
            task_type=TaskType.EXTRACTION,
            payload={
                "source": "ingest",
                "correlation_id": str(task.task_id),
                "candidates_created": candidate_result.candidates_created,
            },
            priority=EXTRACTION_TASK_PRIORITY,
            idempotency_key=f"extraction:{task.task_id}",
        )
        self._task_queue.enqueue(enqueue)


class ExtractionWorker(_BaseWorker):
    """Worker that schedules LLM extraction jobs for processing."""

    def __init__(
        self,
        *,
        task_queue: TaskQueuePort,
        schedule_llm_tasks: LLMScheduler,
        jitter_provider: Callable[[float], float] | None = None,
    ) -> None:
        super().__init__(
            task_queue=task_queue,
            task_type=TaskType.EXTRACTION,
            batch_size=1,
            jitter_provider=jitter_provider,
        )
        self._schedule_llm_tasks = schedule_llm_tasks

    def _handle_task(self, task: Task) -> None:
        payload = task.payload or {}
        correlation_id = payload.get("correlation_id")
        candidates_hint = payload.get("candidates_created")
        batch_hint = candidates_hint if isinstance(candidates_hint, int) else None

        result = self._schedule_llm_tasks(
            correlation_id=correlation_id,
            batch_hint=batch_hint,
        )

        logger.info(
            "extraction_worker_summary",
            correlation_id=correlation_id,
            scheduled=result.candidates_enqueued,
            total_candidates=result.total_candidates,
            batch_hint=batch_hint,
        )


class LLMExtractionWorker(_BaseWorker):
    """Worker that executes LLM extraction for individual candidates."""

    def __init__(
        self,
        *,
        task_queue: TaskQueuePort,
        process_candidate: LLMCandidateProcessor,
        jitter_provider: Callable[[float], float] | None = None,
    ) -> None:
        super().__init__(
            task_queue=task_queue,
            task_type=TaskType.LLM_EXTRACTION,
            batch_size=1,
            jitter_provider=jitter_provider,
        )
        self._process_candidate = process_candidate

    def _handle_task(self, task: Task) -> None:
        payload = task.payload or {}
        message_id = payload.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            msg = "llm task missing message_id"
            raise ValueError(msg)

        correlation_id = payload.get("correlation_id")
        metrics = self._process_candidate(
            message_id=message_id,
            correlation_id=correlation_id,
        )

        logger.info(
            "llm_worker_summary",
            correlation_id=correlation_id,
            message_id=message_id[:8],
            events=metrics.events_extracted,
            llm_calls=metrics.llm_calls,
            cache_hits=metrics.cache_hits,
            dedup_required=metrics.dedup_required,
            budget_exhausted=metrics.budget_exhausted,
            errors=len(metrics.errors),
        )

        if not metrics.dedup_required:
            return

        dedup_task = TaskCreate(
            task_type=TaskType.DEDUP,
            payload={
                "source": "llm",
                "correlation_id": correlation_id,
                "origin_message_id": message_id,
            },
            priority=DEDUP_TASK_PRIORITY,
            idempotency_key=f"dedup:{message_id}",
        )
        self._task_queue.enqueue(dedup_task)


class DedupWorker(_BaseWorker):
    """Worker responsible for deduplicating newly extracted events."""

    def __init__(
        self,
        *,
        task_queue: TaskQueuePort,
        deduplicate_events: Callable[[], DeduplicationResult],
        jitter_provider: Callable[[float], float] | None = None,
    ) -> None:
        super().__init__(
            task_queue=task_queue,
            task_type=TaskType.DEDUP,
            batch_size=1,
            jitter_provider=jitter_provider,
        )
        self._deduplicate_events = deduplicate_events

    def _handle_task(self, task: Task) -> None:
        result = self._deduplicate_events()

        logger.info(
            "dedup_worker_summary",
            new_events=result.new_events,
            merged=result.merged_events,
            total=result.total_events,
        )

        if result.new_events <= 0:
            return

        payload = task.payload or {}
        enqueue = TaskCreate(
            task_type=TaskType.DIGEST,
            payload={
                "source": "dedup",
                "correlation_id": payload.get("correlation_id"),
                "new_events": result.new_events,
            },
            priority=DIGEST_TASK_PRIORITY,
            idempotency_key=f"digest:{task.task_id}",
        )
        self._task_queue.enqueue(enqueue)


class DigestWorker(_BaseWorker):
    """Worker publishing Slack digests."""

    def __init__(
        self,
        *,
        task_queue: TaskQueuePort,
        publish_digest: Callable[[], DigestResult],
        jitter_provider: Callable[[float], float] | None = None,
    ) -> None:
        super().__init__(
            task_queue=task_queue,
            task_type=TaskType.DIGEST,
            batch_size=1,
            jitter_provider=jitter_provider,
        )
        self._publish_digest = publish_digest

    def _handle_task(self, task: Task) -> None:  # noqa: ARG002
        result = self._publish_digest()
        logger.info(
            "digest_worker_summary",
            messages=result.messages_posted,
            events=result.events_included,
        )


__all__ = [
    "IngestWorker",
    "ExtractionWorker",
    "LLMExtractionWorker",
    "DedupWorker",
    "DigestWorker",
]
