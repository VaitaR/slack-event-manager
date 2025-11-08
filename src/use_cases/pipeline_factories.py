"""Factories to compose pipeline worker callables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Protocol, cast

from src.adapters.llm_client import LLMClient
from src.adapters.slack_client import SlackClient as SlackAdapterClient
from src.clients.slack_wrapped import SlackClient as WrappedSlackClient
from src.config.settings import Settings
from src.domain.models import (
    CandidateResult,
    DeduplicationResult,
    DigestResult,
    IngestResult,
    MessageSource,
)
from src.domain.protocols import RepositoryProtocol, SlackClientProtocol
from src.ports.task_queue import TaskQueuePort
from src.services.importance_scorer import ImportanceScorer
from src.services.object_registry import ObjectRegistry
from src.use_cases.build_candidates import build_candidates_use_case
from src.use_cases.deduplicate_events import deduplicate_events_use_case
from src.use_cases.extract_events import (
    CandidateExtractionMetrics,
    LLMTaskScheduleResult,
    build_object_registry,
    process_llm_candidate_task_use_case,
    schedule_llm_extraction_tasks_use_case,
)
from src.use_cases.ingest_messages import ingest_messages_use_case
from src.use_cases.pipeline_priorities import EXTRACTION_TASK_PRIORITY
from src.use_cases.publish_digest import publish_digest_use_case


class LLMScheduler(Protocol):
    def __call__(
        self, *, correlation_id: str | None, batch_hint: int | None
    ) -> LLMTaskScheduleResult: ...


class LLMCandidateProcessor(Protocol):
    def __call__(
        self, *, message_id: str, correlation_id: str | None
    ) -> CandidateExtractionMetrics: ...


class IngestCallable(Protocol):
    def __call__(self, *, correlation_id: str | None) -> IngestResult: ...


class CandidateBuilder(Protocol):
    def __call__(self) -> CandidateResult: ...


@dataclass(frozen=True, slots=True)
class LLMWorkerComponents:
    llm_client: LLMClient
    object_registry: ObjectRegistry
    importance_scorer: ImportanceScorer


def create_slack_ingestion_handlers(
    *,
    settings: Settings,
    repository: RepositoryProtocol,
    slack_client: SlackClientProtocol,
    lookback_hours: int | None = None,
    backfill_from_date: datetime | None = None,
) -> tuple[IngestCallable, CandidateBuilder]:
    """Build callables used by :class:`~src.workers.pipeline.IngestWorker`."""

    def ingest_messages(*, correlation_id: str | None) -> IngestResult:
        client = cast(WrappedSlackClient, slack_client)
        return ingest_messages_use_case(
            slack_client=client,
            repository=repository,
            settings=settings,
            lookback_hours=lookback_hours,
            backfill_from_date=backfill_from_date,
            correlation_id=correlation_id,
        )

    def build_candidates() -> CandidateResult:
        return build_candidates_use_case(repository=repository, settings=settings)

    return ingest_messages, build_candidates


def create_llm_scheduler(
    *,
    settings: Settings,
    repository: RepositoryProtocol,
    task_queue: TaskQueuePort,
    default_batch_size: int | None = 50,
    source_id: MessageSource | None = MessageSource.SLACK,
) -> LLMScheduler:
    """Build scheduler callable for :class:`ExtractionWorker`."""

    def schedule(
        *, correlation_id: str | None, batch_hint: int | None
    ) -> LLMTaskScheduleResult:
        return schedule_llm_extraction_tasks_use_case(
            repository=repository,
            task_queue=task_queue,
            settings=settings,
            priority=EXTRACTION_TASK_PRIORITY,
            batch_size=default_batch_size,
            batch_hint=batch_hint,
            source_id=source_id,
            correlation_id=correlation_id,
        )

    return schedule


def create_llm_worker_components(
    settings: Settings, llm_client: LLMClient
) -> LLMWorkerComponents:
    """Construct reusable components required for LLM extraction workers."""

    object_registry = build_object_registry(settings)
    importance_scorer = ImportanceScorer()
    return LLMWorkerComponents(
        llm_client=llm_client,
        object_registry=object_registry,
        importance_scorer=importance_scorer,
    )


def create_llm_candidate_processor(
    *,
    components: LLMWorkerComponents,
    repository: RepositoryProtocol,
    settings: Settings,
) -> LLMCandidateProcessor:
    """Build callable for :class:`LLMExtractionWorker`."""

    def process(
        *, message_id: str, correlation_id: str | None
    ) -> CandidateExtractionMetrics:
        return process_llm_candidate_task_use_case(
            message_id=message_id,
            llm_client=components.llm_client,
            repository=repository,
            settings=settings,
            object_registry=components.object_registry,
            importance_scorer=components.importance_scorer,
            event_validator=None,
            correlation_id=correlation_id,
        )

    return process


def create_deduplication_handler(
    *, settings: Settings, repository: RepositoryProtocol
) -> Callable[[], DeduplicationResult]:
    """Build callable used by :class:`DedupWorker`."""

    return partial(
        deduplicate_events_use_case, repository=repository, settings=settings
    )


def create_digest_handler(
    *,
    settings: Settings,
    repository: RepositoryProtocol,
    slack_client: SlackAdapterClient,
    dry_run: bool = False,
) -> Callable[[], DigestResult]:
    """Build callable used by :class:`DigestWorker`."""

    def publish() -> DigestResult:
        return publish_digest_use_case(
            slack_client=slack_client,
            repository=repository,
            settings=settings,
            lookback_hours=settings.digest_lookback_hours,
            dry_run=dry_run,
        )

    return publish


__all__ = [
    "CandidateBuilder",
    "LLMScheduler",
    "LLMCandidateProcessor",
    "LLMWorkerComponents",
    "IngestCallable",
    "create_deduplication_handler",
    "create_digest_handler",
    "create_llm_candidate_processor",
    "create_llm_scheduler",
    "create_llm_worker_components",
    "create_slack_ingestion_handlers",
]
