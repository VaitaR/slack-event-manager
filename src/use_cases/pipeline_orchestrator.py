"""Pipeline orchestration utilities for the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol
from uuid import uuid4

from src.adapters.llm_client import LLMClient
from src.adapters.repository_factory import create_repository
from src.adapters.slack_client import SlackClient
from src.config.logging_config import get_logger
from src.config.settings import Settings, get_settings
from src.domain.models import (
    CandidateResult,
    DeduplicationResult,
    ExtractionResult,
    IngestResult,
    MessageSource,
)
from src.domain.protocols import RepositoryProtocol
from src.observability.tracing import correlation_scope
from src.use_cases.build_candidates import build_candidates_use_case
from src.use_cases.deduplicate_events import deduplicate_events_use_case
from src.use_cases.extract_events import extract_events_use_case
from src.use_cases.ingest_messages import ingest_messages_use_case

logger = get_logger(__name__)
_PIPELINE_JOB_NAME: Final[str] = "ingest_and_extract"


class ProgressReporter(Protocol):
    """Interface for publishing job progress updates."""

    def update(
        self, *, progress: float | None = None, message: str | None = None
    ) -> None: ...


@dataclass
class PipelineParams:
    """Parameters provided by the UI when launching a job."""

    message_limit: int
    channel_ids: list[str]


@dataclass
class PipelineResult:
    """Summary of pipeline execution for presentation."""

    correlation_id: str
    ingest: dict[str, object]
    candidates: dict[str, object]
    extract: dict[str, object]
    dedup: dict[str, object]


@dataclass
class PipelineDependencies:
    """Runtime dependencies required for executing the pipeline."""

    settings: Settings
    repository: RepositoryProtocol
    slack_client: SlackClient
    llm_client: LLMClient

    @classmethod
    def from_environment(cls, params: PipelineParams) -> PipelineDependencies:
        settings = get_settings()

        target_channels = set(params.channel_ids)
        selected_channels = [
            channel
            for channel in settings.slack_channels
            if channel.channel_id in target_channels
        ]
        effective_channels = selected_channels or settings.slack_channels

        effective_settings = settings.model_copy(
            update={
                "slack_channels": effective_channels,
                "slack_max_messages_per_run": params.message_limit,
            }
        )

        slack_client = SlackClient(
            bot_token=effective_settings.slack_bot_token.get_secret_value()
        )
        llm_client = LLMClient(
            api_key=effective_settings.openai_api_key.get_secret_value(),
            model=effective_settings.llm_model,
            temperature=(
                1.0
                if effective_settings.llm_model == "gpt-5-nano"
                else effective_settings.llm_temperature
            ),
            timeout=effective_settings.llm_timeout_seconds,
            verbose=False,
        )
        repository = create_repository(effective_settings)

        return cls(
            settings=effective_settings,
            repository=repository,
            slack_client=slack_client,
            llm_client=llm_client,
        )


def run_ingest_and_extract_pipeline(
    params: PipelineParams,
    reporter: ProgressReporter,
    *,
    dependencies: PipelineDependencies | None = None,
) -> PipelineResult:
    """Execute the ingestion/extraction pipeline asynchronously."""

    if params.message_limit <= 0:
        raise ValueError("message_limit must be positive")

    deps = dependencies or PipelineDependencies.from_environment(params)
    correlation_id = str(uuid4())
    ingest_result: IngestResult
    candidate_result: CandidateResult
    extraction_result: ExtractionResult
    dedup_result: DeduplicationResult

    with correlation_scope(correlation_id):
        logger.info(
            "pipeline_job_started",
            job_name=_PIPELINE_JOB_NAME,
            message_limit=params.message_limit,
            channel_count=len(params.channel_ids),
        )

        reporter.update(progress=0.05, message="Starting ingestion")
        ingest_result = ingest_messages_use_case(
            slack_client=deps.slack_client,
            repository=deps.repository,
            settings=deps.settings,
            correlation_id=correlation_id,
        )

        reporter.update(progress=0.35, message="Scoring candidates")
        candidate_result = build_candidates_use_case(
            repository=deps.repository,
            settings=deps.settings,
        )

        reporter.update(progress=0.6, message="Extracting events")
        extraction_result = extract_events_use_case(
            llm_client=deps.llm_client,
            repository=deps.repository,
            settings=deps.settings,
            source_id=MessageSource.SLACK,
            batch_size=None,
            check_budget=False,
            correlation_id=correlation_id,
        )

        reporter.update(progress=0.85, message="Deduplicating events")
        dedup_result = deduplicate_events_use_case(
            repository=deps.repository,
            settings=deps.settings,
            source_id=MessageSource.SLACK,
            correlation_id=correlation_id,
        )

        reporter.update(progress=1.0, message="Pipeline complete")

        logger.info(
            "pipeline_job_finished",
            job_name=_PIPELINE_JOB_NAME,
            correlation_id=correlation_id,
            messages_saved=ingest_result.messages_saved,
            candidates_created=candidate_result.candidates_created,
            events_extracted=extraction_result.events_extracted,
            final_events=dedup_result.total_events,
        )

    return PipelineResult(
        correlation_id=correlation_id,
        ingest=ingest_result.model_dump(),
        candidates=candidate_result.model_dump(),
        extract=extraction_result.model_dump(),
        dedup=dedup_result.model_dump(),
    )


__all__ = [
    "PipelineDependencies",
    "PipelineParams",
    "PipelineResult",
    "ProgressReporter",
    "run_ingest_and_extract_pipeline",
]
