from __future__ import annotations

from typing import Any

import pytest
from structlog.testing import capture_logs

from src.config.logging_config import get_logger
from src.domain.models import (
    CandidateResult,
    DeduplicationResult,
    ExtractionResult,
    IngestResult,
)
from src.observability.metrics import PIPELINE_STAGE_DURATION_SECONDS
from src.observability.tracing import correlation_scope
from src.use_cases.pipeline_orchestrator import (
    PipelineParams,
    PipelineResult,
    run_ingest_and_extract_pipeline,
)


def test_pipeline_emits_correlation_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts_before = _histogram_counts()

    def fake_ingest(**kwargs: Any) -> IngestResult:
        with correlation_scope(kwargs.get("correlation_id")):
            PIPELINE_STAGE_DURATION_SECONDS.labels(stage="ingest").observe(0.001)
            LOGGER.info("ingest_stage")
            return IngestResult(
                messages_fetched=0,
                messages_saved=0,
                channels_processed=[],
                errors=[],
            )

    def fake_build(**kwargs: Any) -> CandidateResult:
        return CandidateResult(
            candidates_created=0,
            messages_processed=0,
            average_score=0.0,
        )

    def fake_extract(**kwargs: Any) -> ExtractionResult:
        with correlation_scope(kwargs.get("correlation_id")):
            PIPELINE_STAGE_DURATION_SECONDS.labels(stage="extract").observe(0.001)
            LOGGER.info("extract_stage")
            return ExtractionResult(
                events_extracted=0,
                candidates_processed=0,
                llm_calls=0,
                cache_hits=0,
                total_cost_usd=0.0,
                errors=[],
            )

    def fake_dedup(**kwargs: Any) -> DeduplicationResult:
        with correlation_scope(kwargs.get("correlation_id")):
            PIPELINE_STAGE_DURATION_SECONDS.labels(stage="dedup").observe(0.001)
            LOGGER.info("dedup_stage")
            return DeduplicationResult(
                new_events=0,
                merged_events=0,
                total_events=0,
            )

    monkeypatch.setattr(
        "src.use_cases.pipeline_orchestrator.ingest_messages_use_case",
        fake_ingest,
    )
    monkeypatch.setattr(
        "src.use_cases.pipeline_orchestrator.build_candidates_use_case",
        fake_build,
    )
    monkeypatch.setattr(
        "src.use_cases.pipeline_orchestrator.extract_events_use_case",
        fake_extract,
    )
    monkeypatch.setattr(
        "src.use_cases.pipeline_orchestrator.deduplicate_events_use_case",
        fake_dedup,
    )

    reporter = _Reporter()
    params = PipelineParams(message_limit=5, channel_ids=[])

    with capture_logs() as logs:
        result = run_ingest_and_extract_pipeline(params, reporter)

    assert isinstance(result, PipelineResult)
    assert any(log.get("correlation_id") for log in logs)

    counts_after = _histogram_counts()
    for stage in ("ingest", "extract", "dedup"):
        assert counts_after[stage] == counts_before[stage] + 1


def _histogram_counts() -> dict[str, float]:
    counts: dict[str, float] = {"ingest": 0.0, "extract": 0.0, "dedup": 0.0}
    for metric in PIPELINE_STAGE_DURATION_SECONDS.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count"):
                counts[sample.labels["stage"]] = sample.value
    return counts


class _Reporter:
    def update(
        self, *, progress: float | None = None, message: str | None = None
    ) -> None:
        return None


LOGGER = get_logger(__name__)
