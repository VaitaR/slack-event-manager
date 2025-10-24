from __future__ import annotations

import time

import pytest

from src.presentation import streamlit_orchestration as orchestration
from src.use_cases.pipeline_orchestrator import (
    PipelineParams,
    PipelineResult,
    ProgressReporter,
)


@pytest.fixture(autouse=True)
def reset_state() -> None:
    orchestration.reset_job_runner()
    orchestration.reset_rate_limiter()


def test_submit_and_poll_job(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = PipelineResult(
        correlation_id="test-corr",
        ingest={"messages_saved": 1},
        candidates={"candidates_created": 0},
        extract={"events_extracted": 0},
        dedup={"total_events": 0},
    )

    def fake_run(
        params: PipelineParams,
        reporter: ProgressReporter,
        *,
        dependencies: object | None = None,
    ) -> PipelineResult:
        reporter.update(progress=1.0, message="done")
        return expected

    monkeypatch.setattr(
        orchestration, "run_ingest_and_extract_pipeline", fake_run, raising=True
    )

    params = PipelineParams(message_limit=5, channel_ids=["C123"])
    job_id = orchestration.submit_ingest_extract_job(params, user_id="user-1")
    assert isinstance(job_id, str)

    for _ in range(20):
        status = orchestration.job_status(job_id)
        if status["status"] == "succeeded":
            break
        time.sleep(0.05)

    assert status["status"] == "succeeded"
    result = orchestration.job_result(job_id)
    assert result == expected.__dict__
