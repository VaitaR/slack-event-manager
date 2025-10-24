"""Streamlit orchestration helpers (auth, rate limiting, job submission)."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Final

from src.adapters.job_runner_inprocess import InProcessJobRunner
from src.config.logging_config import get_logger
from src.ports.job_runner import JobRunnerPort
from src.use_cases.pipeline_orchestrator import (
    PipelineDependencies,
    PipelineParams,
    ProgressReporter,
    run_ingest_and_extract_pipeline,
)

logger = get_logger(__name__)

_JOB_NAME: Final[str] = "ingest_and_extract"
_RATE_LIMIT_MAX_REQUESTS: Final[int] = 3
_RATE_LIMIT_WINDOW_SECONDS: Final[float] = 60.0


class RateLimitExceededError(Exception):
    """Raised when a user submits jobs too quickly."""


@dataclass
class _JobState:
    runner: JobRunnerPort | None
    dependency_factory: Callable[[PipelineParams], PipelineDependencies] | None


_JOB_STATE = _JobState(runner=None, dependency_factory=None)
_RATE_LIMIT_CACHE: dict[str, deque[float]] = {}
_RATE_LIMIT_LOCK = Lock()


def configure_dependency_factory(
    factory: Callable[[PipelineParams], PipelineDependencies] | None,
) -> None:
    """Override dependency factory (useful for testing)."""

    _JOB_STATE.dependency_factory = factory


def reset_rate_limiter() -> None:
    """Clear rate limit state (testing helper)."""

    with _RATE_LIMIT_LOCK:
        _RATE_LIMIT_CACHE.clear()


def reset_job_runner() -> None:
    """Reset job runner instance (testing helper)."""

    _JOB_STATE.runner = None


def submit_ingest_extract_job(params: PipelineParams, user_id: str) -> str:
    """Submit pipeline job after enforcing rate limiting."""

    _enforce_rate_limit(user_id)
    runner = _ensure_runner()
    payload = {
        "message_limit": params.message_limit,
        "channel_ids": list(params.channel_ids),
    }
    job_id = runner.submit(_JOB_NAME, payload)
    logger.info("streamlit_job_submitted", job_id=job_id, user_id=user_id)
    return job_id


def job_status(job_id: str) -> dict[str, object]:
    """Fetch job status from the runner."""

    return _ensure_runner().status(job_id)


def job_result(job_id: str) -> dict[str, object] | None:
    """Retrieve final job result if available."""

    return _ensure_runner().result(job_id)


def _enforce_rate_limit(user_id: str) -> None:
    now = monotonic()
    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_CACHE.setdefault(user_id, deque())
        while bucket and now - bucket[0] > _RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_MAX_REQUESTS:
            logger.warning("streamlit_rate_limited", user_id=user_id)
            raise RateLimitExceededError("Too many requests. Try again shortly.")
        bucket.append(now)


def _ensure_runner() -> JobRunnerPort:
    if _JOB_STATE.runner is None:
        handlers = {
            _JOB_NAME: _create_job_handler(),
        }
        _JOB_STATE.runner = InProcessJobRunner(handlers)
    return _JOB_STATE.runner


def _create_job_handler() -> Callable[
    [dict[str, object], ProgressReporter], dict[str, object]
]:
    def _handler(
        payload: dict[str, object], reporter: ProgressReporter
    ) -> dict[str, object]:
        message_limit_raw = payload.get("message_limit", 0)
        message_limit = (
            int(message_limit_raw) if isinstance(message_limit_raw, int) else 0
        )

        channels_raw = payload.get("channel_ids", [])
        if isinstance(channels_raw, list):
            channel_ids = [str(channel) for channel in channels_raw]
        else:
            channel_ids = []

        params = PipelineParams(message_limit=message_limit, channel_ids=channel_ids)
        factory = _JOB_STATE.dependency_factory
        dependencies = factory(params) if factory else None
        result = run_ingest_and_extract_pipeline(
            params,
            reporter,
            dependencies=dependencies,
        )
        return {
            "correlation_id": result.correlation_id,
            "ingest": result.ingest,
            "candidates": result.candidates,
            "extract": result.extract,
            "dedup": result.dedup,
        }

    return _handler


__all__ = [
    "RateLimitExceededError",
    "configure_dependency_factory",
    "job_result",
    "job_status",
    "reset_job_runner",
    "reset_rate_limiter",
    "submit_ingest_extract_job",
]
