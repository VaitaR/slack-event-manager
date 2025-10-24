"""In-process job runner for development and testing."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Protocol
from uuid import uuid4

from src.config.logging_config import get_logger
from src.observability.metrics import JOB_DURATION_SECONDS, JOBS_SUBMITTED_TOTAL
from src.ports.job_runner import JobRunnerPort

logger = get_logger(__name__)


class JobReporter(Protocol):
    def update(
        self, *, progress: float | None = None, message: str | None = None
    ) -> None: ...


JobHandler = Callable[[dict[str, object], JobReporter], dict[str, object]]

_STATUS_QUEUED: Final[str] = "queued"
_STATUS_RUNNING: Final[str] = "running"
_STATUS_SUCCEEDED: Final[str] = "succeeded"
_STATUS_FAILED: Final[str] = "failed"


@dataclass
class JobProgressReporter:
    """Mutable view exposed to job handlers for progress updates."""

    job_id: str
    _runner: InProcessJobRunner

    def update(
        self, *, progress: float | None = None, message: str | None = None
    ) -> None:
        """Update live job information for UI consumers."""

        self._runner._update_job(self.job_id, progress=progress, message=message)


@dataclass
class JobRecord:
    """Internal representation of a submitted job."""

    name: str
    params: dict[str, object]
    status: str = field(default=_STATUS_QUEUED)
    progress: float = field(default=0.0)
    message: str | None = field(default=None)
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = field(default=None)
    finished_at: float | None = field(default=None)
    result: dict[str, object] | None = field(default=None)
    error: str | None = field(default=None)


class InProcessJobRunner(JobRunnerPort):
    """Simple job runner executing tasks on worker threads."""

    def __init__(self, handlers: dict[str, JobHandler]):
        if not handlers:
            raise ValueError("handlers must not be empty")
        self._handlers = handlers
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()

    def submit(self, name: str, params: dict[str, object]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            raise KeyError(f"Unknown job name: {name}")

        job_id = str(uuid4())
        record = JobRecord(name=name, params=dict(params))
        with self._lock:
            self._jobs[job_id] = record

        logger.info("job_submitted", job_id=job_id, job_name=name)
        JOBS_SUBMITTED_TOTAL.labels(job=name).inc()

        thread = threading.Thread(
            target=self._execute_job, args=(job_id, handler), daemon=True
        )
        thread.start()
        return job_id

    def status(self, job_id: str) -> dict[str, object]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown job_id: {job_id}")
            return {
                "job_id": job_id,
                "name": record.name,
                "status": record.status,
                "progress": record.progress,
                "message": record.message,
                "submitted_at": record.submitted_at,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "error": record.error,
            }

    def result(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown job_id: {job_id}")
            return record.result

    # Internal helpers -------------------------------------------------

    def _execute_job(self, job_id: str, handler: JobHandler) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = _STATUS_RUNNING
            record.started_at = time.time()
            params = dict(record.params)

        reporter = JobProgressReporter(job_id=job_id, _runner=self)
        start_time = time.perf_counter()
        try:
            result = handler(params, reporter)
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - start_time
            JOB_DURATION_SECONDS.labels(job=self._jobs[job_id].name).observe(duration)
            logger.exception("job_failed", job_id=job_id)
            with self._lock:
                record = self._jobs[job_id]
                record.status = _STATUS_FAILED
                record.finished_at = time.time()
                record.error = str(exc)
                record.message = "Job failed"
        else:
            duration = time.perf_counter() - start_time
            JOB_DURATION_SECONDS.labels(job=self._jobs[job_id].name).observe(duration)
            logger.info(
                "job_completed",
                job_id=job_id,
                job_name=self._jobs[job_id].name,
                duration_seconds=duration,
            )
            with self._lock:
                record = self._jobs[job_id]
                record.status = _STATUS_SUCCEEDED
                record.finished_at = time.time()
                record.result = result
                record.progress = 1.0
                record.message = "Completed"

    def _update_job(
        self, job_id: str, *, progress: float | None, message: str | None
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            if progress is not None:
                record.progress = max(0.0, min(progress, 1.0))
            if message is not None:
                record.message = message


__all__ = ["InProcessJobRunner", "JobProgressReporter"]
