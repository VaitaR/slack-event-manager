"""Port definition for asynchronous job execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class JobRunnerPort(Protocol):
    """Interface for submitting and tracking background jobs."""

    def submit(self, name: str, params: dict[str, object]) -> str:
        """Schedule a job for asynchronous execution.

        Args:
            name: Logical job name.
            params: Arbitrary job parameters.

        Returns:
            Unique identifier for the submitted job.
        """

    def status(self, job_id: str) -> dict[str, object]:
        """Retrieve current status for a submitted job."""

    def result(self, job_id: str) -> dict[str, object] | None:
        """Return the terminal result for a job if finished."""


__all__ = ["JobRunnerPort"]
