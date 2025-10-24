from __future__ import annotations

import time

from src.adapters.job_runner_inprocess import InProcessJobRunner, JobProgressReporter


def test_inprocess_job_runner_executes_and_tracks_status() -> None:
    runner = InProcessJobRunner({"double": _double_handler})
    job_id = runner.submit("double", {"value": 21})

    status = runner.status(job_id)
    assert status["status"] in {"queued", "running"}

    for _ in range(20):
        status = runner.status(job_id)
        if status["status"] == "succeeded":
            break
        time.sleep(0.05)

    assert status["status"] == "succeeded"
    assert float(status["progress"]) >= 1.0
    result = runner.result(job_id)
    assert result is not None
    assert result["value"] == 42


def _double_handler(
    params: dict[str, object], reporter: JobProgressReporter
) -> dict[str, object]:
    reporter.update(progress=0.5, message="halfway")
    time.sleep(0.05)
    reporter.update(progress=1.0, message="done")
    value = int(params["value"])
    return {"value": value * 2}
