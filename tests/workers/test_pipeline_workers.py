from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pytest_mock import MockerFixture

from src.domain.models import (
    CandidateResult,
    DeduplicationResult,
    DigestResult,
    ExtractionResult,
    IngestResult,
)
from src.domain.task_queue import Task, TaskStatus, TaskType
from src.workers.pipeline import DigestWorker, ExtractionWorker, IngestWorker


def _task(task_type: TaskType, attempts: int = 1) -> Task:
    now = datetime.now(tz=UTC)
    return Task(
        task_id=uuid4(),
        task_type=task_type,
        payload={},
        priority=10,
        run_at=now,
        status=TaskStatus.IN_PROGRESS,
        attempts=attempts,
        max_attempts=5,
        idempotency_key=f"{task_type.value}:{uuid4()}",
        last_error=None,
        created_at=now,
        updated_at=now,
        locked_at=now,
    )


def test_ingest_worker_processes_and_enqueues_extraction(mocker: MockerFixture) -> None:
    queue = mocker.Mock()
    task = _task(TaskType.INGEST)
    queue.lease.return_value = [task]

    ingest_result = IngestResult(
        messages_fetched=4,
        messages_saved=4,
        channels_processed=["C123"],
        errors=[],
    )
    candidate_result = CandidateResult(
        candidates_created=2,
        messages_processed=4,
        average_score=0.7,
        max_score=0.9,
        min_score=0.5,
    )

    ingest_callable = mocker.Mock(return_value=ingest_result)
    build_callable = mocker.Mock(return_value=candidate_result)

    worker = IngestWorker(
        task_queue=queue,
        ingest_messages=ingest_callable,
        build_candidates=build_callable,
        jitter_provider=lambda base: 0.0,
    )

    processed = worker.process_available_tasks()

    assert processed == 1
    ingest_callable.assert_called_once_with(correlation_id=str(task.task_id))
    build_callable.assert_called_once()
    queue.enqueue.assert_called()
    queue.complete.assert_called_once_with(task.task_id)


def test_ingest_worker_retries_on_error(mocker: MockerFixture) -> None:
    queue = mocker.Mock()
    task = _task(TaskType.INGEST, attempts=1)
    queue.lease.return_value = [task]

    ingest_callable = mocker.Mock(side_effect=RuntimeError("boom"))
    build_callable = mocker.Mock()

    worker = IngestWorker(
        task_queue=queue,
        ingest_messages=ingest_callable,
        build_candidates=build_callable,
        jitter_provider=lambda base: 0.0,
    )

    worker.process_available_tasks()

    queue.fail.assert_called()
    args, kwargs = queue.fail.call_args
    assert args[0] == task.task_id
    assert "boom" in kwargs["error"]
    assert kwargs["retry_at"] is not None


def test_extraction_worker_triggers_digest(mocker: MockerFixture) -> None:
    queue = mocker.Mock()
    task = _task(TaskType.EXTRACTION, attempts=2)
    queue.lease.return_value = [task]

    extraction_result = ExtractionResult(
        events_extracted=3,
        candidates_processed=3,
        llm_calls=3,
        cache_hits=0,
        total_cost_usd=0.45,
        errors=[],
    )
    dedupe_result = DeduplicationResult(new_events=3, merged_events=0, total_events=10)

    extraction_callable = mocker.Mock(return_value=extraction_result)
    dedupe_callable = mocker.Mock(return_value=dedupe_result)

    worker = ExtractionWorker(
        task_queue=queue,
        extract_events=extraction_callable,
        deduplicate_events=dedupe_callable,
        jitter_provider=lambda base: 0.0,
    )

    worker.process_available_tasks()

    extraction_callable.assert_called_once()
    dedupe_callable.assert_called_once()
    queue.enqueue.assert_called()
    queue.complete.assert_called_once_with(task.task_id)


def test_digest_worker_marks_task_complete(mocker: MockerFixture) -> None:
    queue = mocker.Mock()
    task = _task(TaskType.DIGEST)
    queue.lease.return_value = [task]

    digest_result = DigestResult(messages_posted=1, events_included=5, channel="C123")
    publish_callable = mocker.Mock(return_value=digest_result)

    worker = DigestWorker(
        task_queue=queue,
        publish_digest=publish_callable,
        jitter_provider=lambda base: 0.0,
    )

    worker.process_available_tasks()

    publish_callable.assert_called_once()
    queue.complete.assert_called_once_with(task.task_id)
