from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pytest_mock import MockerFixture

from src.domain.models import (
    CandidateResult,
    DeduplicationResult,
    DigestResult,
    IngestResult,
)
from src.domain.task_queue import Task, TaskStatus, TaskType
from src.use_cases.extract_events import (
    CandidateExtractionMetrics,
    LLMTaskScheduleResult,
)
from src.workers.pipeline import (
    DedupWorker,
    DigestWorker,
    ExtractionWorker,
    IngestWorker,
    LLMExtractionWorker,
)


def _task(
    task_type: TaskType,
    attempts: int = 1,
    payload: dict[str, object] | None = None,
) -> Task:
    now = datetime.now(tz=UTC)
    return Task(
        task_id=uuid4(),
        task_type=task_type,
        payload=payload or {},
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


def test_extraction_worker_schedules_llm_tasks(mocker: MockerFixture) -> None:
    queue = mocker.Mock()
    task = _task(
        TaskType.EXTRACTION,
        attempts=2,
        payload={"correlation_id": "corr-1", "candidates_created": 5},
    )
    queue.lease.return_value = [task]

    schedule_result = LLMTaskScheduleResult(total_candidates=5, candidates_enqueued=3)
    scheduler = mocker.Mock(return_value=schedule_result)

    worker = ExtractionWorker(
        task_queue=queue,
        schedule_llm_tasks=scheduler,
        jitter_provider=lambda base: 0.0,
    )

    worker.process_available_tasks()

    scheduler.assert_called_once_with(correlation_id="corr-1", batch_hint=5)
    queue.enqueue.assert_not_called()
    queue.complete.assert_called_once_with(task.task_id)


def test_llm_worker_enqueues_dedup_when_required(mocker: MockerFixture) -> None:
    queue = mocker.Mock()
    task = _task(
        TaskType.LLM_EXTRACTION,
        payload={"message_id": "msg-123", "correlation_id": "corr"},
    )
    queue.lease.return_value = [task]

    metrics = CandidateExtractionMetrics(
        events_extracted=2,
        llm_calls=1,
        cache_hits=0,
        total_cost_usd=0.25,
        dedup_required=True,
    )
    processor = mocker.Mock(return_value=metrics)

    worker = LLMExtractionWorker(
        task_queue=queue,
        process_candidate=processor,
        jitter_provider=lambda base: 0.0,
    )

    worker.process_available_tasks()

    processor.assert_called_once_with(message_id="msg-123", correlation_id="corr")
    queue.enqueue.assert_called_once()
    enqueued = queue.enqueue.call_args.args[0]
    assert enqueued.task_type is TaskType.DEDUP
    assert enqueued.payload["origin_message_id"] == "msg-123"
    queue.complete.assert_called_once_with(task.task_id)


def test_llm_worker_skips_dedup_when_not_needed(mocker: MockerFixture) -> None:
    queue = mocker.Mock()
    task = _task(
        TaskType.LLM_EXTRACTION,
        payload={"message_id": "msg-456", "correlation_id": "corr"},
    )
    queue.lease.return_value = [task]

    metrics = CandidateExtractionMetrics(
        events_extracted=0,
        llm_calls=0,
        cache_hits=1,
        total_cost_usd=0.0,
        dedup_required=False,
    )
    processor = mocker.Mock(return_value=metrics)

    worker = LLMExtractionWorker(
        task_queue=queue,
        process_candidate=processor,
        jitter_provider=lambda base: 0.0,
    )

    worker.process_available_tasks()

    queue.enqueue.assert_not_called()
    queue.complete.assert_called_once_with(task.task_id)


def test_dedup_worker_enqueues_digest(mocker: MockerFixture) -> None:
    queue = mocker.Mock()
    task = _task(TaskType.DEDUP, payload={"correlation_id": "corr"})
    queue.lease.return_value = [task]

    dedupe_result = DeduplicationResult(new_events=2, merged_events=1, total_events=5)
    dedupe_callable = mocker.Mock(return_value=dedupe_result)

    worker = DedupWorker(
        task_queue=queue,
        deduplicate_events=dedupe_callable,
        jitter_provider=lambda base: 0.0,
    )

    worker.process_available_tasks()

    dedupe_callable.assert_called_once()
    queue.enqueue.assert_called_once()
    enqueued = queue.enqueue.call_args.args[0]
    assert enqueued.task_type is TaskType.DIGEST
    assert enqueued.payload["correlation_id"] == "corr"
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
