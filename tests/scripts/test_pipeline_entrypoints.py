from __future__ import annotations

from types import SimpleNamespace

from pydantic import SecretStr


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        log_level="INFO",
        slack_bot_token=SecretStr("slack"),
        openai_api_key=SecretStr("openai"),
        llm_model="gpt-test",
        llm_temperature=0.1,
        llm_timeout_seconds=30,
    )


def test_run_ingest_worker_once(mocker) -> None:
    module = __import__("scripts.run_ingest_worker", fromlist=["main"])

    mocker.patch.object(
        module,
        "parse_args",
        return_value=SimpleNamespace(
            poll_interval_seconds=1.0,
            run_once=True,
            json_logs=False,
        ),
    )
    settings = _settings()
    mocker.patch.object(module, "get_settings", return_value=settings)
    repository = mocker.Mock()
    mocker.patch.object(module, "create_repository", return_value=repository)
    task_queue = mocker.Mock()
    mocker.patch.object(module, "resolve_task_queue", return_value=task_queue)
    mocker.patch.object(module, "SlackClient")
    handlers = (mocker.Mock(), mocker.Mock())
    mocker.patch.object(
        module, "create_slack_ingestion_handlers", return_value=handlers
    )
    worker_instance = mocker.Mock()
    mocker.patch.object(module, "IngestWorker", return_value=worker_instance)
    mocker.patch.object(module.pipeline_runtime, "initialize_logging")
    mocker.patch.object(
        module.pipeline_runtime,
        "create_shutdown_controller",
        return_value=mocker.Mock(),
    )
    mocker.patch.object(module.pipeline_runtime, "install_signal_handlers")
    run_loop = mocker.patch.object(module.pipeline_runtime, "run_worker_loop")

    exit_code = module.main([])

    assert exit_code == 0
    module.IngestWorker.assert_called_once_with(
        task_queue=task_queue,
        ingest_messages=handlers[0],
        build_candidates=handlers[1],
    )
    run_loop.assert_called_once()


def test_run_extraction_worker_once(mocker) -> None:
    module = __import__("scripts.run_extraction_worker", fromlist=["main"])

    mocker.patch.object(
        module,
        "parse_args",
        return_value=SimpleNamespace(
            poll_interval_seconds=2.0,
            run_once=True,
            json_logs=True,
            batch_size=5,
        ),
    )
    settings = _settings()
    mocker.patch.object(module, "get_settings", return_value=settings)
    repository = mocker.Mock()
    mocker.patch.object(module, "create_repository", return_value=repository)
    task_queue = mocker.Mock()
    mocker.patch.object(module, "resolve_task_queue", return_value=task_queue)
    scheduler = mocker.Mock()
    mocker.patch.object(module, "create_llm_scheduler", return_value=scheduler)
    worker_instance = mocker.Mock()
    mocker.patch.object(module, "ExtractionWorker", return_value=worker_instance)
    mocker.patch.object(module.pipeline_runtime, "initialize_logging")
    mocker.patch.object(
        module.pipeline_runtime,
        "create_shutdown_controller",
        return_value=mocker.Mock(),
    )
    mocker.patch.object(module.pipeline_runtime, "install_signal_handlers")
    run_loop = mocker.patch.object(module.pipeline_runtime, "run_worker_loop")

    exit_code = module.main([])

    assert exit_code == 0
    module.ExtractionWorker.assert_called_once_with(
        task_queue=task_queue,
        schedule_llm_tasks=scheduler,
    )
    run_loop.assert_called_once()


def test_run_llm_worker_once(mocker) -> None:
    module = __import__("scripts.run_llm_worker", fromlist=["main"])

    mocker.patch.object(
        module,
        "parse_args",
        return_value=SimpleNamespace(
            poll_interval_seconds=0.5,
            run_once=True,
            json_logs=False,
        ),
    )
    settings = _settings()
    mocker.patch.object(module, "get_settings", return_value=settings)
    repository = mocker.Mock()
    mocker.patch.object(module, "create_repository", return_value=repository)
    task_queue = mocker.Mock()
    mocker.patch.object(module, "resolve_task_queue", return_value=task_queue)
    components = mocker.Mock()
    mocker.patch.object(module, "create_llm_worker_components", return_value=components)
    processor = mocker.Mock()
    mocker.patch.object(
        module, "create_llm_candidate_processor", return_value=processor
    )
    worker_instance = mocker.Mock()
    mocker.patch.object(module, "LLMExtractionWorker", return_value=worker_instance)
    mocker.patch.object(module.pipeline_runtime, "initialize_logging")
    mocker.patch.object(
        module.pipeline_runtime,
        "create_shutdown_controller",
        return_value=mocker.Mock(),
    )
    mocker.patch.object(module.pipeline_runtime, "install_signal_handlers")
    run_loop = mocker.patch.object(module.pipeline_runtime, "run_worker_loop")

    exit_code = module.main([])

    assert exit_code == 0
    module.LLMExtractionWorker.assert_called_once_with(
        task_queue=task_queue,
        process_candidate=processor,
    )
    run_loop.assert_called_once()


def test_run_dedup_worker_once(mocker) -> None:
    module = __import__("scripts.run_dedup_worker", fromlist=["main"])

    mocker.patch.object(
        module,
        "parse_args",
        return_value=SimpleNamespace(
            poll_interval_seconds=3.0,
            run_once=True,
            json_logs=True,
        ),
    )
    settings = _settings()
    mocker.patch.object(module, "get_settings", return_value=settings)
    repository = mocker.Mock()
    mocker.patch.object(module, "create_repository", return_value=repository)
    task_queue = mocker.Mock()
    mocker.patch.object(module, "resolve_task_queue", return_value=task_queue)
    handler = mocker.Mock()
    mocker.patch.object(module, "create_deduplication_handler", return_value=handler)
    worker_instance = mocker.Mock()
    mocker.patch.object(module, "DedupWorker", return_value=worker_instance)
    mocker.patch.object(module.pipeline_runtime, "initialize_logging")
    mocker.patch.object(
        module.pipeline_runtime,
        "create_shutdown_controller",
        return_value=mocker.Mock(),
    )
    mocker.patch.object(module.pipeline_runtime, "install_signal_handlers")
    run_loop = mocker.patch.object(module.pipeline_runtime, "run_worker_loop")

    exit_code = module.main([])

    assert exit_code == 0
    module.DedupWorker.assert_called_once_with(
        task_queue=task_queue,
        deduplicate_events=handler,
    )
    run_loop.assert_called_once()


def test_run_scheduler_once(mocker) -> None:
    module = __import__("scripts.run_pipeline_scheduler", fromlist=["main"])

    mocker.patch.object(
        module,
        "parse_args",
        return_value=SimpleNamespace(
            interval_seconds=10.0,
            include_dedup=True,
            json_logs=False,
            run_once=True,
        ),
    )
    settings = _settings()
    mocker.patch.object(module, "get_settings", return_value=settings)
    repository = mocker.Mock()
    mocker.patch.object(module, "create_repository", return_value=repository)
    task_queue = mocker.Mock()
    mocker.patch.object(module, "resolve_task_queue", return_value=task_queue)
    enqueue = mocker.patch.object(module, "enqueue_pipeline_iteration")
    mocker.patch.object(module.pipeline_runtime, "initialize_logging")
    mocker.patch.object(
        module.pipeline_runtime,
        "create_shutdown_controller",
        return_value=mocker.Mock(),
    )
    mocker.patch.object(module.pipeline_runtime, "install_signal_handlers")
    run_loop = mocker.patch.object(module.pipeline_runtime, "run_scheduler_loop")

    exit_code = module.main([])

    assert exit_code == 0
    run_loop.assert_called_once()
    action = run_loop.call_args.kwargs["action"]
    action()
    enqueue.assert_called()
