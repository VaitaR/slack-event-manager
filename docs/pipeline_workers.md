# Distributed Pipeline Services

The pipeline now runs as independently scalable workers connected through the
shared task queue. Docker Compose defines the following long-lived services:

| Service | Purpose | Default command |
| --- | --- | --- |
| `pipeline-scheduler` | Periodically enqueues ingestion, extraction, and deduplication jobs. | `python scripts/run_pipeline_scheduler.py --interval-seconds 300` |
| `ingest-worker` | Fetches Slack messages and builds new candidates. | `python scripts/run_ingest_worker.py` |
| `extraction-worker` | Leases extraction tasks and schedules downstream LLM jobs. | `python scripts/run_extraction_worker.py` |
| `llm-worker` | Runs LLM extraction for individual candidates and enqueues dedup tasks. | `python scripts/run_llm_worker.py` |
| `dedup-worker` | Merges newly extracted events and signals digest publication. | `python scripts/run_dedup_worker.py` |

Each worker can be scaled independently. For example, to triple the LLM
capacity:

```bash
docker compose up --scale llm-worker=3 -d
```

## Bootstrapping the Queue

The scheduler continuously submits new tasks. For a one-off bootstrap (for
instance after clearing the queue) run a single iteration:

```bash
docker compose run --rm pipeline-scheduler python scripts/run_pipeline_scheduler.py --run-once
```

## Monitoring Queue Depth

Use the built-in PostgreSQL table to inspect outstanding work:

```bash
docker compose exec postgres psql \
  -U ${POSTGRES_USER:-postgres} \
  -d ${POSTGRES_DATABASE:-slack_events} \
  -c "SELECT task_type, status, COUNT(*)\n      FROM pipeline_tasks\n      GROUP BY task_type, status\n      ORDER BY task_type, status;"
```

The scheduler and workers emit structured logs containing correlation IDs for
traceability. When investigating a backlog, filter logs by `correlation_id`
(value is written by the scheduler for every batch).

## Local Execution

All scripts accept `--run-once` for smoke testing and `--json-logs` for
production-style logging. Example:

```bash
SLACK_BOT_TOKEN=xxx OPENAI_API_KEY=yyy python scripts/run_ingest_worker.py --run-once
```
