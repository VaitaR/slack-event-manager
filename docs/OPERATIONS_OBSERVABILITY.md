# Observability and Health Checks

This guide explains how to monitor the Slack Event Manager services once they are running in a
containerized environment. All commands assume you are running them from the host that launched the
Docker Compose stack.

## Metrics Exporter

The Docker Compose stack now exposes metrics through a dedicated `metrics-exporter` container. That
process imports `src.observability.metrics`, starts the Prometheus HTTP server, and waits for a
shutdown signal. All worker-style containers set `METRICS_EXPORTER_AUTO_START=0`, preventing port
conflicts on the host.

- Endpoint: `http://<host>:9000/metrics`
- Default bind address: `0.0.0.0`
- Dedicated container: `metrics-exporter`
- Disable auto-start (e.g., for tests or ad-hoc scripts): set `METRICS_EXPORTER_AUTO_START=0`
- Override port: set `METRICS_PORT=<port>`

Verify that metrics are reachable from the host:

```bash
curl -sf http://localhost:9000/metrics | head
```

If the command above succeeds you should see Prometheus samples such as
`pipeline_jobs_submitted_total` and `pipeline_stage_duration_seconds`.

Need to expose metrics from a one-off script? Export `METRICS_EXPORTER_AUTO_START=0`, import
`ensure_metrics_exporter()`, and call it explicitly. Alternatively, run `python -m
src.observability.metrics` to launch the HTTP server and keep the process alive until it receives a
shutdown signal.

## Streamlit UI Health

Streamlit exposes a lightweight health endpoint that can be queried without rendering the UI. This
is the same endpoint used by the container healthcheck.

```bash
curl -sf http://localhost:8501/_stcore/health
```

Expected output:

```json
{"status": "ok"}
```

If the endpoint is unavailable:

1. Inspect container logs: `docker compose logs streamlit-ui`
2. Confirm dependent services are healthy: `docker compose ps`
3. Check for port conflicts on the host: `lsof -i :8501`

## Operational Notes

- Metrics are served without authentication. Restrict network access to trusted environments or
  front them with an authenticated reverse proxy. If multiple exporters must coexist, front them
  with a reverse proxy that selects the desired target based on the request path.
- When auto-start is disabled, you can still export metrics manually by calling
  `src.observability.metrics.ensure_metrics_exporter()` in your entrypoint before launching the UI,
  or by running the dedicated exporter module as described above.
