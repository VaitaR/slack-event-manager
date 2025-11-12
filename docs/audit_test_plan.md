# Audit Test Plan

This plan enumerates the manual verification steps auditors must perform before approving a
production deployment of the Slack Event Manager stack.

## Infrastructure & Observability

1. Confirm Docker services are healthy:
   ```bash
   docker compose ps
   ```
   All services must be listed as `running (healthy)`.
2. Verify the dedicated metrics exporter is the only process binding the host metrics port:
   ```bash
   docker compose ps metrics-exporter
   curl -sf http://localhost:9000/metrics | head
   ```
   The `metrics-exporter` container should be running and the curl command must return Prometheus
   samples.
3. Inspect worker environment variables to ensure `METRICS_EXPORTER_AUTO_START=0` for every
   container that imports `src.observability.metrics`:
   ```bash
   docker inspect slack_pipeline_scheduler | jq '.[0].Config.Env'
   ```
   No other container may expose port 9000 on the host.
