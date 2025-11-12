"""Metrics helpers for pipeline observability.

Architecture Decision:
----------------------
This module uses a pragmatic approach to handle optional prometheus_client dependency:

1. Variables declared with `Any` type before try/except to avoid mypy redefinition errors
2. Fallback classes provide minimal interface compatibility
3. Works both with and without prometheus_client installed

Why not Protocol-based approach?
- Simpler implementation, less code to maintain
- Direct compatibility with existing tests
- Lower risk for production deployment
- Sufficient for current observability needs

Trade-offs accepted:
- Loss of type safety (using `Any`)
- Implicit interface contracts (no Protocol definitions)

When to refactor:
- If we need strict type checking for metrics
- If fallback metrics need to track actual values
- If we add complex metric operations
"""

from __future__ import annotations

import os
import signal
import threading
from types import FrameType
from typing import Any, Final

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class _FallbackMetric:
    def __init__(self, name: str, metric_type: str) -> None:
        self._name = name
        self._metric_type = metric_type
        self._storage: dict[tuple[tuple[str, str], ...], float] = {}

    def labels(self, **labels: str) -> _FallbackMetricInstance:
        key = tuple(sorted(labels.items()))
        return _FallbackMetricInstance(self, key)

    def _increment(self, key: tuple[tuple[str, str], ...], amount: float) -> None:
        self._storage[key] = self._storage.get(key, 0.0) + amount

    def collect(self) -> list[Any]:
        metric_name = (
            f"{self._name}_count"
            if self._metric_type == "histogram"
            else f"{self._name}_total"
        )
        samples = [
            type(
                "Sample",
                (),
                {"name": metric_name, "labels": dict(labels), "value": value},
            )
            for labels, value in self._storage.items()
        ]
        return [type("Metric", (), {"samples": samples})]


class _FallbackMetricInstance:
    def __init__(
        self, metric: _FallbackMetric, key: tuple[tuple[str, str], ...]
    ) -> None:
        self._metric = metric
        self._key = key

    def inc(self, amount: float = 1.0) -> None:
        self._metric._increment(self._key, amount)

    def observe(self, value: float) -> None:  # noqa: ARG002 - value unused
        self._metric._increment(self._key, 1.0)


class _FallbackCounter(_FallbackMetric):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(str(args[0]), "counter")


class _FallbackHistogram(_FallbackMetric):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(str(args[0]), "histogram")


def _fallback_start_http_server(port: int) -> None:
    logger.warning("prometheus_client_unavailable", port=port)


# Try to import prometheus_client, fall back to our implementations
# Declare variables with Any type before assignment
CounterClass: Any
HistogramClass: Any
_start_http_server: Any

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter, Histogram, start_http_server

    CounterClass = Counter
    HistogramClass = Histogram
    _start_http_server = start_http_server
except ImportError:  # pragma: no cover - fallback for offline environments
    CounterClass = _FallbackCounter
    HistogramClass = _FallbackHistogram
    _start_http_server = _fallback_start_http_server

JOBS_SUBMITTED_TOTAL: Final[Any] = CounterClass(
    "pipeline_jobs_submitted_total",
    "Total number of pipeline jobs submitted",
    labelnames=("job",),
)

JOB_DURATION_SECONDS: Final[Any] = HistogramClass(
    "pipeline_job_duration_seconds",
    "Duration of pipeline jobs in seconds",
    labelnames=("job",),
)

PIPELINE_STAGE_DURATION_SECONDS: Final[Any] = HistogramClass(
    "pipeline_stage_duration_seconds",
    "Duration of pipeline stages in seconds",
    labelnames=("stage",),
)

_EXPORTER_LOCK = threading.Lock()
_EXPORTER_STARTED = False
_EXPORTER_STOP_EVENT = threading.Event()
_DEFAULT_METRICS_PORT: Final[int] = 9000
_METRICS_PORT_ENV: Final[str] = "METRICS_PORT"
METRICS_EXPORTER_AUTO_START_ENV: Final[str] = "METRICS_EXPORTER_AUTO_START"


def _should_autostart() -> bool:
    """Return True when the metrics exporter should auto-start."""

    raw_value = os.getenv(METRICS_EXPORTER_AUTO_START_ENV, "1")
    normalized = raw_value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _resolve_metrics_port() -> int:
    port_raw = os.getenv(_METRICS_PORT_ENV)
    try:
        return int(port_raw) if port_raw else _DEFAULT_METRICS_PORT
    except ValueError:
        logger.warning("invalid_metrics_port", port=port_raw)
        return _DEFAULT_METRICS_PORT


def ensure_metrics_exporter() -> None:
    """Start Prometheus HTTP exporter once per process."""

    global _EXPORTER_STARTED
    with _EXPORTER_LOCK:
        if _EXPORTER_STARTED:
            return

        port = _resolve_metrics_port()

        try:
            _start_http_server(port)
        except OSError as exc:
            logger.error(
                "metrics_exporter_start_failed",
                port=port,
                error=str(exc),
            )
            raise

        _EXPORTER_STARTED = True
        logger.info("metrics_exporter_started", port=port)


def _handle_shutdown_signal(signum: int, frame: FrameType | None) -> None:
    logger.info("metrics_exporter_shutdown_signal", signal=signum)
    _EXPORTER_STOP_EVENT.set()


def run_metrics_exporter_forever() -> None:
    """Start the exporter and block until a shutdown signal is received."""

    ensure_metrics_exporter()
    for watched_signal in (signal.SIGTERM, signal.SIGINT):
        signal.signal(watched_signal, _handle_shutdown_signal)

    logger.info("metrics_exporter_listening", port=_resolve_metrics_port())
    _EXPORTER_STOP_EVENT.wait()
    logger.info("metrics_exporter_stopped")


__all__ = [
    "JOBS_SUBMITTED_TOTAL",
    "JOB_DURATION_SECONDS",
    "PIPELINE_STAGE_DURATION_SECONDS",
    "ensure_metrics_exporter",
    "run_metrics_exporter_forever",
]


if _should_autostart():
    ensure_metrics_exporter()


if __name__ == "__main__":
    run_metrics_exporter_forever()
