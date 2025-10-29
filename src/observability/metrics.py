"""Metrics helpers for pipeline observability."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Final

from src.config.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from prometheus_client import Counter as PromCounter
    from prometheus_client import Histogram as PromHistogram

try:  # pragma: no cover - optional dependency
    from prometheus_client import (
        Counter as PromCounter,
    )
    from prometheus_client import (
        Histogram as PromHistogram,
    )
    from prometheus_client import (
        start_http_server,
    )
except ImportError:  # pragma: no cover - fallback for offline environments
    from typing import Any

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

    def start_http_server(port: int) -> None:  # type: ignore[misc]
        logger.warning("prometheus_client_unavailable", port=port)

    CounterClass: Any = _FallbackCounter
    HistogramClass: Any = _FallbackHistogram
else:
    CounterClass = PromCounter
    HistogramClass = PromHistogram

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
_DEFAULT_METRICS_PORT: Final[int] = 9000


def ensure_metrics_exporter() -> None:
    """Start Prometheus HTTP exporter once per process."""

    global _EXPORTER_STARTED
    with _EXPORTER_LOCK:
        if _EXPORTER_STARTED:
            return

        port_raw = os.getenv("METRICS_PORT")
        try:
            port = int(port_raw) if port_raw else _DEFAULT_METRICS_PORT
        except ValueError:
            logger.warning("invalid_metrics_port", port=port_raw)
            port = _DEFAULT_METRICS_PORT

        start_http_server(port)
        _EXPORTER_STARTED = True
        logger.info("metrics_exporter_started", port=port)


__all__ = [
    "JOBS_SUBMITTED_TOTAL",
    "JOB_DURATION_SECONDS",
    "PIPELINE_STAGE_DURATION_SECONDS",
    "ensure_metrics_exporter",
]
