"""Application metrics exposed for Prometheus scraping.

This module attempts to use the ``prometheus_client`` package. If the
dependency is unavailable (e.g. during offline tests), a minimal fallback
implementation is provided so the rest of the application can continue to
operate.  The fallback implements the small subset of the API used in the
project: ``Counter``, ``Histogram`` and ``start_http_server``.
"""

from __future__ import annotations

import logging

try:  # pragma: no cover - exercised in tests via fallback
    from prometheus_client import Counter, Histogram, start_http_server
except Exception:  # pragma: no cover
    # ------------------------------------------------------------------
    # Fallback lightweight metrics implementation.
    # ------------------------------------------------------------------
    class _MetricInstance:
        def __init__(self, parent: "_Metric", key: tuple[tuple[str, str], ...]):
            self.parent = parent
            self.key = key

        def inc(self, amount: float = 1.0) -> None:
            self.parent._data[self.key] = self.parent._data.get(self.key, 0.0) + amount

        def observe(self, value: float) -> None:
            self.inc(value)

        def time(self):
            import time
            from contextlib import contextmanager

            @contextmanager
            def _timer():
                start = time.perf_counter()
                yield
                self.observe(time.perf_counter() - start)

            return _timer()

        @property
        def _value(self):  # mimic prometheus_client API used in tests
            class _Val:
                def __init__(self, parent: "_Metric", key: tuple[tuple[str, str], ...]):
                    self.parent = parent
                    self.key = key

                def get(self) -> float:
                    return self.parent._data.get(self.key, 0.0)

            return _Val(self.parent, self.key)

    class _Metric:
        def __init__(self, name: str, description: str, labelnames: list[str]):
            self._data: dict[tuple[tuple[str, str], ...], float] = {}

        def labels(self, **labels: str) -> _MetricInstance:
            key = tuple(sorted(labels.items()))
            if key not in self._data:
                self._data[key] = 0.0
            return _MetricInstance(self, key)

    class Counter(_Metric):
        pass

    class Histogram(_Metric):
        pass

    def start_http_server(port: int) -> None:  # no-op
        return None


# ----------------------------------------------------------------------
# Metric definitions
# ----------------------------------------------------------------------
REQUEST_LATENCY = Histogram(
    "query_latency_seconds",
    "Time spent processing user queries",
    ["user"],
)

REQUEST_COUNTER = Counter(
    "queries_total", "Total number of user queries processed", ["user"]
)

ERROR_COUNTER = Counter(
    "errors_total", "Number of errors encountered while processing queries", ["user"]
)


def start_metrics_server(port: int = 8000) -> None:
    """Start an HTTP server for Prometheus to scrape metrics.

    If the port is already in use, a warning is logged and the function
    returns without raising an exception.
    """

    try:
        start_http_server(port)
    except OSError as exc:  # pragma: no cover - depends on system state
        logging.getLogger(__name__).warning(
            "Metrics server not started on port %s: %s", port, exc
        )


def get_metric_value(metric, **labels) -> float:
    """Return the current value of a metric for the given label set."""

    instance = metric.labels(**labels)
    value = getattr(instance, "_value", None)
    if value is None:  # pragma: no cover
        return 0.0
    getter = getattr(value, "get", None)
    return getter() if callable(getter) else value


__all__ = [
    "REQUEST_LATENCY",
    "REQUEST_COUNTER",
    "ERROR_COUNTER",
    "start_metrics_server",
    "get_metric_value",
]
