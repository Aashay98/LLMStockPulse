"""OpenTelemetry tracing helpers.

The real ``opentelemetry`` package is used when available.  In test
environments where the dependency may be missing, a no-op tracer is
provided so the application code does not fail on import.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

try:  # pragma: no cover - exercised when opentelemetry is installed
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
except Exception:  # pragma: no cover
    trace = None


def init_tracer() -> None:
    """Initialise the global tracer provider if OpenTelemetry is available."""

    if trace is None or trace.get_tracer_provider() is not None:
        return

    provider = TracerProvider()

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint)
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


class _NoOpTracer:
    """Fallback tracer used when OpenTelemetry is not installed."""

    @contextmanager
    def start_as_current_span(self, name: str):  # pragma: no cover - trivial
        yield


def get_tracer(name: str):
    """Return a tracer instance.

    If OpenTelemetry is installed, a real tracer is returned. Otherwise a
    lightweight no-op tracer is provided that supports the ``start_as_current_span``
    context manager used in the codebase.
    """

    if trace is None:
        return _NoOpTracer()

    init_tracer()
    return trace.get_tracer(name)


__all__ = ["get_tracer", "init_tracer"]
