"""Logging configuration utilities with structured output and correlation IDs."""

from __future__ import annotations

import contextvars
import json
import logging
from logging import LogRecord
from typing import Any

# Context variable used to propagate a correlation ID through the call stack
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def set_correlation_id(correlation_id: str) -> None:
    """Set a correlation ID for subsequent log messages."""

    _correlation_id.set(correlation_id)


class CorrelationIdFilter(logging.Filter):
    """Inject the current correlation ID into log records."""

    def filter(self, record: LogRecord) -> bool:  # type: ignore[override]
        record.correlation_id = _correlation_id.get("")
        return True


class JsonFormatter(logging.Formatter):
    """Format log records as structured JSON."""

    def format(self, record: LogRecord) -> str:  # type: ignore[override]
        log_dict: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        correlation_id = getattr(record, "correlation_id", "")
        if correlation_id:
            log_dict["correlation_id"] = correlation_id
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_dict)


def configure_logging(log_file: str = "app.log") -> None:
    """Configure root logging with JSON formatting and correlation IDs.

    This function is idempotent; calling it multiple times will have no effect
    once logging has been configured.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    root_logger.setLevel(logging.INFO)

    formatter = JsonFormatter()
    correlation_filter = CorrelationIdFilter()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(correlation_filter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(correlation_filter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


__all__ = ["configure_logging", "set_correlation_id"]
