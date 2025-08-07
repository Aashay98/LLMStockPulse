import json
import logging
import sys
from pathlib import Path

# Ensure repository root is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from log_config import configure_logging, set_correlation_id


def test_structured_logging(tmp_path):
    """Logs should be emitted as JSON and include the correlation ID."""

    # Ensure a clean logging slate
    logging.getLogger().handlers.clear()

    log_file = tmp_path / "app.log"
    set_correlation_id("abc123")
    configure_logging(str(log_file))

    logger = logging.getLogger("test")
    logger.info("hello")

    log_line = log_file.read_text().strip().splitlines()[-1]
    data = json.loads(log_line)

    assert data["message"] == "hello"
    assert data["correlation_id"] == "abc123"
    assert data["level"] == "INFO"
