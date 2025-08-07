import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Provide required environment variables for config
os.environ.setdefault("ALPHA_VANTAGE_API_KEY", "test")
os.environ.setdefault("NEWS_API_KEY", "test")
os.environ.setdefault("TAVILY_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("MAX_AGENT_ITERATIONS", "5")
os.environ.setdefault("MEMORY_WINDOW_SIZE", "5")

# Ensure repository root on Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools import _fetch_stock_data


def test_fetch_stock_data_uses_cache():
    """Repeated calls with same args should hit the cache."""
    mock_response = {"Time Series (1min)": {}}

    with patch("tools.make_api_request", return_value=mock_response) as mock_api, patch(
        "tools._format_stock_data", return_value="ok"
    ):
        _fetch_stock_data.cache_clear()

        first = _fetch_stock_data("AAPL", "intraday")
        second = _fetch_stock_data("AAPL", "intraday")

        assert first == second == "ok"
        assert mock_api.call_count == 1
