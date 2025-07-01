import pytest
import sys
import types
import importlib.util
from pathlib import Path

# utils depends on the `tiktoken` package which isn't required for these tests.
# Create a lightweight stub so the import succeeds even if the package isn't
# installed in the test environment.
sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))

# Import utils via its file path to avoid issues with PYTHONPATH configuration
utils_path = Path(__file__).resolve().parents[1] / "utils.py"
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
classify_query = utils.classify_query


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is the current stock price of Apple?", "stock"),
        ("Show me the latest market data", "stock"),
    ],
)
def test_classify_stock_queries(query, expected):
    assert classify_query(query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What's the sentiment trend for Tesla?", "sentiment"),
        ("Summarize the news sentiment", "sentiment"),
    ],
)
def test_classify_sentiment_queries(query, expected):
    assert classify_query(query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is the market sentiment for Nasdaq?", "both"),
        ("Provide news and stock market trends", "both"),
    ],
)
def test_classify_both_queries(query, expected):
    assert classify_query(query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("How's the weather today?", "general"),
        ("Tell me a joke", "general"),
    ],
)
def test_classify_general_queries(query, expected):
    assert classify_query(query) == expected
