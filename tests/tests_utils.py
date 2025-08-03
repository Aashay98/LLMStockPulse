import sys
from pathlib import Path

import pytest

# Ensure the repository root is on the Python path for module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from exceptions import ValidationException
from utils import (
    classify_query,
    format_large_number,
    safe_float_conversion,
    validate_password,
    validate_stock_symbol,
    validate_username,
)


def test_validate_stock_symbol_valid():
    assert validate_stock_symbol("aapl") == "AAPL"


def test_validate_stock_symbol_invalid():
    with pytest.raises(ValidationException):
        validate_stock_symbol("invalid$")


def test_classify_query_stock():
    assert classify_query("What is the stock price?") == "stock"


def test_classify_query_sentiment():
    assert classify_query("latest sentiment news") == "sentiment"


def test_classify_query_both():
    assert classify_query("stock sentiment analysis") == "both"


def test_classify_query_general():
    assert classify_query("hello world") == "general"


def test_safe_float_conversion():
    assert safe_float_conversion("1.23") == pytest.approx(1.23)
    assert safe_float_conversion("bad", 2.0) == 2.0


def test_format_large_number():
    assert format_large_number(1500) == "$1.50K"
    assert format_large_number(2_000_000) == "$2.00M"
    assert format_large_number(3_000_000_000) == "$3.00B"
    assert format_large_number(4_000_000_000_000) == "$4.00T"
    assert format_large_number(500) == "$500.00"


def test_validate_username_valid():
    assert validate_username("user_123") == "user_123"


def test_validate_username_invalid():
    with pytest.raises(ValidationException):
        validate_username("ab")
    with pytest.raises(ValidationException):
        validate_username("bad!")


def test_validate_password_valid():
    assert validate_password("Passw0rd") == "Passw0rd"


def test_validate_password_invalid():
    with pytest.raises(ValidationException):
        validate_password("short")
    with pytest.raises(ValidationException):
        validate_password("allletters")
    with pytest.raises(ValidationException):
        validate_password("12345678")
