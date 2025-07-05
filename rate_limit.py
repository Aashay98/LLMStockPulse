import json
import os
from datetime import datetime
from threading import Lock
from typing import Dict

# Usage tracking file
USAGE_FILE = "api_usage.json"
_lock = Lock()

# Default API limits (can be overridden via environment variables)
USAGE_LIMITS = {
    "alpha_vantage": int(os.getenv("ALPHA_VANTAGE_DAILY_LIMIT", 25)),
    "newsapi": int(os.getenv("NEWSAPI_MONTHLY_LIMIT", 1000)),
    "tavily": int(os.getenv("TAVILY_MONTHLY_LIMIT", 1000)),
    "groq": int(os.getenv("GROQ_DAILY_LIMIT", 1000)),
    "twitter": int(os.getenv("TWITTER_DAILY_LIMIT", 300)),
}

PERIODS = {
    "alpha_vantage": "day",
    "newsapi": "month",
    "tavily": "month",
    "groq": "day",
    "twitter": "day",
}


def _load_usage() -> Dict[str, Dict[str, int]]:
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def _save_usage(data: Dict[str, Dict[str, int]]) -> None:
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _current_period(api_name: str) -> str:
    now = datetime.utcnow()
    if PERIODS.get(api_name) == "month":
        return now.strftime("%Y-%m")
    return now.strftime("%Y-%m-%d")


def increment_usage(api_name: str) -> int:
    """Increment usage counter and return current count."""
    with _lock:
        data = _load_usage()
        api_data = data.get(api_name, {})
        period = _current_period(api_name)
        if api_data.get("period") != period:
            api_data = {"count": 0, "period": period}
        api_data["count"] = api_data.get("count", 0) + 1
        data[api_name] = api_data
        _save_usage(data)
        return api_data["count"]


def check_api_usage(api_name: str) -> None:
    """Raise an exception if API quota exceeded."""
    count = increment_usage(api_name)
    limit = USAGE_LIMITS.get(api_name)
    remaining = limit - count
    if remaining <= 5:
        # Log warning when quota is low
        import logging

        logging.getLogger(__name__).warning(
            "%s API quota low: %d calls remaining", api_name, remaining
        )
    if remaining < 0:
        from exceptions import APIException

        raise APIException(f"{api_name} API quota exceeded")
