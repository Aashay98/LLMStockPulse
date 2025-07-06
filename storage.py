"""Database-backed storage for user conversation history."""

from typing import Dict, List

from database import append_history as db_append
from database import clear_history as db_clear
from database import load_history as db_load
from database import load_relevant_history as db_load_relevant


def load_history(user_id: str = "default") -> List[Dict[str, str]]:
    """Return conversation history for the given user."""
    return db_load(user_id)


def append_history(entries: List[Dict[str, str]], user_id: str = "default") -> None:
    """Append messages to a user's history."""
    db_append(entries, user_id)


def clear_history(user_id: str = "default") -> None:
    """Clear all history for a user."""
    db_clear(user_id)


def load_relevant_history(user_id: str, query: str, limit: int) -> List[Dict[str, str]]:
    """Fetch the most relevant history entries for the given query."""
    return db_load_relevant(user_id, query, limit)
