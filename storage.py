import json
import os
from threading import Lock
from typing import List, Dict

HISTORY_FILE = "conversation_history.json"
_lock = Lock()


def _load_all() -> Dict[str, List[Dict[str, str]]]:
    """Load the entire history JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def load_history(user_id: str = "default") -> List[Dict[str, str]]:
    """Return conversation history for a given user."""
    data = _load_all()
    return data.get(user_id, [])


def append_history(entries: List[Dict[str, str]], user_id: str = "default") -> None:
    """Append new conversation entries for a user."""
    with _lock:
        data = _load_all()
        history = data.get(user_id, [])
        history.extend(entries)
        data[user_id] = history
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def clear_history(user_id: str = "default") -> None:
    """Remove history for a user from disk."""
    with _lock:
        if not os.path.exists(HISTORY_FILE):
            return
        data = _load_all()
        if user_id in data:
            del data[user_id]
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
