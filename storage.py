"""Database-backed storage for user conversation history."""

from typing import Any, Dict, List

from database import append_history as db_append
from database import clear_history as db_clear
from database import create_conversation as db_create_conversation
from database import get_conversations as db_get_conversations
from database import load_history as db_load
from database import load_relevant_history as db_load_relevant


def load_history(user_id: str, conversation_id: int) -> List[Dict[str, str]]:
    """Return conversation history for the given user and conversation."""
    if user_id == "guest" or conversation_id is None:
        return []
    return db_load(user_id, conversation_id)


def append_history(
    entries: List[Dict[str, str]], user_id: str, conversation_id: int
) -> None:
    """Append messages to a user's conversation history."""
    if user_id == "guest" or conversation_id is None:
        return
    db_append(entries, user_id, conversation_id)


def clear_history(user_id: str, conversation_id: int) -> None:
    """Clear all history for a conversation."""
    if user_id == "guest" or conversation_id is None:
        return
    db_clear(user_id, conversation_id)


def load_relevant_history(
    user_id: str, conversation_id: int, query: str, limit: int
) -> List[Dict[str, str]]:
    """Fetch the most relevant history entries for the given query."""
    if user_id == "guest" or conversation_id is None:
        return []
    return db_load_relevant(user_id, conversation_id, query, limit)


def create_conversation(user_id: str, title: str) -> int:
    """Create a new conversation for a user."""
    if user_id == "guest":
        return -1
    return db_create_conversation(user_id, title)


def get_conversations(user_id: str) -> List[Dict[str, Any]]:
    """Return all conversations for a user."""
    if user_id == "guest":
        return []
    return db_get_conversations(user_id)
