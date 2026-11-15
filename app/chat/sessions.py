"""Chat sessions in the shared cache (Redis when configured, else in-process), keyed by session id.

A record is {owner, created_at, updated_at, messages}. Only the owner can read or write it: anyone else sees
the session as missing. Messages are kept as sent by the user and returned by the model-free engine.
"""
import uuid
from datetime import UTC, datetime

from app.cache import ResultCache

MAX_MESSAGES = 20  # the context window: older messages are dropped (CHAT_MAX_MESSAGES in the web client)


def canonical_id(value: object) -> str | None:
    """The canonical UUID string, or None. Only canonical ids are ever used as cache keys."""
    if not isinstance(value, str):
        return None
    try:
        return str(uuid.UUID(value.strip()))
    except ValueError:
        return None


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ChatSessions:
    def __init__(self, store: ResultCache):
        self._store = store

    def load(self, owner: str, session_id: str) -> dict | None:
        record = self._store.get(session_id)
        if not record or record.get("owner") != owner:
            return None
        return record

    def save(self, owner: str, session_id: str, messages: list[dict], created_at: str | None = None) -> dict:
        now = now_iso()
        record = {
            "owner": owner,
            "created_at": created_at or now,
            "updated_at": now,
            "messages": messages[-MAX_MESSAGES:],
        }
        self._store.set(session_id, record)
        return record

    def delete(self, owner: str, session_id: str) -> None:
        if self.load(owner, session_id) is not None:
            self._store.delete(session_id)
