"""Redis-backed conversation memory for LLM multi-turn chat (issue #360)."""
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import redis
except ImportError:
    redis = None


class ConversationSummarizer:
    """Token-budget-aware summarizer for conversation history.

    Uses the same ``~4 chars per token`` heuristic as
    ``BlockchainContextBuilder.analyze_token_size`` (AC4.4).
    """

    def __init__(self, recent_verbatim: int = 10, token_threshold: int = 3000):
        self.recent_verbatim = recent_verbatim
        self.token_threshold = token_threshold

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def _estimate_tokens(self, messages: List[dict]) -> int:
        """Return a rough token count for a list of messages."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, total_chars // 4)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def needs_summarization(self, messages: List[dict]) -> bool:
        """Return True when the estimated token count exceeds the threshold (AC4.1)."""
        return self._estimate_tokens(messages) > self.token_threshold

    def summarize(self, messages: List[dict]) -> List[dict]:
        """Summarize older messages and return ``[summary_msg] + recent`` (AC4.2).

        The oldest ``len(messages) - recent_verbatim`` messages are collapsed
        into a single ``role: "summary"`` message; the last ``recent_verbatim``
        messages are kept verbatim.
        """
        if len(messages) <= self.recent_verbatim:
            # Nothing to summarise — return as-is.
            return list(messages)

        cutoff = len(messages) - self.recent_verbatim
        older = messages[:cutoff]
        recent = messages[cutoff:]

        # Build a bounded summary string (kept well under 900 chars so the
        # summary message itself is stored within the 1 KB budget).
        snippets = "; ".join(
            m.get("content", "")[:50] for m in older
        )
        summary_content = f"Summary of {len(older)} earlier messages: {snippets}"
        # Truncate to 900 chars to respect the per-message size limit.
        summary_content = summary_content[:900]

        summary_msg = {
            "role": "summary",
            "content": summary_content,
            "ts": datetime.utcnow().isoformat(),
        }

        return [summary_msg] + list(recent)


class ConversationMemory:
    """Manages per-session conversation history in Redis.

    Falls back to an in-process dict when Redis is unavailable (AC1.4).

    Redis key layout:
        ``conv:{session_id}:messages``  — JSON list of message dicts
        ``conv:{session_id}:meta``      — JSON session metadata dict
    """

    def __init__(
        self,
        ttl: int = 86400,
        max_messages: int = 40,
        summarize_threshold_tokens: int = 3000,
        recent_verbatim: int = 10,
    ):
        self.ttl = ttl
        self.max_messages = max_messages
        self._summarizer = ConversationSummarizer(
            recent_verbatim=recent_verbatim,
            token_threshold=summarize_threshold_tokens,
        )

        # Attempt Redis connection — same pattern as SemanticCache.
        self._redis: Optional[object] = None
        if redis is not None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            try:
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                # Verify connectivity eagerly; fall back if the server is down.
                self._redis.ping()
            except Exception:
                logger.warning(
                    "ConversationMemory: Redis unavailable — using in-memory fallback."
                )
                self._redis = None

        # In-process fallback stores (keyed by session_id).
        self._fallback_messages: Dict[str, List[dict]] = {}
        self._fallback_meta: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _msg_key(self, session_id: str) -> str:
        return f"conv:{session_id}:messages"

    def _meta_key(self, session_id: str) -> str:
        return f"conv:{session_id}:meta"

    def _redis_ok(self) -> bool:
        return self._redis is not None

    # ---------- low-level read/write ----------

    def _read_messages(self, session_id: str) -> List[dict]:
        if self._redis_ok():
            try:
                raw = self._redis.get(self._msg_key(session_id))
                if raw:
                    return json.loads(raw)
                return []
            except Exception:
                pass
        return list(self._fallback_messages.get(session_id, []))

    def _write_messages(self, session_id: str, messages: List[dict]) -> None:
        payload = json.dumps(messages, ensure_ascii=False)
        if self._redis_ok():
            try:
                self._redis.setex(self._msg_key(session_id), self.ttl, payload)
                return
            except Exception:
                pass
        self._fallback_messages[session_id] = messages

    def _read_meta(self, session_id: str) -> Optional[dict]:
        if self._redis_ok():
            try:
                raw = self._redis.get(self._meta_key(session_id))
                if raw:
                    return json.loads(raw)
                return None
            except Exception:
                pass
        return self._fallback_meta.get(session_id)

    def _write_meta(self, session_id: str, meta: dict) -> None:
        payload = json.dumps(meta, ensure_ascii=False)
        if self._redis_ok():
            try:
                self._redis.setex(self._meta_key(session_id), self.ttl, payload)
                return
            except Exception:
                pass
        self._fallback_meta[session_id] = meta

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self) -> dict:
        """Create a new session and return its metadata (AC2.1, AC2.4).

        Returns a dict with ``id``, ``created_at``, ``turn_count``, and
        ``is_summarized``.
        """
        session_id = str(uuid.uuid4())
        meta = {
            "id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "turn_count": 0,
            "is_summarized": False,
        }
        self._write_meta(session_id, meta)
        # Initialise an empty message list so the key exists.
        self._write_messages(session_id, [])
        return meta

    def get_session(self, session_id: str) -> Optional[dict]:
        """Return session metadata or ``None`` if the session doesn't exist (AC2.2, AC2.5)."""
        return self._read_meta(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Remove both Redis keys for the session.

        Returns ``True`` if the session existed, ``False`` otherwise (AC2.3).
        """
        existed = self._read_meta(session_id) is not None

        if self._redis_ok():
            try:
                self._redis.delete(self._msg_key(session_id), self._meta_key(session_id))
            except Exception:
                pass
        # Always clean up fallback stores too (handles mixed-mode edge case).
        self._fallback_messages.pop(session_id, None)
        self._fallback_meta.pop(session_id, None)

        return existed

    # ------------------------------------------------------------------
    # Message I/O
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history and refresh the TTL (AC1.1–AC1.5).

        ``content`` is truncated to 900 characters to keep per-message overhead
        under 1 KB (AC1.3).
        """
        messages = self._read_messages(session_id)
        message = {
            "role": role,
            "content": content[:900],
            "ts": datetime.utcnow().isoformat(),
        }
        messages.append(message)
        self._write_messages(session_id, messages)

        # Update turn_count in metadata if the session exists.
        meta = self._read_meta(session_id)
        if meta is not None:
            meta["turn_count"] = meta.get("turn_count", 0) + 1
            self._write_meta(session_id, meta)

    def get_messages(self, session_id: str) -> List[dict]:
        """Return the full stored message list (empty list if session missing)."""
        return self._read_messages(session_id)

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, session_id: str) -> None:
        """Trim history to ``max_messages``, summarising the excess (AC5.1–AC5.3).

        The operation is idempotent: calling it multiple times produces the
        same result as calling it once (AC5.3).
        """
        messages = self._read_messages(session_id)

        if not messages:
            return

        # Step 1 — summarise if token budget exceeded.
        if self._summarizer.needs_summarization(messages):
            messages = self._summarizer.summarize(messages)
            # Mark the session as summarised.
            meta = self._read_meta(session_id)
            if meta is not None:
                meta["is_summarized"] = True
                self._write_meta(session_id, meta)

        # Step 2 — hard-trim to max_messages.
        if len(messages) > self.max_messages:
            # Keep the most-recent max_messages entries.
            messages = messages[-self.max_messages:]

        self._write_messages(session_id, messages)
