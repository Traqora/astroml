import time
import uuid
from typing import Dict, List, Any, Optional

class ReconnectionManager:
    def __init__(self, ttl_seconds: int = 300):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds

    def create_session(self, prompt: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "prompt": prompt,
            "tokens": [],
            "created_at": time.time(),
            "last_accessed": time.time()
        }
        return session_id

    def append_tokens(self, session_id: str, tokens: List[str]):
        if session_id in self.sessions:
            self.sessions[session_id]["tokens"].extend(tokens)
            self.sessions[session_id]["last_accessed"] = time.time()

    def resume_stream(self, session_id: str, last_token_index: int) -> Optional[List[str]]:
        self._cleanup()
        if session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        session["last_accessed"] = time.time()
        tokens = session["tokens"]
        if last_token_index < len(tokens):
            return tokens[last_token_index:]
        return []

    def _cleanup(self):
        now = time.time()
        expired = [sid for sid, s in self.sessions.items() if now - s["last_accessed"] > self.ttl]
        for sid in expired:
            del self.sessions[sid]

_reconnect_manager = ReconnectionManager()

def get_reconnection_manager() -> ReconnectionManager:
    return _reconnect_manager
