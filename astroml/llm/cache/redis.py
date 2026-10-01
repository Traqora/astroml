from typing import Any, Dict, List, Optional, Union, Callable
class RedisCacheBackend:
    def __init__(self, host -> Any: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        # Fallback local in-memory storage
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str, expire_seconds: int | None = None) -> bool:
        self._store[key] = value
        return True

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False
