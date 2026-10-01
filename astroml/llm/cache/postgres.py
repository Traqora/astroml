from typing import Any, Dict, List, Optional, Union, Callable
class PostgresCacheBackend:
    def __init__(self, connection_string -> Any: str | None = None):
        self.connection_string = connection_string
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> bool:
        self._store[key] = value
        return True

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False
