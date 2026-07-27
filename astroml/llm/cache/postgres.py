from typing import Optional, Dict

class PostgresCacheBackend:
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string
        self._store: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def set(self, key: str, value: str) -> bool:
        self._store[key] = value
        return True

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False
