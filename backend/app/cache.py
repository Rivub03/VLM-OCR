import time
from typing import Any


class ResultCache:
    """Process-local, TTL-only compatibility cache; it is not a job queue or database."""

    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, tuple[float, Any]] = {}

    def put(self, key: str, value: Any) -> None:
        self._prune()
        self._items[key] = (time.monotonic() + self.ttl_seconds, value)

    def get(self, key: str) -> Any | None:
        self._prune()
        entry = self._items.get(key)
        return entry[1] if entry else None

    def _prune(self) -> None:
        now = time.monotonic()
        for key, (expires, _) in list(self._items.items()):
            if expires <= now:
                del self._items[key]

