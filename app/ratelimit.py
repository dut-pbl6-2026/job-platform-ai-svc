"""Per-user token bucket (in-process)."""
import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, per_minute: int):
        self.capacity = max(per_minute, 1)
        self.rate = self.capacity / 60.0
        self._state: dict[str, tuple[float, float]] = defaultdict(lambda: (float(self.capacity), time.monotonic()))

    def allow(self, key: str) -> bool:
        tokens, last = self._state[key]
        now = time.monotonic()
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens < 1.0:
            self._state[key] = (tokens, now)
            return False
        self._state[key] = (tokens - 1.0, now)
        return True
