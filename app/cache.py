"""Result cache: Redis when configured, else in-process LRU. Only score results are stored, never CV text."""
import json
import logging
import time
from collections import OrderedDict

log = logging.getLogger("ai.cache")


class ResultCache:
    def __init__(self, redis_url: str = "", ttl: int = 86400, max_items: int = 512):
        self.ttl = ttl
        self._mem: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._max = max_items
        self._redis = None
        if redis_url:
            try:
                import redis

                url = redis_url if "://" in redis_url else f"redis://{redis_url}"
                self._redis = redis.Redis.from_url(url, socket_timeout=1.0, socket_connect_timeout=1.0)
                self._redis.ping()
            except Exception as exc:  # noqa: BLE001
                log.warning("Redis unavailable (%s); using in-process cache", type(exc).__name__)
                self._redis = None

    @property
    def backend(self) -> str:
        return "redis" if self._redis else "memory"

    def get(self, key: str) -> dict | None:
        if self._redis:
            try:
                raw = self._redis.get(f"ai:score:{key}")
                return json.loads(raw) if raw else None
            except Exception:  # noqa: BLE001
                return None
        item = self._mem.get(key)
        if not item or item[0] < time.time():
            self._mem.pop(key, None)
            return None
        self._mem.move_to_end(key)
        return json.loads(item[1])

    def set(self, key: str, value: dict) -> None:
        raw = json.dumps(value, ensure_ascii=False)
        if self._redis:
            try:
                self._redis.setex(f"ai:score:{key}", self.ttl, raw)
                return
            except Exception:  # noqa: BLE001
                pass
        self._mem[key] = (time.time() + self.ttl, raw)
        self._mem.move_to_end(key)
        while len(self._mem) > self._max:
            self._mem.popitem(last=False)
