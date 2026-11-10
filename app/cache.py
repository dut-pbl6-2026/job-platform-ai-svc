"""Result cache: Redis when configured, else in-process LRU. Only score results are stored, never CV text."""
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from urllib.parse import unquote

log = logging.getLogger("ai.cache")


def _bad(what: str) -> ValueError:
    # Never echo the value: it carries the Redis password.
    return ValueError(f"REDIS_URL is invalid: {what}")


def _port(text: str) -> int:
    if not text.isdigit() or not 0 < int(text) < 65536:
        raise _bad("port must be a number between 1 and 65535")
    return int(text)


def redis_kwargs(value: str) -> dict:
    """Keyword arguments for ``redis.Redis`` from either flavour of ``REDIS_URL``.

    ``redis://[user:pass@]host[:port][/db]`` / ``rediss://...``, or the StackExchange.Redis string the .NET
    services share: ``host[:port][,password=..][,user=..][,ssl=true][,defaultDatabase=n]`` (keys are
    case-insensitive, unknown keys are ignored). Errors never contain the value.
    """
    value = value.strip()
    out = {"host": "", "port": 6379, "db": 0, "username": None, "password": None, "ssl": False}
    if "://" in value:
        scheme, _, rest = value.partition("://")
        if scheme.lower() not in ("redis", "rediss"):
            raise _bad("scheme must be redis or rediss")
        out["ssl"] = scheme.lower() == "rediss"
        auth, at, hostpart = rest.rpartition("@")
        if at:
            user, colon, password = auth.partition(":")
            out["username"] = unquote(user) or None
            out["password"] = unquote(password) if colon else None
        hostport, _, path = hostpart.partition("/")
        host, colon, port = hostport.rpartition(":")
        if not colon:
            host, port = hostport, ""
        if port:
            out["port"] = _port(port)
        if path:
            if not path.isdigit():
                raise _bad("database must be a number")
            out["db"] = int(path)
    else:
        head, *options = value.split(",")
        host, colon, port = head.strip().rpartition(":")
        if not colon:
            host, port = head.strip(), ""
        if port:
            out["port"] = _port(port)
        for option in options:
            key, eq, val = option.partition("=")
            if not eq:
                continue
            key, val = key.strip().lower(), val.strip()
            if key == "password":
                out["password"] = val or None
            elif key == "user":
                out["username"] = val or None
            elif key == "ssl":
                out["ssl"] = val.lower() == "true"
            elif key == "defaultdatabase":
                if not val.isdigit():
                    raise _bad("defaultDatabase must be a number")
                out["db"] = int(val)
    if not host.strip():
        raise _bad("host is missing")
    out["host"] = host.strip()
    return out


def connect(redis_url: str):
    """Build a Redis client from ``REDIS_URL`` (no network I/O). None when unset or invalid."""
    if not redis_url:
        return None
    try:
        import redis

        return redis.Redis(**redis_kwargs(redis_url), socket_timeout=1.0, socket_connect_timeout=1.0)
    except Exception as exc:  # noqa: BLE001 - the message of a redis error can contain the password; log the type only
        log.warning("Redis not configured (%s); using in-process cache", type(exc).__name__)
        return None


class ResultCache:
    """Redis with an in-process LRU behind it.

    A failing Redis command switches to memory; Redis is probed again after ``retry_seconds`` so a Redis that
    comes up later (or recovers) is picked up without a restart. Calls are blocking: run them in a worker thread.
    """

    def __init__(self, redis_url: str = "", ttl: int = 86400, max_items: int = 512, *, prefix: str = "ai:score:", client=None, clock: Callable[[], float] = time.time, retry_seconds: float = 30.0):
        self.ttl = ttl
        self._prefix = prefix
        self._mem: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_items
        self._clock = clock
        self._retry = retry_seconds
        self._client = client if client is not None else connect(redis_url)
        self._down_until = 0.0  # while clock() < this, Redis is not used
        self._up = False
        if self._client is not None:
            self._probe()

    def _probe(self) -> bool:
        try:
            self._client.ping()
        except Exception as exc:  # noqa: BLE001
            self._mark_down(exc)
            return False
        if not self._up:
            log.info("Redis cache available")
        self._up = True
        return True

    def _mark_down(self, exc: Exception) -> None:
        if self._up or self._down_until == 0.0:
            log.warning("Redis unavailable (%s); using in-process cache, retrying in %ds", type(exc).__name__, self._retry)
        self._up = False
        self._down_until = self._clock() + self._retry

    def _redis(self):
        """The client if Redis may be used right now, else None (memory)."""
        if self._client is None:
            return None
        if self._up:
            return self._client
        if self._clock() >= self._down_until and self._probe():
            return self._client
        return None

    @property
    def backend(self) -> str:
        return "redis" if self._redis() is not None else "memory"

    def get(self, key: str) -> dict | None:
        client = self._redis()
        if client is not None:
            try:
                raw = client.get(f"{self._prefix}{key}")
                return json.loads(raw) if raw else None
            except Exception as exc:  # noqa: BLE001
                self._mark_down(exc)
        with self._lock:
            item = self._mem.get(key)
            if not item or item[0] < self._clock():
                self._mem.pop(key, None)
                return None
            self._mem.move_to_end(key)
            return json.loads(item[1])

    def set(self, key: str, value: dict) -> None:
        raw = json.dumps(value, ensure_ascii=False)
        client = self._redis()
        if client is not None:
            try:
                client.setex(f"{self._prefix}{key}", self.ttl, raw)
                return
            except Exception as exc:  # noqa: BLE001
                self._mark_down(exc)
        with self._lock:
            self._mem[key] = (self._clock() + self.ttl, raw)
            self._mem.move_to_end(key)
            while len(self._mem) > self._max:
                self._mem.popitem(last=False)

    def delete(self, key: str) -> None:
        """Remove ``key`` from Redis (when reachable) and from memory."""
        client = self._redis()
        if client is not None:
            try:
                client.delete(f"{self._prefix}{key}")
            except Exception as exc:  # noqa: BLE001
                self._mark_down(exc)
        with self._lock:
            self._mem.pop(key, None)
