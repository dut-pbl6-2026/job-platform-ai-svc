"""Result cache: Redis when configured, else in-process LRU. Only score results are stored, never CV text."""
import json
import logging
import time
from collections import OrderedDict
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


class ResultCache:
    def __init__(self, redis_url: str = "", ttl: int = 86400, max_items: int = 512):
        self.ttl = ttl
        self._mem: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._max = max_items
        self._redis = None
        if redis_url:
            try:
                import redis

                self._redis = redis.Redis(**redis_kwargs(redis_url), socket_timeout=1.0, socket_connect_timeout=1.0)
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
