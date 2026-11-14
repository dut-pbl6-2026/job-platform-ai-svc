import pytest
import redis

from app.cache import ResultCache, redis_kwargs


def kw(**over):
    base = {"host": "h", "port": 6379, "db": 0, "username": None, "password": None, "ssl": False}
    return {**base, **over}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("redis://h:1/2", kw(port=1, db=2)),
        ("redis://h", kw()),
        ("redis://:s3cret@h:6380/0", kw(port=6380, password="s3cret")),
        ("redis://bob:pw@h/1", kw(username="bob", password="pw", db=1)),
        ("rediss://h:6380", kw(port=6380, ssl=True)),
        ("h", kw()),
        ("h:6380", kw(port=6380)),
        ("redis:6379,password=p", kw(host="redis", password="p")),
        ("h:6379,password=p,ssl=true", kw(password="p", ssl=True)),
        ("h,SSL=False,Password=p", kw(password="p")),
        ("h:6379,defaultDatabase=1", kw(db=1)),
        ("h:6379,user=bob,password=a=b#c!", kw(username="bob", password="a=b#c!")),
        ("h:6379,abortConnect=false,connectTimeout=5000,password=p", kw(password="p")),
        ("  h:6379 , password=p ", kw(password="p")),
    ],
)
def test_redis_kwargs_accepts_both_formats(value, expected):
    assert redis_kwargs(value) == expected


@pytest.mark.parametrize("value", ["h:abc,password=hunter2", "h:99999,password=hunter2", "redis://h:abc", "h:1,defaultDatabase=x,password=hunter2", "", ",password=hunter2", "http://h:1"])
def test_redis_kwargs_rejects_bad_values_without_leaking_the_password(value):
    with pytest.raises(ValueError) as exc:
        redis_kwargs(value)
    assert "hunter2" not in str(exc.value) and (not value or value not in str(exc.value))


def test_stackexchange_style_url_reaches_redis(monkeypatch):
    # Regression: the prod compose shares `redis:6379,password=...` between services; the cache used to
    # prefix it with redis:// and hand it to Redis.from_url, which raised and silently fell back to memory.
    built = []

    class RecordingRedis:
        """Stands in for redis.Redis: records how it was built, answers ping."""

        def __init__(self, **kwargs):
            built.append(kwargs)

        def ping(self):
            return True

    monkeypatch.setattr(redis, "Redis", RecordingRedis)
    cache = ResultCache("redis:6379,password=x")
    assert cache.backend == "redis"
    assert built[0]["host"] == "redis" and built[0]["password"] == "x"


class StubRedis:
    """In-memory stand-in for the few Redis commands the cache uses. `down` makes every command raise."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.plain_sets: list[str] = []  # SET without expiry: must stay empty
        self.down = False

    def _check(self):
        if self.down:
            raise redis.ConnectionError("down")

    def ping(self):
        self._check()
        return True

    def get(self, key):
        self._check()
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self._check()
        self.data[key] = value
        self.ttls[key] = ttl

    def set(self, key, value, *args, **kwargs):
        self.plain_sets.append(key)
        self.data[key] = value


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_writes_use_setex_with_the_ttl_and_prefix():
    stub = StubRedis()
    cache = ResultCache(ttl=86400, client=stub)
    cache.set("abc", {"overall_score": 80})
    assert cache.backend == "redis"
    assert stub.ttls == {"ai:score:abc": 86400} and stub.plain_sets == []
    assert cache.get("abc") == {"overall_score": 80}
    assert cache.get("missing") is None


def test_redis_errors_fall_back_to_memory_then_recover_after_30_seconds():
    stub, clock = StubRedis(), Clock()
    cache = ResultCache(client=stub, clock=clock)
    stub.down = True
    cache.set("k", {"v": 1})  # write fails -> memory
    assert cache.backend == "memory"
    assert cache.get("k") == {"v": 1}  # served from memory while Redis is down
    clock.now += 29
    assert cache.backend == "memory"  # not retried yet
    stub.down = False
    clock.now += 2  # 31 s after the failure
    assert cache.backend == "redis"
    cache.set("k2", {"v": 2})
    assert "ai:score:k2" in stub.data


def test_read_error_also_falls_back():
    stub, clock = StubRedis(), Clock()
    cache = ResultCache(client=stub, clock=clock)
    stub.down = True
    assert cache.get("k") is None
    assert cache.backend == "memory"


def test_redis_down_at_start_is_retried():
    stub, clock = StubRedis(), Clock()
    stub.down = True
    cache = ResultCache(client=stub, clock=clock)
    assert cache.backend == "memory"
    stub.down = False
    clock.now += 31
    assert cache.backend == "redis"


def test_memory_entries_expire():
    clock = Clock()
    cache = ResultCache(ttl=10, clock=clock)
    cache.set("k", {"v": 1})
    clock.now += 11
    assert cache.get("k") is None


def test_invalid_redis_url_falls_back_to_memory_and_logs_no_secret(caplog):
    with caplog.at_level("WARNING"):
        cache = ResultCache("h:abc,password=hunter2")
    assert cache.backend == "memory"
    assert "hunter2" not in caplog.text
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
