"""Chat unit tests: intent rules, search client mapping and failures, engine answers, SSE framing, session store."""
import json

import httpx
import pytest

from app.cache import ResultCache
from app.chat import engine
from app.chat import intent as I
from app.chat.search import SearchClient
from app.chat.sessions import MAX_MESSAGES, ChatSessions, canonical_id
from app.chat.sse import frame, token_frames
from app.errors import SearchUnavailable
from tests.test_cache import Clock, StubRedis

HN = "Hà Nội"
JOB_ID = "22222222-2222-2222-2222-222222222222"


# --- intent -----------------------------------------------------------------

@pytest.mark.parametrize(
    "message,name",
    [
        ("Xin chào bạn", I.GREETING),
        ("hello", I.GREETING),
        ("tìm việc Java ở Hà Nội", I.JOB_SEARCH),
        ("Tìm việc kế toán tại TP.HCM", I.JOB_SEARCH),
        ("React developer Đà Nẵng", I.JOB_SEARCH),
        ("Python dev", I.JOB_SEARCH),
        ("tìm việc", I.JOB_SEARCH),
        ("Node.js và Docker", I.JOB_SEARCH),
        ("javascript", I.JOB_SEARCH),
        ("Tôi muốn tìm việc viết CV", I.JOB_SEARCH),  # an explicit job word beats the CV topic
        ("Mình muốn viết CV thế nào?", I.CV),
        ("tư vấn CV cho fresher", I.CV),
        ("Chuẩn bị phỏng vấn ra sao", I.INTERVIEW),
        ("Phỏng vấn Java dev thế nào", I.INTERVIEW),  # interview topic beats a bare skill
        ("Bạn giúp được gì?", I.HELP),
        ("cách hack ngân hàng", I.OFF_TOPIC),
        ("thời tiết hôm nay", I.UNKNOWN),
        ("", I.UNKNOWN),
        ("   ", I.UNKNOWN),
    ],
)
def test_intent_names(message, name):
    assert I.classify(message).name == name


@pytest.mark.parametrize(
    "message,keyword,location,skills",
    [
        ("tìm việc Java ở Hà Nội", None, HN, ("Java",)),
        ("Tìm việc kế toán tại TP.HCM", "kế toán", "Hồ Chí Minh", ()),
        ("React developer Đà Nẵng", "developer", "Đà Nẵng", ("React",)),
        ("Senior .NET developer ở hcm", "developer", "Hồ Chí Minh", (".NET",)),
        ("Node.js và Docker", None, None, ("Node.js", "Docker")),
        ("javascript", None, None, ("JavaScript",)),  # "java" must not match inside "javascript"
        ("ban giup minh tim viec flutter o ha noi", None, HN, ("Flutter",)),
    ],
)
def test_parameter_extraction(message, keyword, location, skills):
    found = I.classify(message)
    assert (found.keyword, found.location, found.skills) == (keyword, location, skills)


def test_job_request_without_criteria_is_detected_but_not_searchable():
    found = I.classify("tìm việc")
    assert found.name == I.JOB_SEARCH and not found.has_criteria


def test_describe_label():
    assert engine.describe(I.classify("tìm việc Java ở Hà Nội")) == "Java tại Hà Nội"
    assert engine.describe(I.classify("Python developer")) == "developer Python"
    assert engine.describe(I.classify("kế toán ở Đà Nẵng")) == "kế toán tại Đà Nẵng"


# --- SSE framing -----------------------------------------------------------

def test_token_frames_reassemble_the_exact_text():
    text = "Mình tìm thấy 2 tin:\n- Java  Developer tại Acme\n\nBạn có thể mở từng tin."
    body = "".join(token_frames(text))
    parts = [json.loads(block.split("data: ", 1)[1])["delta"] for block in body.split("\n\n") if block]
    assert "".join(parts) == text
    assert all(parts) and not any(p == "" for p in parts)


def test_frames_are_lf_only_one_data_line_and_json():
    body = frame("token", {"delta": "a\nb\r\n"}) + frame("done", {"sessionId": "x", "done": True})
    assert "\r" not in body
    for block in body.split("\n\n"):
        if not block:
            continue
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ") and len(lines) == 2
        json.loads(lines[1][6:])


# --- search client ---------------------------------------------------------

def search_with(handler, base="http://search.test") -> SearchClient:
    client = SearchClient(base, 1.0)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def test_search_client_builds_the_query_and_maps_items():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        return httpx.Response(200, json={"items": [
            {"id": JOB_ID, "title": "Java Dev", "company_name": "Acme", "description": "secret text"},
            {"title": "no id, skipped"},
            "not a dict",
        ], "total": 3})

    hits = await search_with(handler).find_jobs(I.classify("Java developer ở Hà Nội"))
    assert hits == [{"id": JOB_ID, "title": "Java Dev", "company": "Acme"}]
    assert seen["url"].path == "/api/search/jobs"
    params = dict(seen["url"].params)
    assert params == {"size": "3", "skillOp": "or", "q": "developer", "location": HN, "skills": "Java"}


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(503),
        lambda request: httpx.Response(200, text="<html>not json</html>"),
        lambda request: httpx.Response(200, json={"total": 0}),
        _refuse,
    ],
    ids=["http-503", "not-json", "no-items", "connection-refused"],
)
async def test_search_client_upstream_failures_raise_search_unavailable(handler):
    with pytest.raises(SearchUnavailable):
        await search_with(handler).find_jobs(I.classify("Java ở Hà Nội"))


async def test_unconfigured_search_is_unavailable_without_a_network_call():
    with pytest.raises(SearchUnavailable, match="not configured"):
        await SearchClient("", 1.0).find_jobs(I.classify("Java"))


# --- engine ----------------------------------------------------------------

class FakeSearch:
    def __init__(self, hits=None, error=None):
        self.hits, self.error, self.calls = hits or [], error, []

    async def find_jobs(self, found):
        self.calls.append(found)
        if self.error:
            raise self.error
        return list(self.hits)


HIT = {"id": JOB_ID, "title": "Java Developer", "company": "Acme"}


async def test_engine_lists_hits_with_the_disclaimer():
    reply = await engine.answer("tìm việc Java ở Hà Nội", FakeSearch([HIT]))
    assert reply.jobs == [HIT] and reply.error is None
    assert "Java Developer tại Acme" in reply.text and engine.DISCLAIMER in reply.text


async def test_engine_without_hits_says_so():
    reply = await engine.answer("tìm việc Java ở Hà Nội", FakeSearch([]))
    assert reply.jobs == [] and "chưa có tin nào" in reply.text


async def test_engine_skips_search_when_there_are_no_criteria():
    search = FakeSearch([HIT])
    reply = await engine.answer("tìm việc", search)
    assert search.calls == [] and reply.text == engine.NO_CRITERIA


async def test_engine_search_failure_gives_a_text_answer_and_an_error():
    reply = await engine.answer("tìm việc Java", FakeSearch(error=SearchUnavailable("down")))
    assert reply.jobs == [] and reply.text == engine.SEARCH_FAILED
    assert reply.error == {"code": "search_service_unavailable", "message": engine.SEARCH_FAILED}


async def test_engine_template_answers_do_not_search():
    search = FakeSearch([HIT])
    for message in ["Xin chào", "viết CV thế nào", "phỏng vấn thế nào", "cách hack ngân hàng", "xyz"]:
        assert (await engine.answer(message, search)).error is None
    assert search.calls == []


# --- session store ---------------------------------------------------------

def sessions_with(clock=None, ttl=86400, stub=None) -> ChatSessions:
    return ChatSessions(ResultCache("", ttl=ttl, client=stub, clock=clock or Clock()))


def test_sessions_are_private_to_their_owner():
    store = sessions_with()
    sid = "33333333-3333-3333-3333-333333333333"
    store.save("alice", sid, [{"role": "user", "content": "hi"}])
    assert store.load("alice", sid)["messages"][0]["content"] == "hi"
    assert store.load("bob", sid) is None
    store.delete("bob", sid)  # a stranger's delete changes nothing
    assert store.load("alice", sid) is not None
    store.delete("alice", sid)
    assert store.load("alice", sid) is None


def test_sessions_keep_only_the_last_twenty_messages():
    store = sessions_with()
    sid = "44444444-4444-4444-4444-444444444444"
    messages = [{"id": str(i), "role": "user", "content": f"m{i}"} for i in range(MAX_MESSAGES + 5)]
    record = store.save("alice", sid, messages)
    assert len(record["messages"]) == MAX_MESSAGES
    assert record["messages"][0]["content"] == "m5"


def test_sessions_expire_after_the_ttl_and_are_refreshed_by_writes():
    clock = Clock()
    store = sessions_with(clock=clock, ttl=60)
    sid = "55555555-5555-5555-5555-555555555555"
    store.save("alice", sid, [])
    clock.now += 50
    store.save("alice", sid, [{"role": "user", "content": "still here"}])  # a write refreshes the 60 s
    clock.now += 50
    assert store.load("alice", sid) is not None
    clock.now += 61
    assert store.load("alice", sid) is None


def test_sessions_use_the_chat_prefix_and_ttl_in_redis():
    stub = StubRedis()
    store = ChatSessions(ResultCache("", ttl=86400, prefix="ai:chat:", client=stub))
    sid = "66666666-6666-6666-6666-666666666666"
    store.save("alice", sid, [])
    assert stub.ttls == {f"ai:chat:{sid}": 86400} and stub.plain_sets == []


def test_canonical_id_rejects_anything_but_a_uuid():
    assert canonical_id("ABCDEF00-0000-4000-8000-000000000000") == "abcdef00-0000-4000-8000-000000000000"
    for bad in ["", "ai:chat:x", "../x", None, 12, "1234"]:
        assert canonical_id(bad) is None
