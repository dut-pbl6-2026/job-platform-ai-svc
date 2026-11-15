"""Chat HTTP tests: SSE stream, session endpoints, ownership, auth, rate limit, error event, and both client readers."""
import json
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from app.chat import engine
from app.chat import intent as I
from app.config import Settings
from app.errors import SearchUnavailable
from app.main import create_app
from tests.test_cache import StubRedis

HIT = {"id": "22222222-2222-2222-2222-222222222222", "title": "Java Developer", "company": "Acme"}
ALICE = {"X-User-Id": "alice"}
BOB = {"X-User-Id": "bob"}
SECRET = "x" * 40


class FakeSearch:
    def __init__(self, hits=None, error=None):
        self.hits, self.error, self.calls = hits if hits is not None else [HIT], error, []

    async def find_jobs(self, found):
        self.calls.append(found)
        if self.error:
            raise self.error
        return list(self.hits)

    async def close(self):
        pass


@pytest.fixture
def search() -> FakeSearch:
    return FakeSearch()


@pytest.fixture
def chat_client(emb, search):
    settings = Settings(jwt_secret="", redis_url="", job_service_url="http://job.test", ocr_concurrency=1)
    with TestClient(create_app(settings, emb, None, search_client=search)) as c:
        yield c


def chat(c, message="tìm việc Java ở Hà Nội", headers=ALICE, **extra):
    return c.post("/api/ai/chat", json={"message": message, **extra}, headers=headers)


def parse_frames(body: str) -> list[tuple[str, str]]:
    out = []
    for block in body.split("\n\n"):
        if block.strip():
            lines = block.split("\n")
            out.append((lines[0][len("event: "):], lines[1][len("data: "):]))
    return out


# --- the two client readers, ported line for line ----------------------------

def web_reader(body: str):
    """Port of job-platform-web chatApi.ts: split on blank lines, consumeSseBlock per block."""
    text, jobs, sid = [], None, None
    buf = body
    for block in buf.split("\n\n"):
        event, datas = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                datas.append(line[5:].strip())
        data = "\n".join(datas)
        if not data:
            continue
        if event in ("token", "delta", "message"):
            parsed = json.loads(data)
            text.append(parsed.get("token") or parsed.get("delta") or parsed.get("content") or data)
        elif event == "jobs":
            jobs = json.loads(data)
        elif event in ("done", "end"):
            sid = json.loads(data).get("sessionId") or sid
    return "".join(text), jobs, sid


def mobile_reader(body: str):
    """Port of job-platform-mobile ApiAiRepository.streamChat (LineSplitter, data: lines, delta/text/response)."""
    text, sid, done = [], None, False
    for line in body.replace("\r\n", "\n").split("\n"):
        trimmed = line.strip()
        if not trimmed or trimmed.startswith(":"):
            continue
        if trimmed.startswith("data:"):
            raw = line[5:] if line.startswith("data:") else trimmed[5:]
            raw = raw.removeprefix(" ")
            if raw.strip() == "[DONE]":
                break
            try:
                decoded = json.loads(raw.strip())
            except ValueError:
                text.append(raw)  # the Dart client yields unparsed data as text
                continue
            if isinstance(decoded, dict):
                delta = decoded.get("delta") or decoded.get("text") or decoded.get("response") or ""
                sid = decoded.get("sessionId") or sid
                done = done or decoded.get("done") is True
                text.append(delta)
            elif isinstance(decoded, str):
                text.append(decoded)
    return "".join(text), sid, done


# --- stream ------------------------------------------------------------------

def test_chat_streams_sse_with_the_event_order_the_clients_expect(chat_client, search):
    r = chat(chat_client)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache" and r.headers["x-accel-buffering"] == "no"
    events = [e for e, _ in parse_frames(r.text)]
    assert events[0] == "token" and events[-2:] == ["jobs", "done"]
    assert set(events) == {"token", "jobs", "done"}
    assert "\r" not in r.text
    assert search.calls[0].keyword is None and search.calls[0].location == "Hà Nội"


def test_web_and_mobile_readers_rebuild_the_same_reply_and_session(chat_client):
    r = chat(chat_client, "tìm việc Java ở Hà Nội")
    web_text, web_jobs, web_sid = web_reader(r.text)
    mob_text, mob_sid, mob_done = mobile_reader(r.text)
    assert web_text == mob_text and web_text.strip()
    assert web_sid == mob_sid and web_sid
    assert mob_done is True
    assert web_jobs == [HIT]
    plain = chat(chat_client, "tìm việc Java ở Hà Nội", stream=False).json()
    assert plain["response"] == web_text  # the JSON path carries the same answer text


def test_both_readers_handle_template_answers_with_newlines_and_vietnamese(chat_client):
    r = chat(chat_client, "Bạn giúp được gì?")
    web_text, web_jobs, _ = web_reader(r.text)
    mob_text, _, _ = mobile_reader(r.text)
    assert "\n" in web_text and web_text == mob_text and web_jobs is None


def test_search_failure_streams_an_error_event_and_a_text_answer(emb):
    settings = Settings(jwt_secret="", redis_url="", ocr_concurrency=1)
    with TestClient(create_app(settings, emb, None, search_client=FakeSearch(error=SearchUnavailable("down")))) as c:
        r = chat(c, "tìm việc Java")
        assert r.status_code == 200  # the stream has started; the failure is an event, not a status
        events = parse_frames(r.text)
        error = [json.loads(d) for e, d in events if e == "error"]
        assert error == [{"code": "search_service_unavailable", "message": engine.SEARCH_FAILED}]
        assert [e for e, _ in events][-1] == "done"
        assert web_reader(r.text)[0] == engine.SEARCH_FAILED
        assert mobile_reader(r.text)[0] == engine.SEARCH_FAILED


def test_unconfigured_search_takes_the_failure_path(emb):
    settings = Settings(jwt_secret="", redis_url="", ocr_concurrency=1)  # search_service_url empty
    with TestClient(create_app(settings, emb, None)) as c:
        r = chat(c, "tìm việc Java")
        assert any(e == "error" for e, _ in parse_frames(r.text))


def test_stream_false_returns_json_with_response_and_session(chat_client):
    r = chat(chat_client, "viết CV thế nào", stream=False)
    body = r.json()
    assert r.headers["content-type"].startswith("application/json")
    assert set(body) == {"response", "sessionId", "jobs"} and body["jobs"] == [] and body["sessionId"]


# --- sessions ----------------------------------------------------------------

def test_session_create_history_and_delete(chat_client):
    created = chat_client.post("/api/ai/session", headers=ALICE)
    assert created.status_code == 201
    sid = created.json()["sessionId"]
    assert chat_client.get(f"/api/ai/session/{sid}/history", headers=ALICE).json() == []

    done_sid = web_reader(chat(chat_client, "viết CV thế nào", headers=ALICE, sessionId=sid).text)[2]
    assert done_sid == sid  # the id the client sent is kept while it is valid

    history = chat_client.get(f"/api/ai/session/{sid}/history", headers=ALICE).json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "viết CV thế nào" and history[0]["createdAt"]
    assert history[1]["content"] == engine.TEXT[I.CV]

    detail = chat_client.get(f"/api/ai/session/{sid}", headers=ALICE)
    assert detail.status_code == 200 and detail.json()["sessionId"] == sid and len(detail.json()["messages"]) == 2

    assert chat_client.delete(f"/api/ai/session/{sid}", headers=ALICE).status_code == 204
    assert chat_client.get(f"/api/ai/session/{sid}/history", headers=ALICE).json() == []
    assert chat_client.get(f"/api/ai/session/{sid}", headers=ALICE).status_code == 404
    assert chat_client.delete(f"/api/ai/session/{sid}", headers=ALICE).status_code == 204  # idempotent


def test_sessions_are_owned_by_the_user_who_created_them(chat_client):
    sid = chat_client.post("/api/ai/session", headers=ALICE).json()["sessionId"]
    chat(chat_client, "viết CV", headers=ALICE, sessionId=sid)

    assert chat_client.get(f"/api/ai/session/{sid}/history", headers=BOB).json() == []
    assert chat_client.get(f"/api/ai/session/{sid}", headers=BOB).status_code == 404
    assert chat_client.delete(f"/api/ai/session/{sid}", headers=BOB).status_code == 204  # no effect for Bob
    assert len(chat_client.get(f"/api/ai/session/{sid}/history", headers=ALICE).json()) == 2

    # Bob sending Alice's id gets a fresh session; Alice's history is untouched
    bob_sid = web_reader(chat(chat_client, "viết CV", headers=BOB, sessionId=sid).text)[2]
    assert bob_sid and bob_sid != sid
    assert len(chat_client.get(f"/api/ai/session/{sid}/history", headers=ALICE).json()) == 2
    assert len(chat_client.get(f"/api/ai/session/{bob_sid}/history", headers=BOB).json()) == 2


def test_unknown_session_id_on_chat_starts_a_new_session(chat_client):
    unknown = "77777777-7777-7777-7777-777777777777"
    sid = web_reader(chat(chat_client, "viết CV", sessionId=unknown).text)[2]
    assert sid and sid != unknown
    assert chat_client.get(f"/api/ai/session/{unknown}/history", headers=ALICE).json() == []


def test_non_uuid_session_ids_are_no_ops_not_errors(chat_client):
    assert chat_client.get("/api/ai/session/not-a-uuid/history", headers=ALICE).json() == []
    assert chat_client.delete("/api/ai/session/not-a-uuid", headers=ALICE).status_code == 204
    assert web_reader(chat(chat_client, "viết CV", sessionId="ai:chat:x").text)[2]


def test_session_store_is_keyed_by_the_prefixed_canonical_id_in_redis(emb):
    stub = StubRedis()
    settings = Settings(jwt_secret="", redis_url="", ocr_concurrency=1)
    with TestClient(create_app(settings, emb, None, redis_client=stub, search_client=FakeSearch())) as c:
        sid = web_reader(chat(c, "viết CV").text)[2]
    assert f"ai:chat:{sid}" in stub.data and stub.ttls[f"ai:chat:{sid}"] == 24 * 3600
    assert not any(k.startswith("ai:score:") for k in stub.data)


# --- auth, validation and limits ---------------------------------------------

def test_chat_without_identity_is_401(chat_client):
    r = chat_client.post("/api/ai/chat", json={"message": "hi"})
    assert r.status_code == 401 and r.json()["code"] == "unauthorized"


def test_session_endpoints_without_identity_are_401(chat_client):
    assert chat_client.post("/api/ai/session").status_code == 401
    assert chat_client.get("/api/ai/session/22222222-2222-2222-2222-222222222222/history").status_code == 401
    assert chat_client.delete("/api/ai/session/22222222-2222-2222-2222-222222222222").status_code == 401


@pytest.fixture
def jwt_client(emb, search):
    settings = Settings(jwt_secret=SECRET, redis_url="", ocr_concurrency=1)
    with TestClient(create_app(settings, emb, None, search_client=search)) as c:
        yield c


def _token(sub="alice", secret=SECRET, expires_in=300):
    now = int(time.time())
    claims = {"sub": sub, "iss": "job-platform", "aud": "job-platform", "iat": now, "exp": now + expires_in}
    return jwt.encode(claims, secret, algorithm="HS256")


def test_jwt_mode_needs_a_bearer_token(jwt_client):
    assert jwt_client.post("/api/ai/chat", json={"message": "hi"}).status_code == 401
    assert jwt_client.post("/api/ai/chat", json={"message": "hi"}, headers={"X-User-Id": "alice"}).status_code == 401


def test_jwt_mode_rejects_a_bad_signature_and_an_expired_token(jwt_client):
    bad = {"Authorization": f"Bearer {_token(secret='y' * 40)}"}
    expired = {"Authorization": f"Bearer {_token(expires_in=-60)}"}
    assert jwt_client.post("/api/ai/chat", json={"message": "hi"}, headers=bad).status_code == 401
    assert jwt_client.post("/api/ai/chat", json={"message": "hi"}, headers=expired).status_code == 401


def test_jwt_subject_is_the_session_owner(jwt_client):
    alice = {"Authorization": f"Bearer {_token('alice')}"}
    bob = {"Authorization": f"Bearer {_token('bob')}"}
    sid = web_reader(chat(jwt_client, "viết CV", headers=alice).text)[2]
    assert len(jwt_client.get(f"/api/ai/session/{sid}/history", headers=alice).json()) == 2
    assert jwt_client.get(f"/api/ai/session/{sid}/history", headers=bob).json() == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": "   "},
        {"message": 12},
        {"message": "x" * 2001},
        {"message": "hi", "stream": "yes"},
    ],
)
def test_invalid_bodies_are_400_before_any_stream(chat_client, payload):
    r = chat_client.post("/api/ai/chat", json=payload, headers=ALICE)
    assert r.status_code == 400 and r.headers["content-type"].startswith("application/problem+json")


def test_non_json_body_is_400(chat_client):
    r = chat_client.post("/api/ai/chat", content=b"not json", headers={**ALICE, "Content-Type": "application/json"})
    assert r.status_code == 400


def test_chat_is_rate_limited_before_the_stream_starts(emb, search):
    settings = Settings(jwt_secret="", redis_url="", ocr_concurrency=1, chat_rate_per_hour=2)
    with TestClient(create_app(settings, emb, None, search_client=search)) as c:
        assert chat(c, "viết CV").status_code == 200
        assert chat(c, "viết CV").status_code == 200
        r = chat(c, "viết CV")
        assert r.status_code == 429 and r.json()["code"] == "rate_limited"
        assert chat(c, "viết CV", headers=BOB).status_code == 200  # the bucket is per user


def test_session_endpoints_are_not_rate_limited(emb, search):
    settings = Settings(jwt_secret="", redis_url="", ocr_concurrency=1, chat_rate_per_hour=1)
    with TestClient(create_app(settings, emb, None, search_client=search)) as c:
        for _ in range(5):
            assert c.post("/api/ai/session", headers=ALICE).status_code == 201


def test_chat_has_its_own_limiter_so_the_score_bucket_is_untouched(chat_client):
    assert chat_client.app.state.chat_limiter is not chat_client.app.state.limiter
