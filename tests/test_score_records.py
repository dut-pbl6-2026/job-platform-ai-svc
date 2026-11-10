"""Score records: POST returns a stable score_id, GET/DELETE give the owner access to it."""
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app import service
from app.main import create_app
from tests.conftest import CV_TEXT, make_text_pdf
from tests.test_api import JOB_ID, NURSE_CV, PY_JOB, FakeJobs, post
from tests.test_cache import StubRedis

REC = "ai:scorerec:"
SHARED = "ai:score:"


@pytest.fixture
def stub():
    return StubRedis()


@pytest.fixture
def jobs():
    return FakeJobs()


@pytest.fixture
def client(settings, emb, stub, jobs):
    with TestClient(create_app(settings, emb, jobs, redis_client=stub)) as c:
        yield c


def as_user(user):
    return {"X-User-Id": user}


def records(stub):
    return {k: v for k, v in stub.data.items() if k.startswith(REC)}


def test_post_returns_a_stable_score_id_per_user(client):
    cv = make_text_pdf(CV_TEXT)
    a1 = post(client, cv).json()
    a2 = post(client, cv).json()
    b = post(client, cv, headers=as_user("u2")).json()
    assert uuid.UUID(a1["score_id"]) and a1["score_id"] == a2["score_id"]
    assert b["score_id"] != a1["score_id"]
    other_cv = post(client, make_text_pdf(NURSE_CV)).json()
    assert other_cv["score_id"] != a1["score_id"]


def test_score_id_is_a_function_of_user_and_key():
    assert service.score_id_for("u1", "k") == service.score_id_for("u1", "k")
    assert service.score_id_for("u1", "k") != service.score_id_for("u2", "k")
    assert service.score_id_for("u1", "k") != service.score_id_for("u1", "k2")


def test_record_is_stored_with_owner_ttl_and_prefix(client, stub, settings):
    body = post(client, make_text_pdf(CV_TEXT)).json()
    key = REC + body["score_id"]
    assert key in stub.data and stub.ttls[key] == settings.score_retention_seconds == 30 * 24 * 3600
    rec = json.loads(stub.data[key])
    assert rec["owner"] == "u1" and rec["job_id"] == JOB_ID and rec["cache_key"].startswith(service.scorer.RULESET)
    assert rec["created_at"] and rec["result"]["overall_score"] == body["overall_score"]
    assert stub.plain_sets == []


def test_score_id_stays_out_of_the_shared_cache_and_owner_out_of_the_response(client, stub):
    body = post(client, make_text_pdf(CV_TEXT)).json()
    shared = [v for k, v in stub.data.items() if k.startswith(SHARED)]
    assert len(shared) == 1 and "score_id" not in json.loads(shared[0])
    assert "owner" not in body and "score_id" not in json.loads(stub.data[REC + body["score_id"]])["result"]


def test_existing_response_fields_are_unchanged(client, settings, emb):
    cv = make_text_pdf(CV_TEXT)
    direct = service.score_cv_bytes(cv, PY_JOB, settings, emb)  # what POST returned before score_id existed
    assert set(post(client, cv).json()) == set(direct) | {"score_id"}


def test_failure_to_store_the_record_does_not_fail_the_post(settings, emb, jobs):
    stub = StubRedis()
    stub.down = True
    with TestClient(create_app(settings, emb, jobs, redis_client=stub)) as c:
        r = post(c, make_text_pdf(CV_TEXT))
    assert r.status_code == 200 and r.json()["score_id"]


def test_post_without_identity_is_rejected_before_touching_redis(client, stub):
    r = client.post("/api/ai/score-resume", files={"cv_file": ("a.pdf", make_text_pdf(CV_TEXT))}, data={"job_id": JOB_ID})
    assert r.status_code == 401 and stub.commands == [] and stub.data == {}


URL = "/api/ai/score-resume/"


def test_owner_gets_the_stored_result_without_rescoring(client, jobs):
    body = post(client, make_text_pdf(CV_TEXT)).json()
    calls = jobs.calls
    r = client.get(URL + body["score_id"], headers=as_user("u1"))
    assert r.status_code == 200 and r.json() == body
    assert jobs.calls == calls  # no job-svc call, so it works after the job is deleted


def test_get_reflects_the_cached_flag_the_post_returned(client):
    cv = make_text_pdf(CV_TEXT)
    first = post(client, cv).json()
    assert client.get(URL + first["score_id"], headers=as_user("u1")).json()["meta"]["cached"] is False
    second = post(client, cv).json()
    assert second["meta"]["cached"] is True
    assert client.get(URL + second["score_id"], headers=as_user("u1")).json()["meta"]["cached"] is True


def test_another_user_gets_404_and_the_record_survives(client, stub):
    body = post(client, make_text_pdf(CV_TEXT)).json()
    r = client.get(URL + body["score_id"], headers=as_user("u2"))
    assert r.status_code == 404 and r.json()["code"] == "score_not_found"
    assert r.headers["content-type"].startswith("application/problem+json")
    assert REC + body["score_id"] in stub.data
    assert client.get(URL + body["score_id"], headers=as_user("u1")).status_code == 200


def test_unknown_and_expired_ids_are_the_same_404(client, stub):
    unknown = client.get(URL + str(uuid.uuid4()), headers=as_user("u1"))
    body = post(client, make_text_pdf(CV_TEXT)).json()
    stub.expire_all()
    expired = client.get(URL + body["score_id"], headers=as_user("u1"))
    assert unknown.status_code == expired.status_code == 404
    assert unknown.json()["code"] == expired.json()["code"] == "score_not_found"


@pytest.mark.parametrize("bad", ["abc", "*", "ai:score:x", "x" * 300, "%2A", "..%2Fx"])
def test_malformed_ids_never_reach_redis(client, stub, bad):
    post(client, make_text_pdf(CV_TEXT))
    before = list(stub.commands)
    r = client.get(URL + bad, headers=as_user("u1"))
    assert r.status_code in (400, 404) and r.status_code != 200
    if "%2F" not in bad:
        assert r.status_code == 400 and r.json()["code"] == "bad_request"
    assert stub.commands == before


def test_get_requires_identity(client, stub):
    body = post(client, make_text_pdf(CV_TEXT)).json()
    before = list(stub.commands)
    r = client.get(URL + body["score_id"])
    assert r.status_code == 401 and r.json()["code"] == "unauthorized" and stub.commands == before


def test_get_works_from_memory_while_redis_is_down(settings, emb, jobs):
    stub = StubRedis()
    stub.down = True
    with TestClient(create_app(settings, emb, jobs, redis_client=stub)) as c:
        body = post(c, make_text_pdf(CV_TEXT)).json()
        assert c.get(URL + body["score_id"], headers=as_user("u1")).json() == body


def test_get_is_not_rate_limited(settings, emb, jobs):
    s = settings.model_copy(update={"rate_limit_per_min": 1})
    with TestClient(create_app(s, emb, jobs, redis_client=StubRedis())) as c:
        body = post(c, make_text_pdf(CV_TEXT)).json()
        assert post(c, make_text_pdf(CV_TEXT)).status_code == 429
        assert all(c.get(URL + body["score_id"], headers=as_user("u1")).status_code == 200 for _ in range(5))


def test_delete_removes_the_record_and_the_shared_cache_entry(client, stub):
    cv = make_text_pdf(CV_TEXT)
    body = post(client, cv).json()
    assert any(k.startswith(SHARED) for k in stub.data)
    r = client.delete(URL + body["score_id"], headers=as_user("u1"))
    assert r.status_code == 204 and r.content == b""
    assert stub.data == {}  # both keys are gone from Redis
    assert client.get(URL + body["score_id"], headers=as_user("u1")).status_code == 404
    assert client.delete(URL + body["score_id"], headers=as_user("u1")).status_code == 404
    assert post(client, cv).json()["meta"]["cached"] is False  # nothing derived from the file was left behind


def test_another_user_cannot_delete_and_nothing_is_removed(client, stub):
    body = post(client, make_text_pdf(CV_TEXT)).json()
    before = dict(stub.data)
    r = client.delete(URL + body["score_id"], headers=as_user("u2"))
    assert r.status_code == 404 and r.json()["code"] == "score_not_found"
    assert stub.data == before
    assert client.get(URL + body["score_id"], headers=as_user("u1")).status_code == 200


def test_deleting_one_users_score_leaves_the_other_users_record(client, stub):
    cv = make_text_pdf(CV_TEXT)
    a = post(client, cv).json()
    b = post(client, cv, headers=as_user("u2")).json()
    assert b["meta"]["cached"] is True and a["score_id"] != b["score_id"]  # u2 hit the shared cache
    assert client.delete(URL + a["score_id"], headers=as_user("u1")).status_code == 204
    assert client.get(URL + b["score_id"], headers=as_user("u2")).status_code == 200


@pytest.mark.parametrize("bad", ["abc", "*", "ai:score:x"])
def test_delete_malformed_id_never_reaches_redis(client, stub, bad):
    post(client, make_text_pdf(CV_TEXT))
    before = list(stub.commands)
    r = client.delete(URL + bad, headers=as_user("u1"))
    assert r.status_code == 400 and r.json()["code"] == "bad_request" and stub.commands == before


def test_delete_requires_identity(client, stub):
    body = post(client, make_text_pdf(CV_TEXT)).json()
    before = dict(stub.data)
    r = client.delete(URL + body["score_id"])
    assert r.status_code == 401 and stub.data == before


def test_delete_during_a_redis_outage_clears_memory(settings, emb, jobs):
    stub = StubRedis()
    stub.down = True
    with TestClient(create_app(settings, emb, jobs, redis_client=stub)) as c:
        body = post(c, make_text_pdf(CV_TEXT)).json()
        assert c.delete(URL + body["score_id"], headers=as_user("u1")).status_code == 204
        assert c.get(URL + body["score_id"], headers=as_user("u1")).status_code == 404
