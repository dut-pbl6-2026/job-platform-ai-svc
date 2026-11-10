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
