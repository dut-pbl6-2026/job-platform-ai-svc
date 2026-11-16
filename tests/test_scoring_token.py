"""AI_SCORING_SERVICE_TOKEN (app-svc background scoring): token policy (unit) and the scoring caller's behaviour (API).

No network, no model. The token is accepted only by POST /api/ai/score-resume, exempt from the per-user cap only,
and still subject to the global concurrency semaphore.
"""
import hmac

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import auth
from app.auth import (
    SCORING_SUBJECT,
    AuthConfigError,
    Caller,
    bearer_matches,
    current_user,
    scoring_caller,
    validate_auth,
)
from app.config import Settings
from app.main import create_app
from tests.conftest import CV_TEXT, SEARCH_KEY, make_text_pdf
from tests.jwks_support import METADATA_URL, sign
from tests.test_api import JOB_ID, FakeJobs
from tests.test_chat_api import FakeSearch
from tests.test_security_api import PROTECTED, fill

SCORING = "t" * 40  # distinct from SECRET
DEV = {"dev_auth_enabled": True, "environment": "Development"}


def request(headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw, "query_string": b""})


def user_token(sub: str = "alice", role: str | None = "Recruiter", role_claim: str = "role") -> str:
    claims = {"sub": sub, "iss": "job-platform", "aud": "job-platform", "exp": 4102444800}
    if role is not None:
        claims[role_claim] = role
    return sign(claims)


# --- unit: startup policy ---------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"jwt_metadata_address": METADATA_URL, "ai_scoring_service_token": "t" * 32},  # exactly the minimum, Production
        {"jwt_metadata_address": METADATA_URL, "ai_scoring_service_token": "t" * 8, **DEV},  # short is tolerated outside Production
        {"jwt_metadata_address": METADATA_URL},  # unset
        {"jwt_metadata_address": METADATA_URL, "ai_scoring_service_token": ""},  # empty is the same as unset
    ],
)
def test_scoring_token_policy_allows_valid_configurations(kwargs):
    validate_auth(Settings(**{"ai_internal_token_for_search": SEARCH_KEY, **kwargs}))


def test_short_scoring_token_refuses_to_start_in_production_even_with_a_configured_jwks():
    # the JWT branch returns early, so the token check must run before it
    settings = Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token="t" * 31)
    with pytest.raises(AuthConfigError, match="AI_SCORING_SERVICE_TOKEN") as info:
        validate_auth(settings)
    assert "t" * 31 not in str(info.value)  # errors never echo the value


def test_short_scoring_token_refuses_to_start_in_production_without_a_jwks_address_too():
    with pytest.raises(AuthConfigError, match="AI_SCORING_SERVICE_TOKEN"):
        validate_auth(Settings(jwt_metadata_address="", ai_scoring_service_token="short"))


# --- unit: token matching ---------------------------------------------------

def test_correct_token_matches():
    assert bearer_matches(f"Bearer {SCORING}", SCORING)
    assert bearer_matches(f"bearer {SCORING}", SCORING)  # scheme is case-insensitive


@pytest.mark.parametrize("header", ["", "Bearer ", "Bearer wrong", f"Bearer {SCORING}x", f"Basic {SCORING}", SCORING])
def test_wrong_empty_or_malformed_headers_never_match(header):
    assert not bearer_matches(header, SCORING)


def test_an_empty_expected_token_never_matches_anything():
    assert not bearer_matches("Bearer ", "")
    assert not bearer_matches("Bearer anything", "")


def test_comparison_is_done_on_sha256_digests_with_compare_digest(monkeypatch):
    real_compare = hmac.compare_digest
    seen = []

    def spy(a, b):
        seen.append((a, b))
        return real_compare(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)
    assert bearer_matches(f"Bearer {'x' * 200}", SCORING) is False
    assert bearer_matches(f"Bearer {SCORING}", SCORING) is True
    for a, b in seen:
        assert len(a) == len(b) == 32  # SHA-256 digests: equal length whatever the presented token's length
        assert SCORING.encode() not in (a, b)  # the raw token is never compared


# --- unit: identity per route ----------------------------------------------

def test_scoring_caller_is_the_scoring_subject_and_bypasses_nothing_else():
    settings = Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token=SCORING)
    assert scoring_caller(request({"Authorization": f"Bearer {SCORING}"}), settings) == Caller(SCORING_SUBJECT, scoring_service=True)


def test_scoring_route_still_accepts_a_user_jwt_as_a_normal_user():
    settings = Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token=SCORING)
    caller = scoring_caller(request({"Authorization": f"Bearer {user_token('alice')}"}), settings)
    assert caller == Caller("alice", scoring_service=False)


# --- unit: score-resume is recruiter/admin-only ------------------------------

@pytest.mark.parametrize("role", ["Recruiter", "Admin", "recruiter", "ADMIN"])
def test_scoring_route_accepts_recruiter_and_admin_user_jwts(role):
    settings = Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token=SCORING)
    caller = scoring_caller(request({"Authorization": f"Bearer {user_token('alice', role=role)}"}), settings)
    assert caller == Caller("alice", scoring_service=False)


def test_scoring_route_accepts_the_aspnet_claim_types_role_uri():
    settings = Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token=SCORING)
    token = user_token("alice", role="Recruiter",
                       role_claim="http://schemas.microsoft.com/ws/2008/06/identity/claims/role")
    caller = scoring_caller(request({"Authorization": f"Bearer {token}"}), settings)
    assert caller == Caller("alice", scoring_service=False)


@pytest.mark.parametrize("role", ["User", "Candidate", ""])
def test_scoring_route_refuses_a_candidate_with_403_forbidden(role):
    settings = Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token=SCORING)
    headers = {"Authorization": f"Bearer {user_token('candidate', role=role)}"}
    with pytest.raises(HTTPException) as info:
        scoring_caller(request(headers), settings)
    assert info.value.status_code == 403
    assert info.value.detail["code"] == auth.FORBIDDEN_CODE == "forbidden"


def test_scoring_route_refuses_a_user_jwt_with_no_role_claim():
    settings = Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token=SCORING)
    headers = {"Authorization": f"Bearer {user_token('alice', role=None)}"}
    with pytest.raises(HTTPException) as info:
        scoring_caller(request(headers), settings)
    assert info.value.status_code == 403
    assert info.value.detail["code"] == "forbidden"


def test_dev_auth_scoring_route_reads_x_user_role():
    settings = Settings(ai_scoring_service_token=SCORING, **DEV)
    assert scoring_caller(request({"X-User-Id": "alice", "X-User-Role": "Recruiter"}), settings) == Caller("alice")
    assert scoring_caller(request({"X-User-Id": "bob", "X-User-Role": "Admin"}), settings) == Caller("bob")
    with pytest.raises(HTTPException) as info:
        scoring_caller(request({"X-User-Id": "candidate", "X-User-Role": "User"}), settings)
    assert info.value.status_code == 403
    assert info.value.detail["code"] == "forbidden"


def test_dev_auth_scoring_route_refuses_a_missing_role_header():
    settings = Settings(ai_scoring_service_token=SCORING, **DEV)
    with pytest.raises(HTTPException) as info:
        scoring_caller(request({"X-User-Id": "candidate"}), settings)
    assert info.value.status_code == 403


def test_a_user_whose_subject_is_the_scoring_name_is_not_the_scoring_service():
    settings = Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token=SCORING)
    caller = scoring_caller(request({"Authorization": f"Bearer {user_token(SCORING_SUBJECT)}"}), settings)
    assert caller.scoring_service is False


@pytest.mark.parametrize("settings", [
    Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token=SCORING),
    Settings(ai_scoring_service_token=SCORING, **DEV),
])
def test_scoring_token_is_refused_as_a_user_identity_on_every_other_route(settings):
    # JWT mode, and dev mode where X-User-Id would otherwise be trusted
    headers = {"Authorization": f"Bearer {SCORING}", "X-User-Id": "victim"}
    with pytest.raises(HTTPException) as info:
        current_user(request(headers), settings)
    assert info.value.status_code == 401


def test_wrong_token_on_the_scoring_route_is_401_in_jwt_mode():
    settings = Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token=SCORING)
    with pytest.raises(HTTPException) as info:
        scoring_caller(request({"Authorization": "Bearer nope"}), settings)
    assert info.value.status_code == 401


def test_unset_or_empty_token_never_authenticates_anything():
    for settings in (Settings(**DEV), Settings(jwt_metadata_address=METADATA_URL), Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token="")):
        with pytest.raises(HTTPException) as info:
            scoring_caller(request({"Authorization": "Bearer "}), settings)
        assert info.value.status_code == 401


# --- API: integration -------------------------------------------------------

def make_client(settings, emb):
    return TestClient(create_app(settings, emb, FakeJobs(), search_client=FakeSearch()))


@pytest.fixture
def scoring_settings() -> Settings:
    return Settings(jwt_metadata_address=METADATA_URL, ai_scoring_service_token=SCORING, ai_internal_token_for_search=SEARCH_KEY, redis_url="", job_service_url="http://job.test", ocr_concurrency=1)


@pytest.fixture
def client(scoring_settings, emb):
    with make_client(scoring_settings, emb) as c:
        yield c


def scoring_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {SCORING}"}


def score(client, headers, data=None):
    return client.post("/api/ai/score-resume", files={"cv_file": ("cv.pdf", data or make_text_pdf(CV_TEXT))},
                       data={"job_id": JOB_ID}, headers=headers)


def test_the_scoring_caller_scores_more_than_ten_times_a_minute(client):
    data = make_text_pdf(CV_TEXT)
    codes = [score(client, scoring_headers(), data).status_code for _ in range(12)]
    assert codes == [200] * 12


def test_a_score_from_the_scoring_caller_is_keyed_to_its_subject_and_not_reopenable_by_the_token(client):
    # Known limitation (reported, not fixed): records are keyed by the caller subject. The token is refused on GET,
    # so app-svc reopens nothing; it keeps the response body from POST instead.
    r = score(client, scoring_headers())
    assert r.status_code == 200 and r.json()["score_id"]
    assert client.get(f"/api/ai/score-resume/{r.json()['score_id']}", headers=scoring_headers()).status_code == 401


def test_a_real_user_is_capped_at_ten_per_minute(client):
    data = make_text_pdf(CV_TEXT)
    headers = {"Authorization": f"Bearer {user_token('alice')}"}
    codes = [score(client, headers, data).status_code for _ in range(11)]
    assert codes == [200] * 10 + [429]


def test_a_candidate_gets_403_forbidden_from_the_score_route(client):
    response = score(client, {"Authorization": f"Bearer {user_token('candidate', role='User')}"})

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"
    assert response.headers["content-type"].startswith("application/problem+json")


def test_a_recruiter_user_jwt_can_score(client):
    response = score(client, {"Authorization": f"Bearer {user_token('recruiter', role='Recruiter')}"})

    assert response.status_code == 200
    assert "score_id" in response.json()


def test_the_scoring_caller_does_not_consume_a_real_users_budget(client):
    data = make_text_pdf(CV_TEXT)
    alice = {"Authorization": f"Bearer {user_token('alice')}"}
    for _ in range(10):
        assert score(client, alice, data).status_code == 200
    assert score(client, alice, data).status_code == 429
    assert score(client, scoring_headers(), data).status_code == 200


@pytest.mark.parametrize("method,path", sorted(PROTECTED - {("POST", "/api/ai/score-resume")}))
def test_the_token_gets_401_on_every_other_protected_route_in_jwt_mode(client, method, path):
    r = client.request(method, fill(path), headers=scoring_headers())  # no body: auth answers first
    assert r.status_code == 401, (method, path, r.text)
    assert r.json()["code"] == "unauthorized"


@pytest.mark.parametrize("method,path", sorted(PROTECTED - {("POST", "/api/ai/score-resume")}))
def test_the_token_gets_401_on_every_other_protected_route_in_dev_mode(emb, method, path):
    settings = Settings(ai_scoring_service_token=SCORING, redis_url="", job_service_url="http://job.test", **DEV)
    with make_client(settings, emb) as c:
        r = c.request(method, fill(path), headers={**scoring_headers(), "X-User-Id": "victim"})
    assert r.status_code == 401, (method, path, r.text)


def test_a_wrong_token_is_401_on_the_scoring_route(client):
    assert score(client, {"Authorization": "Bearer wrong"}).status_code == 401


def test_global_saturation_returns_busy_to_the_scoring_caller(scoring_settings, emb):
    scoring_settings.busy_timeout_seconds = 0.05
    with make_client(scoring_settings, emb) as c:
        sem = c.app.state.sem
        c.portal.call(sem.acquire)  # hold the only scoring slot, on the app's own event loop
        try:
            r = score(c, scoring_headers())
        finally:
            c.portal.call(sem.release)
    assert r.status_code == 503
    assert r.json()["code"] == "busy"
