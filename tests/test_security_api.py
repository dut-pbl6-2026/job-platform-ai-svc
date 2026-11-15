"""Integration tests (FastAPI TestClient, no network, no model): auth on every route, body caps, startup, health."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import main
from app.auth import AuthConfigError
from app.config import Settings
from app.embed import HashingEmbedder
from app.main import create_app
from tests.conftest import CV_TEXT, SEARCH_KEY, make_text_pdf
from tests.jwks_support import METADATA_URL, sign
from tests.test_api import FakeJobs
from tests.test_chat_api import FakeSearch

UUID = "77777777-7777-7777-7777-777777777777"
PROTECTED = {
    ("POST", "/api/ai/score-resume"),
    ("GET", "/api/ai/score-resume/{score_id}"),
    ("DELETE", "/api/ai/score-resume/{score_id}"),
    ("POST", "/api/ai/session"),
    ("GET", "/api/ai/session/{session_id}/history"),
    ("GET", "/api/ai/session/{session_id}"),
    ("DELETE", "/api/ai/session/{session_id}"),
    ("POST", "/api/ai/chat"),
}
PUBLIC = {("GET", "/health"), ("GET", "/api/ai/health")}
INTERNAL = {("POST", "/api/internal/embed"), ("GET", "/api/internal/embed/info")}  # bearer AI_INTERNAL_TOKEN_FOR_SEARCH, tested in test_internal_api.py


def route_table(app) -> set[tuple[str, str]]:
    """Every (method, path) the app serves. FastAPI 0.162 wraps include_router() in _IncludedRouter, so unwrap it."""
    out = set()
    stack = list(app.routes)
    while stack:
        route = stack.pop()
        if hasattr(route, "original_router"):
            stack.extend(route.original_router.routes)
            continue
        methods = getattr(route, "methods", None)
        if not methods or route.path.startswith(("/docs", "/redoc", "/openapi")):
            continue
        out |= {(m, route.path) for m in methods - {"HEAD", "OPTIONS"}}
    return out


def fill(path: str) -> str:
    return path.replace("{score_id}", UUID).replace("{session_id}", UUID)


@pytest.fixture
def jwt_settings() -> Settings:
    return Settings(jwt_metadata_address=METADATA_URL, ai_internal_token_for_search=SEARCH_KEY, redis_url="", job_service_url="http://job.test", ocr_concurrency=1)


@pytest.fixture
def jwt_client(jwt_settings, emb):
    with TestClient(create_app(jwt_settings, emb, FakeJobs(), search_client=FakeSearch())) as c:
        yield c


def bearer(sub: str = "alice") -> dict[str, str]:

    token = sign({"sub": sub, "iss": "job-platform", "aud": "job-platform", "exp": 4102444800})
    return {"Authorization": f"Bearer {token}"}


def test_every_route_is_classified_as_protected_public_or_internal(jwt_client):
    # a new route must be added to one of the sets above, so it cannot ship without an auth decision
    assert route_table(jwt_client.app) == PROTECTED | PUBLIC | INTERNAL


@pytest.mark.parametrize("method,path", sorted(PROTECTED))
def test_protected_routes_refuse_requests_without_a_token(jwt_client, method, path):
    r = jwt_client.request(method, fill(path))  # no body on purpose: auth must answer before body validation
    assert r.status_code == 401, (method, path, r.text)
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"] == "unauthorized"


@pytest.mark.parametrize("method,path", sorted(PROTECTED))
def test_protected_routes_ignore_x_user_id_in_jwt_mode(jwt_client, method, path):
    r = jwt_client.request(method, fill(path), headers={"X-User-Id": "victim"})
    assert r.status_code == 401, (method, path, r.text)


def test_a_valid_token_reaches_the_route(jwt_client):
    r = jwt_client.post("/api/ai/session", headers=bearer("alice"))
    assert r.status_code == 201
    assert jwt_client.get(f"/api/ai/session/{r.json()['sessionId']}/history", headers=bearer("bob")).json() == []


def test_a_forged_token_is_refused_on_chat(jwt_client):
    r = jwt_client.post("/api/ai/chat", json={"message": "tìm việc Java"}, headers={"Authorization": "Bearer abc.def.ghi"})
    assert r.status_code == 401


# --- body caps --------------------------------------------------------------

def test_declared_oversized_upload_is_413_before_anything_is_read(jwt_client, jwt_settings):
    limit = jwt_settings.max_upload_bytes + jwt_settings.max_body_slack_bytes
    r = jwt_client.post("/api/ai/score-resume", content=b"x" * (limit + 1), headers={**bearer(), "content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413
    assert r.json()["code"] == "file_too_large"


def test_chunked_oversized_body_is_413_without_content_length(jwt_client, jwt_settings):
    limit = jwt_settings.max_upload_bytes + jwt_settings.max_body_slack_bytes

    def chunks():
        for _ in range(limit // (64 * 1024) + 2):
            yield b"a" * (64 * 1024)

    r = jwt_client.post("/api/ai/chat", content=chunks(), headers={**bearer(), "content-type": "application/json"})
    assert r.status_code == 413
    assert r.json()["code"] == "file_too_large" and r.headers["content-type"].startswith("application/problem+json")


def test_upload_just_over_the_file_limit_is_413_from_the_route(jwt_client, jwt_settings):
    # inside the body slack, so the route's own check answers with the same code
    data = b"%PDF" + b"0" * jwt_settings.max_upload_bytes
    r = jwt_client.post("/api/ai/score-resume", files={"cv_file": ("cv.pdf", data)}, data={"job_id": "11111111-1111-1111-1111-111111111111"}, headers=bearer())
    assert r.status_code == 413 and r.json()["code"] == "file_too_large"


def test_normal_upload_still_scores(jwt_client):
    r = jwt_client.post("/api/ai/score-resume", files={"cv_file": ("cv.pdf", make_text_pdf(CV_TEXT))},
                        data={"job_id": "11111111-1111-1111-1111-111111111111"}, headers=bearer())
    assert r.status_code == 200 and r.json()["meta"]["parser"] == "text"


# --- startup and health -----------------------------------------------------

def test_app_refuses_to_build_without_a_verifiable_identity(emb):
    with pytest.raises(AuthConfigError):
        create_app(Settings(jwt_metadata_address="", redis_url=""), emb, FakeJobs())


class _ActiveModel:
    """Stands in for a loaded ONNX model: a real name, not the hashing fallback."""

    name = "paraphrase-multilingual-MiniLM-L12-v2"
    dim = 384

    def embed(self, texts):
        return np.zeros((len(texts), self.dim), dtype=np.float32)


@pytest.mark.parametrize("path", ["/health", "/api/ai/health"])
def test_health_reports_the_active_embedder(jwt_settings, path):
    with TestClient(create_app(jwt_settings, _ActiveModel(), FakeJobs())) as c:
        body = c.get(path).json()
    assert body["embedder"] == _ActiveModel.name and body["degraded"] is False and body["models_loaded"] is True


@pytest.mark.parametrize("path", ["/health", "/api/ai/health"])
def test_health_flags_the_hashing_fallback_as_degraded(jwt_settings, path):
    with TestClient(create_app(jwt_settings, HashingEmbedder(), FakeJobs())) as c:
        body = c.get(path).json()
    assert body["embedder"] == "hashing-256" and body["degraded"] is True


def test_startup_fails_when_the_model_cannot_load(jwt_settings, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("model files missing")

    monkeypatch.setattr(main, "OnnxEmbedder", broken)
    with pytest.raises(RuntimeError, match="model files missing"), TestClient(create_app(jwt_settings, None, FakeJobs())):
        pass


def test_startup_with_the_hashing_opt_in_is_degraded_and_says_so(jwt_settings, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("model files missing")

    monkeypatch.setattr(main, "OnnxEmbedder", broken)
    settings = Settings(**{**jwt_settings.model_dump(), "embedder_fallback": "hashing"})
    with TestClient(create_app(settings, None, FakeJobs())) as c:
        body = c.get("/api/ai/health").json()
    assert body["embedder"] == "hashing-256" and body["degraded"] is True


def test_embedder_fallback_only_accepts_known_values():
    with pytest.raises(ValidationError):
        Settings(embedder_fallback="random")
