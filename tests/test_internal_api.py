"""Internal embed endpoints (search-svc): token gate, batch shape, validation, and refusal to serve placeholder vectors."""
import jwt
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.auth import AuthConfigError
from app.config import Settings
from app.embed import HashingEmbedder
from app.main import create_app

TOKEN = "t" * 40  # AI_INTERNAL_TOKEN_FOR_SEARCH for these tests
SHARED_TOKEN = "shared-internal-token-0123456789abcdef"  # the old INTERNAL_SERVICE_TOKEN value: must not open this API
SCORING_TOKEN = "s" * 40  # AI_SCORING_SERVICE_TOKEN (app-svc): a different caller
JWT_SECRET = "j" * 40


class FixedEmbedder:
    """Stands in for the ONNX model: a real (non-hashing) embedder with a known dimension."""

    name = "fixed-model"
    dim = 3

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts], dtype=np.float32)


def _client(emb, token: str = TOKEN) -> TestClient:
    settings = Settings(jwt_secret="", dev_auth_enabled=True, environment="Development", redis_url="", ocr_concurrency=1, ai_internal_token_for_search=token)
    return TestClient(create_app(settings, emb, None))


@pytest.fixture
def client():
    with _client(FixedEmbedder()) as c:
        yield c


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_embed_returns_one_vector_per_text_with_model_and_dimension(client):
    r = client.post("/api/internal/embed", json={"texts": ["ab", "xyz"]}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body == {"model": "fixed-model", "dimension": 3, "vectors": [[2.0, 1.0, 0.0], [3.0, 1.0, 0.0]]}


def test_info_reports_model_and_dimension(client):
    r = client.get("/api/internal/embed/info", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"model": "fixed-model", "dimension": 3}


def test_bearer_scheme_is_case_insensitive(client):
    r = client.get("/api/internal/embed/info", headers={"Authorization": f"bearer {TOKEN}"})
    assert r.status_code == 200


@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer wrong-token-value-that-is-long-enough-xxxxxx"},
    {"Authorization": "Basic dG9rZW4="},
    {"Authorization": "Bearer "},
    {"Authorization": TOKEN},
])
def test_wrong_or_missing_credentials_are_401(client, headers):
    assert client.post("/api/internal/embed", json={"texts": ["a"]}, headers=headers).status_code == 401
    assert client.get("/api/internal/embed/info", headers=headers).status_code == 401


def test_unconfigured_token_fails_closed_even_with_a_bearer_header():
    with _client(FixedEmbedder(), token="") as c:
        r = c.post("/api/internal/embed", json={"texts": ["a"]}, headers={"Authorization": "Bearer anything"})
        assert r.status_code == 401
        assert r.json()["code"] == "unauthorized"


@pytest.mark.parametrize("payload", [
    {"texts": []},
    {"texts": [""]},
    {"texts": ["a"] * 257},
    {},
])
def test_invalid_batches_are_rejected_with_422(client, payload):
    assert client.post("/api/internal/embed", json=payload, headers=AUTH).status_code == 422


def test_placeholder_hashing_embedder_is_refused_with_503():
    with _client(HashingEmbedder()) as c:
        r = c.post("/api/internal/embed", json={"texts": ["a"]}, headers=AUTH)
        assert r.status_code == 503
        assert r.json()["code"] == "embedder_unavailable"
        assert c.get("/api/internal/embed/info", headers=AUTH).status_code == 503


def _production_client(emb, **keys) -> TestClient:
    # Production (the default environment), JWT mode: the internal routes must still take only the search key.
    settings = Settings(_env_file=None, jwt_secret=JWT_SECRET, redis_url="", ocr_concurrency=1, **keys)
    return TestClient(create_app(settings, emb, None))


def test_shared_token_is_rejected_and_only_the_search_key_is_accepted(monkeypatch):
    # Built from the environment as in production: the shared token is set, but only the caller's own key counts.
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", SHARED_TOKEN)
    monkeypatch.setenv("AI_INTERNAL_TOKEN_FOR_SEARCH", TOKEN)
    settings = Settings(_env_file=None, jwt_secret=JWT_SECRET, redis_url="", ocr_concurrency=1)
    assert settings.ai_internal_token_for_search == TOKEN
    with TestClient(create_app(settings, FixedEmbedder(), None)) as c:
        shared = {"Authorization": f"Bearer {SHARED_TOKEN}"}
        assert c.get("/api/internal/embed/info", headers=shared).status_code == 401
        assert c.post("/api/internal/embed", json={"texts": ["a"]}, headers=shared).status_code == 401
        own = {"Authorization": f"Bearer {TOKEN}"}
        assert c.get("/api/internal/embed/info", headers=own).status_code == 200
        assert c.post("/api/internal/embed", json={"texts": ["a"]}, headers=own).status_code == 200


def test_another_callers_key_is_rejected_on_both_internal_routes():
    with _production_client(FixedEmbedder(), ai_internal_token_for_search=TOKEN, ai_scoring_service_token=SCORING_TOKEN) as c:
        scoring = {"Authorization": f"Bearer {SCORING_TOKEN}"}
        assert c.post("/api/internal/embed", json={"texts": ["a"]}, headers=scoring).status_code == 401
        assert c.get("/api/internal/embed/info", headers=scoring).status_code == 401
        user = {"Authorization": f"Bearer {jwt.encode({'sub': 'alice', 'iss': 'job-platform', 'aud': 'job-platform', 'exp': 4102444800}, JWT_SECRET, algorithm='HS256')}"}
        assert c.get("/api/internal/embed/info", headers=user).status_code == 401
        assert c.get("/api/internal/embed/info", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_production_without_the_search_key_refuses_to_start():
    with pytest.raises(AuthConfigError, match="AI_INTERNAL_TOKEN_FOR_SEARCH"):
        _production_client(FixedEmbedder())
