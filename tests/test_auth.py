"""Auth policy (unit): startup validation matrix and the per-request identity rule. No model; JWKS is loopback."""
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.auth import AuthConfigError, current_user, validate_auth
from app.config import Settings
from tests.jwks_support import METADATA_URL, claims, sign, sign_hs256

DEV = {"dev_auth_enabled": True, "environment": "Development"}
INTERNAL = "i" * 40  # AI_INTERNAL_TOKEN_FOR_SEARCH: required outside Development


def request(headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw, "query_string": b""})


# --- startup validation -----------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"jwt_metadata_address": METADATA_URL},  # production: JWT only
        {"jwt_metadata_address": METADATA_URL, **DEV},  # a configured JWKS wins even in dev
        {"jwt_metadata_address": "", **DEV},  # explicit local dev: no discovery address, Development, flag on
        {"jwt_metadata_address": "", **DEV, "environment": "development"},  # .NET names are case-insensitive
    ],
)
def test_valid_identity_configurations_start(kwargs):
    validate_auth(Settings(**{"ai_internal_token_for_search": INTERNAL, **kwargs}))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"jwt_metadata_address": ""}, "JWT_METADATA_ADDRESS"),  # missing address, no dev flag
        ({"jwt_metadata_address": "", "dev_auth_enabled": True}, "only allowed"),  # the flag outside Development is a misconfiguration
        ({"jwt_metadata_address": "", "environment": "Development"}, "JWT_METADATA_ADDRESS"),  # Development alone is not enough
        ({"jwt_metadata_address": "auth-svc:5001/.well-known"}, "http"),  # not a URL
        ({"jwt_metadata_address": "", "dev_auth_enabled": True, "environment": "Production"}, "only allowed"),
        ({"jwt_metadata_address": METADATA_URL, "dev_auth_enabled": True, "environment": "Production"}, "only allowed"),
    ],
)
def test_unverifiable_configurations_refuse_to_start(kwargs, message):
    with pytest.raises(AuthConfigError, match=message):
        validate_auth(Settings(**{"ai_internal_token_for_search": INTERNAL, **kwargs}))


def test_default_environment_is_production():
    assert Settings(jwt_metadata_address="").environment == "Production"


def test_environment_is_read_from_the_shared_dotnet_variables(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("DOTNET_ENVIRONMENT", "Development")
    assert Settings(_env_file=None).environment == "Development"
    monkeypatch.setenv("DOTNET_ENVIRONMENT", "Production")
    monkeypatch.setenv("ASPNETCORE_ENVIRONMENT", "Development")
    assert Settings(_env_file=None).environment == "Production"  # DOTNET_ENVIRONMENT wins over ASPNETCORE_*


# --- per-request identity ---------------------------------------------------

def test_jwt_mode_ignores_x_user_id_entirely():
    settings = Settings(jwt_metadata_address=METADATA_URL)
    with pytest.raises(HTTPException) as info:
        current_user(request({"X-User-Id": "victim"}), settings)
    assert info.value.status_code == 401


def test_jwt_mode_reads_the_subject_from_a_valid_rs256_token():
    settings = Settings(jwt_metadata_address=METADATA_URL)
    assert current_user(request({"Authorization": f"Bearer {sign()}"}), settings) == "user-1"


@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        sign(claims(exp=1)),  # expired
        sign(claims(aud="other")),  # foreign audience
        sign(claims(iss="other")),  # foreign issuer
        sign({k: v for k, v in claims().items() if k != "sub"}),  # no subject
        sign(claims(), kid=None),  # no kid
        sign(claims(), kid="unpublished"),  # kid not in the key set
        sign(claims(), foreign=True),  # signed by a key nobody publishes
        sign_hs256(claims()),  # HS256 is refused, whatever the secret
    ],
)
def test_jwt_mode_rejects_forged_expired_and_foreign_tokens(token):
    settings = Settings(jwt_metadata_address=METADATA_URL)
    with pytest.raises(HTTPException) as info:
        current_user(request({"Authorization": f"Bearer {token}"}), settings)
    assert info.value.status_code == 401


@pytest.mark.parametrize("header", [None, "Basic abc"])
def test_jwt_mode_rejects_a_missing_or_non_bearer_header(header):
    settings = Settings(jwt_metadata_address=METADATA_URL)
    headers = {"Authorization": header} if header else {"X-User-Id": "victim"}
    with pytest.raises(HTTPException) as info:
        current_user(request(headers), settings)
    assert info.value.status_code == 401


def test_unreachable_discovery_fails_closed():
    # Nothing listens on this port: the key set cannot load, so a correctly signed token is still a 401.
    settings = Settings(jwt_metadata_address="http://127.0.0.1:9/.well-known/openid-configuration")
    with pytest.raises(HTTPException) as info:
        current_user(request({"Authorization": f"Bearer {sign()}"}), settings)
    assert info.value.status_code == 401


def test_dev_header_is_trusted_only_in_explicit_development_mode():
    assert current_user(request({"X-User-Id": "dev-user"}), Settings(**DEV)) == "dev-user"
    for settings in (Settings(), Settings(dev_auth_enabled=True), Settings(environment="Development")):
        with pytest.raises(HTTPException) as info:
            current_user(request({"X-User-Id": "attacker"}), settings)
        assert info.value.status_code == 401


# --- AI_INTERNAL_TOKEN_FOR_SEARCH (search-svc's key for /api/internal/*) ----

@pytest.mark.parametrize("kwargs", [{"jwt_metadata_address": METADATA_URL}, {"jwt_metadata_address": "", "environment": "Production"}])
def test_missing_search_key_refuses_to_start_outside_development(kwargs):
    with pytest.raises(AuthConfigError, match="AI_INTERNAL_TOKEN_FOR_SEARCH is required") as info:
        validate_auth(Settings(**kwargs))
    assert "AI_INTERNAL_TOKEN_FOR_SEARCH is required" in str(info.value)


def test_missing_search_key_is_allowed_in_development():
    validate_auth(Settings(jwt_metadata_address="", **DEV))  # internal endpoints then refuse every call (see test_internal_api)


def test_short_search_key_refuses_to_start_outside_development():
    with pytest.raises(AuthConfigError, match="must be at least 32"):
        validate_auth(Settings(jwt_metadata_address=METADATA_URL, ai_internal_token_for_search="i" * 31))


def test_search_key_must_differ_from_the_scoring_key_and_never_echoes_it():
    settings = Settings(jwt_metadata_address=METADATA_URL, ai_internal_token_for_search=INTERNAL, ai_scoring_service_token=INTERNAL)
    with pytest.raises(AuthConfigError, match="must differ") as info:
        validate_auth(settings)
    assert INTERNAL not in str(info.value)
