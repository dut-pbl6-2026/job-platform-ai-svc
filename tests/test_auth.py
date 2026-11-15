"""Auth policy (unit): startup validation matrix and the per-request identity rule. No network, no model."""
import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.auth import AuthConfigError, current_user, validate_auth
from app.config import Settings

SECRET = "s" * 32
DEV = {"dev_auth_enabled": True, "environment": "Development"}


def request(headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw, "query_string": b""})


# --- startup validation -----------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"jwt_secret": SECRET},  # production: JWT only
        {"jwt_secret": "x" * 64, **DEV},  # a valid secret wins even in dev
        {"jwt_secret": "", **DEV},  # explicit local dev: no secret, Development, flag on
        {"jwt_secret": "", **DEV, "environment": "development"},  # .NET names are case-insensitive
    ],
)
def test_valid_identity_configurations_start(kwargs):
    validate_auth(Settings(**kwargs))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"jwt_secret": ""}, "JWT_SECRET"),  # missing secret, no dev flag
        ({"jwt_secret": "", "dev_auth_enabled": True}, "only allowed"),  # the flag outside Development is a misconfiguration
        ({"jwt_secret": "", "environment": "Development"}, "JWT_SECRET"),  # Development alone is not enough
        ({"jwt_secret": "s" * 31}, "at least 32"),  # one character short
        ({"jwt_secret": "s" * 31, **DEV}, "at least 32"),  # a short secret never falls back to dev
        ({"jwt_secret": "", "dev_auth_enabled": True, "environment": "Production"}, "only allowed"),
        ({"jwt_secret": SECRET, "dev_auth_enabled": True, "environment": "Production"}, "only allowed"),
    ],
)
def test_unverifiable_configurations_refuse_to_start(kwargs, message):
    with pytest.raises(AuthConfigError, match=message) as info:
        validate_auth(Settings(**kwargs))
    assert SECRET not in str(info.value)  # errors never echo the secret


def test_default_environment_is_production():
    assert Settings(jwt_secret="").environment == "Production"


def test_environment_is_read_from_the_shared_dotnet_variables(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("DOTNET_ENVIRONMENT", "Development")
    assert Settings(_env_file=None).environment == "Development"
    monkeypatch.setenv("DOTNET_ENVIRONMENT", "Production")
    monkeypatch.setenv("ASPNETCORE_ENVIRONMENT", "Development")
    assert Settings(_env_file=None).environment == "Production"  # DOTNET_ENVIRONMENT wins over ASPNETCORE_*


# --- per-request identity ---------------------------------------------------

def token(**claims) -> str:
    base = {"sub": "user-1", "iss": "job-platform", "aud": "job-platform", "exp": 4102444800}
    return jwt.encode({**base, **claims}, SECRET, algorithm="HS256")


def test_jwt_mode_ignores_x_user_id_entirely():
    settings = Settings(jwt_secret=SECRET)
    with pytest.raises(HTTPException) as info:
        current_user(request({"X-User-Id": "victim"}), settings)
    assert info.value.status_code == 401


def test_jwt_mode_reads_the_subject_from_a_valid_token():
    settings = Settings(jwt_secret=SECRET)
    assert current_user(request({"Authorization": f"Bearer {token()}"}), settings) == "user-1"


@pytest.mark.parametrize("header", [None, "Basic abc", "Bearer not-a-jwt", f"Bearer {token(exp=1)}", f"Bearer {token(aud='other')}"])
def test_jwt_mode_rejects_missing_forged_expired_and_foreign_tokens(header):
    settings = Settings(jwt_secret=SECRET)
    headers = {"Authorization": header} if header else {"X-User-Id": "victim"}
    with pytest.raises(HTTPException) as info:
        current_user(request(headers), settings)
    assert info.value.status_code == 401


def test_dev_header_is_trusted_only_in_explicit_development_mode():
    assert current_user(request({"X-User-Id": "dev-user"}), Settings(**DEV)) == "dev-user"
    for settings in (Settings(), Settings(dev_auth_enabled=True), Settings(environment="Development")):
        with pytest.raises(HTTPException) as info:
            current_user(request({"X-User-Id": "attacker"}), settings)
        assert info.value.status_code == 401
