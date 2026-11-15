"""Caller identity. Production: HS256 JWT issued by auth-svc (same secret/issuer/audience as the .NET services).

Fail closed: the service refuses to start unless JWT_SECRET (>= 32 chars) is set, or the explicit dev mode is on
(DEV_AUTH_ENABLED=true AND environment Development). Only in dev mode is the X-User-Id header trusted, and only
when JWT_SECRET is unset. `validate_auth` is called once at app creation; `current_user` re-checks the same rule.
"""
import jwt
from fastapi import HTTPException, Request

from app.config import Settings

MIN_JWT_SECRET_LENGTH = 32
DEVELOPMENT = "development"


class AuthConfigError(RuntimeError):
    """Raised at startup when identity cannot be verified. Never includes the secret value."""


def _is_development(settings: Settings) -> bool:
    return settings.environment.strip().lower() == DEVELOPMENT  # .NET IsDevelopment() is case-insensitive


def dev_auth_allowed(settings: Settings) -> bool:
    return settings.dev_auth_enabled and _is_development(settings)


def validate_auth(settings: Settings) -> None:
    """Refuse to start with an unverifiable identity configuration."""
    if settings.dev_auth_enabled and not _is_development(settings):
        raise AuthConfigError("DEV_AUTH_ENABLED=true is only allowed when the environment is Development.")
    if settings.jwt_secret:
        if len(settings.jwt_secret) < MIN_JWT_SECRET_LENGTH:
            raise AuthConfigError(f"JWT_SECRET must be at least {MIN_JWT_SECRET_LENGTH} characters.")
        return
    if not dev_auth_allowed(settings):
        raise AuthConfigError(
            f"JWT_SECRET (at least {MIN_JWT_SECRET_LENGTH} characters) is required. "
            "For local development only: DEV_AUTH_ENABLED=true with environment Development."
        )


def current_user(request: Request, settings: Settings) -> str:
    if settings.jwt_secret:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise HTTPException(401, detail={"code": "unauthorized", "title": "Missing bearer token"})
        try:
            claims = jwt.decode(
                header[7:], settings.jwt_secret, algorithms=["HS256"], issuer=settings.jwt_issuer, audience=settings.jwt_audience,
                options={"require": ["exp"]},
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(401, detail={"code": "unauthorized", "title": "Invalid token"}) from exc
        uid = claims.get("sub") or claims.get("nameid") or claims.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier")
        if not uid:
            raise HTTPException(401, detail={"code": "unauthorized", "title": "Token has no subject"})
        return str(uid)
    if not dev_auth_allowed(settings):
        # unreachable after validate_auth at startup; kept so that a bypassed validation still fails closed
        raise HTTPException(401, detail={"code": "unauthorized", "title": "Authentication is not configured"})
    uid = request.headers.get("x-user-id", "").strip()
    if not uid:
        raise HTTPException(401, detail={"code": "unauthorized", "title": "Missing X-User-Id (dev auth)"})
    return uid
