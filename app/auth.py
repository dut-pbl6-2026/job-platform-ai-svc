"""Caller identity. Production: HS256 JWT issued by auth-svc (same secret/issuer/audience as the .NET services).

Fail closed: the service refuses to start unless JWT_SECRET (>= 32 chars) is set, or the explicit dev mode is on
(DEV_AUTH_ENABLED=true AND environment Development). Only in dev mode is the X-User-Id header trusted, and only
when JWT_SECRET is unset. `validate_auth` is called once at app creation; `current_user` re-checks the same rule.

Scoring service identity: app-svc scores applications in the background with AI_SCORING_SERVICE_TOKEN as a bearer.
That token is accepted only by POST /api/ai/score-resume (`scoring_caller`); every other route refuses it (`current_user`).
"""
import hashlib
import hmac
from dataclasses import dataclass

import jwt
from fastapi import HTTPException, Request

from app.config import Settings

MIN_JWT_SECRET_LENGTH = 32
MIN_SCORING_TOKEN_LENGTH = 32  # enforced in Production only
SCORING_SUBJECT = "app-svc-scoring"
DEVELOPMENT = "development"
_BEARER = "bearer "


class AuthConfigError(RuntimeError):
    """Raised at startup when identity cannot be verified. Never includes the secret value."""


@dataclass(frozen=True)
class Caller:
    subject: str
    # True only when the caller presented AI_SCORING_SERVICE_TOKEN. Keyed on how the caller authenticated,
    # never on the subject string, so a user JWT cannot inherit the scoring exemptions.
    scoring_service: bool = False


def _is_development(settings: Settings) -> bool:
    return settings.environment.strip().lower() == DEVELOPMENT  # .NET IsDevelopment() is case-insensitive


def dev_auth_allowed(settings: Settings) -> bool:
    return settings.dev_auth_enabled and _is_development(settings)


def bearer_matches(header: str, expected: str) -> bool:
    """Constant-time check of an `Authorization: Bearer` header against a configured token. Empty never matches."""
    if not expected or not header.lower().startswith(_BEARER):
        return False
    token = header[len(_BEARER):].strip()
    if not token:
        return False
    # Hash both sides so the compared digests always have the same length (no length leak).
    return hmac.compare_digest(hashlib.sha256(token.encode()).digest(), hashlib.sha256(expected.encode()).digest())


def is_scoring_token(request: Request, settings: Settings) -> bool:
    return bearer_matches(request.headers.get("authorization", ""), settings.ai_scoring_service_token)


def validate_auth(settings: Settings) -> None:
    """Refuse to start with an unverifiable identity configuration."""
    if settings.dev_auth_enabled and not _is_development(settings):
        raise AuthConfigError("DEV_AUTH_ENABLED=true is only allowed when the environment is Development.")
    if settings.ai_scoring_service_token and not _is_development(settings) and len(settings.ai_scoring_service_token) < MIN_SCORING_TOKEN_LENGTH:
        raise AuthConfigError(f"AI_SCORING_SERVICE_TOKEN must be at least {MIN_SCORING_TOKEN_LENGTH} characters in Production.")
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
    """User identity for every route except POST /api/ai/score-resume, which uses `scoring_caller`."""
    if is_scoring_token(request, settings):
        # the scoring token is a credential for one route only; it is never a user identity anywhere else
        raise HTTPException(401, detail={"code": "unauthorized", "title": "Invalid token"})
    if settings.jwt_secret:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith(_BEARER):
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


def scoring_caller(request: Request, settings: Settings) -> Caller:
    """The score-resume route only: the scoring token (app-svc) or the normal user identity."""
    if is_scoring_token(request, settings):
        return Caller(SCORING_SUBJECT, scoring_service=True)
    return Caller(current_user(request, settings))
