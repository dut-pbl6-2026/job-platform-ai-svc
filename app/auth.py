"""Caller identity. Production: RS256 JWT issued by auth-svc, verified with the public keys from its JWKS.

Fail closed: the service refuses to start unless JWT_METADATA_ADDRESS (auth-svc discovery URL) is set, or the
explicit dev mode is on (DEV_AUTH_ENABLED=true AND environment Development). Only in dev mode is the X-User-Id
header trusted, and only when JWT_METADATA_ADDRESS is unset. `validate_auth` is called once at app creation;
`current_user` re-checks the same rule.

Tokens: algorithm RS256 only, `kid` required and looked up in the published key set, `iss`/`aud`/`exp`/`sub` checked.
Keys are cached per discovery address; PyJWKClient refreshes at most every JWKS_CACHE_SECONDS and on an unknown kid.
A failed key load is a 401 (fail closed) and is retried on the next request.

Scoring service identity: app-svc scores applications in the background with AI_SCORING_SERVICE_TOKEN as a bearer.
That token is accepted only by POST /api/ai/score-resume (`scoring_caller`); every other route refuses it (`current_user`).
The scoring route also accepts a user identity, but only with role Recruiter or Admin: a candidate (role User or no
role) gets 403 `forbidden`, because a match score is recruiter/admin-only.
"""
import hashlib
import hmac
import logging
import threading
from dataclasses import dataclass

import httpx
import jwt
from fastapi import HTTPException, Request

from app.config import Settings

JWKS_CACHE_SECONDS = 900  # bounded refresh of the published key set
JWKS_HTTP_TIMEOUT_SECONDS = 5.0
MIN_SCORING_TOKEN_LENGTH = 32  # enforced in Production only
MIN_INTERNAL_TOKEN_LENGTH = 32  # AI_INTERNAL_TOKEN_FOR_SEARCH, enforced in Production only
SCORING_SUBJECT = "app-svc-scoring"
DEVELOPMENT = "development"
FORBIDDEN_CODE = "forbidden"
# Roles that may obtain a match score. A candidate (role User) must never call the scoring route.
SCORING_ROLES = frozenset({"recruiter", "admin"})
# auth-svc issues the short JWT claim "role"; ASP.NET's ClaimTypes.Role is the ws/2008/06 URI. Accept both.
ROLE_CLAIMS = (
    "role",
    "http://schemas.microsoft.com/ws/2008/06/identity/claims/role",
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/role",
)
_BEARER = "bearer "
_LOG = logging.getLogger("ai.auth")


class _JwksSource:
    """Signing keys for one auth-svc discovery address. Built lazily; a failed build is not cached."""

    def __init__(self, metadata_address: str) -> None:
        self._metadata_address = metadata_address
        self._client: jwt.PyJWKClient | None = None
        self._lock = threading.Lock()

    def _client_or_build(self) -> jwt.PyJWKClient:
        with self._lock:
            if self._client is None:
                response = httpx.get(self._metadata_address, timeout=JWKS_HTTP_TIMEOUT_SECONDS)
                response.raise_for_status()
                jwks_uri = response.json()["jwks_uri"]
                self._client = jwt.PyJWKClient(
                    jwks_uri, cache_jwk_set=True, lifespan=JWKS_CACHE_SECONDS, timeout=JWKS_HTTP_TIMEOUT_SECONDS,
                )
            return self._client

    def signing_key(self, token: str):
        return self._client_or_build().get_signing_key_from_jwt(token).key


_sources: dict[str, _JwksSource] = {}
_sources_lock = threading.Lock()


def _source_for(metadata_address: str) -> _JwksSource:
    with _sources_lock:
        source = _sources.get(metadata_address)
        if source is None:
            source = _sources[metadata_address] = _JwksSource(metadata_address)
        return source


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
    internal = settings.ai_internal_token_for_search
    if not _is_development(settings):
        # Fail closed: search-svc's embedding calls need their own key; there is no shared fallback.
        if not internal:
            raise AuthConfigError("AI_INTERNAL_TOKEN_FOR_SEARCH is required outside Development (search-svc bearer for /api/internal/*).")
        if len(internal) < MIN_INTERNAL_TOKEN_LENGTH:
            raise AuthConfigError(f"AI_INTERNAL_TOKEN_FOR_SEARCH must be at least {MIN_INTERNAL_TOKEN_LENGTH} characters outside Development.")
    if internal and internal == settings.ai_scoring_service_token:
        # Otherwise app-svc's scoring key would also open the internal embedding endpoints.
        raise AuthConfigError("AI_INTERNAL_TOKEN_FOR_SEARCH must differ from AI_SCORING_SERVICE_TOKEN.")
    if settings.jwt_metadata_address:
        if not settings.jwt_metadata_address.startswith(("http://", "https://")):
            raise AuthConfigError("JWT_METADATA_ADDRESS must be an http(s) URL (auth-svc discovery document).")
        return
    if not dev_auth_allowed(settings):
        raise AuthConfigError(
            "JWT_METADATA_ADDRESS (auth-svc discovery URL) is required. "
            "For local development only: DEV_AUTH_ENABLED=true with environment Development."
        )


def _claim_role(claims: dict) -> str | None:
    """Role from a verified JWT: the short `role` claim or an ASP.NET ClaimTypes.Role URI. First non-empty wins."""
    for name in ROLE_CLAIMS:
        value = claims.get(name)
        if isinstance(value, list):
            value = next((v for v in value if v), None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _user_identity(request: Request, settings: Settings) -> tuple[str, str | None]:
    """Verified (user id, role) for a user identity. Raises 401 exactly like the previous `current_user`."""
    if is_scoring_token(request, settings):
        # the scoring token is a credential for one route only; it is never a user identity anywhere else
        raise HTTPException(401, detail={"code": "unauthorized", "title": "Invalid token"})
    if settings.jwt_metadata_address:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith(_BEARER):
            raise HTTPException(401, detail={"code": "unauthorized", "title": "Missing bearer token"})
        token = header[7:].strip()
        try:
            if not jwt.get_unverified_header(token).get("kid"):
                raise jwt.InvalidTokenError("token has no kid")
            key = _source_for(settings.jwt_metadata_address).signing_key(token)
            claims = jwt.decode(
                token, key, algorithms=["RS256"], issuer=settings.jwt_issuer, audience=settings.jwt_audience,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except Exception as exc:  # every failure to verify the token, including a JWKS outage, is a 401
            _LOG.info("Token rejected: %s", type(exc).__name__)
            raise HTTPException(401, detail={"code": "unauthorized", "title": "Invalid token"}) from exc
        uid = claims.get("sub") or claims.get("nameid") or claims.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier")
        if not uid:
            raise HTTPException(401, detail={"code": "unauthorized", "title": "Token has no subject"})
        return str(uid), _claim_role(claims)
    if not dev_auth_allowed(settings):
        # unreachable after validate_auth at startup; kept so that a bypassed validation still fails closed
        raise HTTPException(401, detail={"code": "unauthorized", "title": "Authentication is not configured"})
    uid = request.headers.get("x-user-id", "").strip()
    if not uid:
        raise HTTPException(401, detail={"code": "unauthorized", "title": "Missing X-User-Id (dev auth)"})
    role = request.headers.get("x-user-role", "").strip() or None
    return uid, role


def current_user(request: Request, settings: Settings) -> str:
    """User identity for every route except POST /api/ai/score-resume, which uses `scoring_caller`."""
    return _user_identity(request, settings)[0]


def scoring_caller(request: Request, settings: Settings) -> Caller:
    """The score-resume route only: the scoring token (app-svc) or a Recruiter/Admin user identity.

    A candidate is refused with 403 `forbidden`: the match score is recruiter/admin-only. The check is
    keyed on how the caller authenticated, so the scoring token keeps working unchanged and a user JWT
    cannot inherit its exemptions.
    """
    if is_scoring_token(request, settings):
        return Caller(SCORING_SUBJECT, scoring_service=True)

    uid, role = _user_identity(request, settings)
    if role is None or role.strip().lower() not in SCORING_ROLES:
        raise HTTPException(
            403,
            detail={
                "code": FORBIDDEN_CODE,
                "title": "Forbidden",
                "detail": "Resume scoring is available to Recruiter and Admin accounts only.",
            },
        )
    return Caller(uid, scoring_service=False)
