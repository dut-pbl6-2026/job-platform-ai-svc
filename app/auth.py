"""Caller identity. Production: HS256 JWT issued by auth-svc (same secret/issuer/audience as the .NET services).
Dev (no JWT_SECRET): trust X-User-Id like DevAuthMiddleware. Returns a user id string or raises 401."""
import jwt
from fastapi import HTTPException, Request

from app.config import Settings


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
    uid = request.headers.get("x-user-id", "").strip()
    if not uid:
        raise HTTPException(401, detail={"code": "unauthorized", "title": "Missing X-User-Id (dev auth)"})
    return uid
