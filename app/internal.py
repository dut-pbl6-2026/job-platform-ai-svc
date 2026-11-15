"""Service-to-service endpoints for search-svc: the local embedding model as a service (PBL6-56).

Not routed by the gateway (it only forwards /api/ai/*). Bearer INTERNAL_SERVICE_TOKEN, fail closed:
an unset token refuses every call. Texts are never logged.
"""
import logging
from typing import Annotated

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.auth import bearer_matches
from app.config import Settings
from app.embed import Embedder, HashingEmbedder
from app.errors import EmbedderUnavailable, Unauthorized

log = logging.getLogger("ai.internal")
EMBED_MAX_TEXTS = 256


class EmbedRequest(BaseModel):
    texts: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1, max_length=EMBED_MAX_TEXTS)


def require_internal(request: Request, settings: Settings) -> None:
    if not settings.internal_service_token:
        log.error("INTERNAL_SERVICE_TOKEN is not configured; refusing the internal call.")
        raise Unauthorized("Internal endpoints are disabled until INTERNAL_SERVICE_TOKEN is set.")
    if not bearer_matches(request.headers.get("authorization", ""), settings.internal_service_token):
        raise Unauthorized("A valid internal bearer token is required.")


def _model_or_fail(request: Request) -> Embedder:
    emb = request.app.state.embedder
    if isinstance(emb, HashingEmbedder):
        # Hashing vectors would be indexed as if they were real embeddings: refuse instead.
        raise EmbedderUnavailable("The embedding model is not loaded.")
    return emb


def internal_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/internal", tags=["internal"], include_in_schema=False)

    @router.get("/embed/info")
    async def embed_info(request: Request):
        require_internal(request, settings)
        emb = _model_or_fail(request)
        return {"model": emb.name, "dimension": emb.dim}

    @router.post("/embed")
    async def embed(request: Request, body: EmbedRequest):
        require_internal(request, settings)
        emb = _model_or_fail(request)
        vectors = await run_in_threadpool(emb.embed, body.texts)
        return {"model": emb.name, "dimension": emb.dim, "vectors": vectors.tolist()}

    return router
