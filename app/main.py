"""FastAPI app: POST /api/ai/score-resume, GET /health, GET /api/ai/health."""
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app import service
from app.auth import current_user
from app.cache import ResultCache
from app.config import Settings, get_settings
from app.embed import Embedder, HashingEmbedder, OnnxEmbedder
from app.errors import ApiError, BadRequest, Busy, FileTooLarge, RateLimited
from app.job_client import JobClient
from app.parsing.ocr import ocr_available
from app.ratelimit import RateLimiter

log = logging.getLogger("ai")
_LEVELS = {"TRACE": "DEBUG", "DEBUG": "DEBUG", "INFORMATION": "INFO", "INFO": "INFO", "WARNING": "WARNING", "WARN": "WARNING", "ERROR": "ERROR", "CRITICAL": "CRITICAL"}


def log_level(name: str) -> str:
    """The shared .env uses .NET level names (Information, ...); accept both conventions."""
    return _LEVELS.get(name.strip().upper(), "INFO")


def build_embedder(settings: Settings) -> Embedder:
    try:
        emb = OnnxEmbedder(settings.embedding_model, settings.embedding_cache_dir)
        emb.embed(["warm-up"])  # first inference pays the session start-up cost, not the first user
        return emb
    except Exception:
        log.exception("Embedding model failed to load; falling back to hashing embedder (degraded quality)")
        return HashingEmbedder()


def create_app(settings: Settings | None = None, embedder: Embedder | None = None, job_client: JobClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=log_level(settings.log_level))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.embedder = embedder or await run_in_threadpool(build_embedder, settings)
        app.state.jobs = job_client or JobClient(settings.job_service_url, settings.job_http_timeout_seconds, settings.job_cache_ttl_seconds)
        app.state.cache = ResultCache(settings.redis_url, settings.cache_ttl_seconds)
        app.state.limiter = RateLimiter(settings.rate_limit_per_min)
        app.state.sem = asyncio.Semaphore(settings.ocr_concurrency)
        yield
        await app.state.jobs.close()

    app = FastAPI(title="job-platform-ai-svc", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError):
        return JSONResponse(
            {"type": f"https://job-platform/errors/{exc.code}", "title": exc.title, "status": exc.status, "detail": exc.detail, "code": exc.code},
            status_code=exc.status, media_type="application/problem+json",
        )

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        d = exc.detail if isinstance(exc.detail, dict) else {"title": str(exc.detail), "code": "error"}
        return JSONResponse({"status": exc.status_code, **d}, status_code=exc.status_code, media_type="application/problem+json")

    def health():
        emb = getattr(app.state, "embedder", None)
        return {"status": "ok", "service": "ai", "models_loaded": emb is not None, "embedder": getattr(emb, "name", None),
                "degraded": isinstance(emb, HashingEmbedder), "ocr_available": ocr_available(), "cache": app.state.cache.backend}

    app.get("/health")(health)
    app.get("/api/ai/health")(health)

    @app.post("/api/ai/score-resume")
    async def score_resume(request: Request, cv_file: UploadFile = File(...), job_id: str = Form(...), lang: str | None = Form(None)):
        user = current_user(request, settings)
        if not app.state.limiter.allow(user):
            raise RateLimited("Rate limit exceeded; try again in a minute.")
        data = await cv_file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise FileTooLarge(f"Maximum upload size is {settings.max_upload_bytes // (1024 * 1024)} MB.")
        if not data:
            raise BadRequest("cv_file is empty.")
        try:
            uuid.UUID(job_id.strip())
        except ValueError:
            raise BadRequest("job_id must be a UUID.") from None
        job = await app.state.jobs.get(job_id.strip())
        lang = lang.strip().lower() if lang and lang.strip().lower() in ("vi", "en") else None
        key = service.cache_key(data, job_id.strip(), str(job.get("updatedAt", "")), lang or "")
        cached = app.state.cache.get(key)
        if cached:
            cached["meta"]["cached"] = True
            return cached
        try:
            await asyncio.wait_for(app.state.sem.acquire(), timeout=settings.busy_timeout_seconds)
        except TimeoutError:
            raise Busy("All scoring workers are busy.") from None
        try:
            result = await run_in_threadpool(service.score_cv_bytes, data, job, settings, app.state.embedder, None, lang)
        finally:
            app.state.sem.release()
        app.state.cache.set(key, result)
        return result

    return app


# run with: uvicorn app.main:create_app --factory --port 6000
