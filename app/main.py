"""FastAPI app: POST /api/ai/score-resume, GET /health, GET /api/ai/health."""
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app import service
from app.auth import current_user
from app.cache import ResultCache, connect
from app.chat import engine
from app.chat.search import SearchClient
from app.chat.sessions import MAX_MESSAGES, ChatSessions, canonical_id, now_iso
from app.chat.sse import frame, token_frames
from app.config import Settings, get_settings
from app.embed import Embedder, HashingEmbedder, OnnxEmbedder
from app.errors import ApiError, BadRequest, Busy, FileTooLarge, RateLimited, ScoreNotFound, SessionNotFound
from app.job_client import JobClient
from app.parsing.ocr import ocr_available
from app.ratelimit import RateLimiter
from app.scoring import scorer

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


CHAT_MAX_CHARS = 2000


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    job_client: JobClient | None = None,
    redis_client=None,
    search_client: SearchClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=log_level(settings.log_level))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.embedder = embedder or await run_in_threadpool(build_embedder, settings)
        app.state.jobs = job_client or JobClient(settings.job_service_url, settings.job_http_timeout_seconds, settings.job_cache_ttl_seconds)
        client = redis_client if redis_client is not None else await run_in_threadpool(connect, settings.redis_url)
        # one connection, two keyspaces: the 24 h cache shared by file, and the per-user score records
        app.state.cache = await run_in_threadpool(ResultCache, "", settings.cache_ttl_seconds, client=client)
        app.state.records = await run_in_threadpool(ResultCache, "", settings.score_retention_seconds, prefix="ai:scorerec:", client=client)
        app.state.chat = ChatSessions(await run_in_threadpool(ResultCache, "", settings.chat_session_ttl_seconds, 2048, prefix="ai:chat:", client=client))
        app.state.search = search_client or SearchClient(settings.search_service_url, settings.search_http_timeout_seconds)
        app.state.chat_limiter = RateLimiter(settings.chat_rate_per_hour, window_seconds=3600)
        app.state.limiter = RateLimiter(settings.rate_limit_per_min)
        app.state.sem = asyncio.Semaphore(settings.ocr_concurrency)
        yield
        await app.state.jobs.close()
        await app.state.search.close()

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

    async def _remember(user: str, score_id: str, key: str, job_id: str, result: dict) -> dict:
        """Store the response for its owner and return it with its score_id (the shared cache never holds the id)."""
        record = {"owner": user, "cache_key": key, "job_id": job_id, "created_at": datetime.now(UTC).isoformat(), "result": result}
        await run_in_threadpool(app.state.records.set, score_id, record)
        return {**result, "score_id": score_id}

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
        key = service.cache_key(data, job_id.strip(), str(job.get("updatedAt", "")), lang or "", app.state.embedder.name, scorer.load_params().version)
        score_id = service.score_id_for(user, key)
        cached = await run_in_threadpool(app.state.cache.get, key)
        if cached:
            cached["meta"]["cached"] = True
            return await _remember(user, score_id, key, job_id.strip(), cached)
        try:
            await asyncio.wait_for(app.state.sem.acquire(), timeout=settings.busy_timeout_seconds)
        except TimeoutError:
            raise Busy("All scoring workers are busy.") from None
        try:
            result = await run_in_threadpool(service.score_cv_bytes, data, job, settings, app.state.embedder, None, lang)
        finally:
            app.state.sem.release()
        await run_in_threadpool(app.state.cache.set, key, result)
        return await _remember(user, score_id, key, job_id.strip(), result)

    async def _own_record(request: Request, score_id: str) -> tuple[str, dict]:
        """Authenticate, validate the id, and load the caller's record. Anything else is 404."""
        user = current_user(request, settings)
        try:
            score_id = str(uuid.UUID(score_id))  # canonical form only; nothing else is ever used as a Redis key
        except ValueError:
            raise BadRequest("scoreId must be a UUID.") from None
        record = await run_in_threadpool(app.state.records.get, score_id)
        if not record or record.get("owner") != user:
            raise ScoreNotFound("No score with this id for the current user (it may have expired).")
        return score_id, record

    @app.get("/api/ai/score-resume/{score_id}")
    async def get_score(request: Request, score_id: str):
        score_id, record = await _own_record(request, score_id)
        return {**record["result"], "score_id": score_id}

    @app.delete("/api/ai/score-resume/{score_id}", status_code=204)
    async def delete_score(request: Request, score_id: str):
        score_id, record = await _own_record(request, score_id)
        # the record and the shared 24 h entry for the same file: "delete" must leave nothing derived from the CV
        await run_in_threadpool(app.state.records.delete, score_id)
        await run_in_threadpool(app.state.cache.delete, record["cache_key"])
        return Response(status_code=204)

    async def _own_session(request: Request, session_id: str) -> tuple[str, str, dict | None]:
        user = current_user(request, settings)
        sid = canonical_id(session_id)
        record = await run_in_threadpool(app.state.chat.load, user, sid) if sid else None
        return user, sid or "", record

    @app.post("/api/ai/session", status_code=201)
    async def create_session(request: Request):
        user = current_user(request, settings)
        sid = str(uuid.uuid4())
        record = await run_in_threadpool(app.state.chat.save, user, sid, [])
        return {"sessionId": sid, "createdAt": record["created_at"]}

    @app.get("/api/ai/session/{session_id}/history")
    async def session_history(request: Request, session_id: str):
        # missing, expired and other users' sessions all read as empty: no 404 for the web client to misread
        _, _, record = await _own_session(request, session_id)
        return record["messages"] if record else []

    @app.get("/api/ai/session/{session_id}")
    async def get_session(request: Request, session_id: str):
        _, sid, record = await _own_session(request, session_id)
        if not record:
            raise SessionNotFound("No chat session with this id for the current user (it may have expired).")
        return {"sessionId": sid, "createdAt": record["created_at"], "updatedAt": record["updated_at"], "messages": record["messages"]}

    @app.delete("/api/ai/session/{session_id}", status_code=204)
    async def delete_session(request: Request, session_id: str):
        user, sid, _ = await _own_session(request, session_id)
        if sid:
            await run_in_threadpool(app.state.chat.delete, user, sid)
        return Response(status_code=204)

    async def _chat_body(request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError:
            raise BadRequest("Body must be JSON.") from None
        if not isinstance(body, dict):
            raise BadRequest("Body must be a JSON object.")
        message = body.get("message")
        if not isinstance(message, str) or not message.strip():
            raise BadRequest("message is required.")
        if len(message) > CHAT_MAX_CHARS:
            raise BadRequest(f"message must be at most {CHAT_MAX_CHARS} characters.")
        stream = body.get("stream", True)
        if not isinstance(stream, bool):
            raise BadRequest("stream must be a boolean.")
        return {"message": message.strip(), "session_id": body.get("sessionId"), "stream": stream}

    @app.post("/api/ai/chat")
    async def chat(request: Request):
        user = current_user(request, settings)
        body = await _chat_body(request)
        if not app.state.chat_limiter.allow(user):
            raise RateLimited("Too many chat messages; try again later.")
        # an unknown, expired or someone else's session id starts a new session; the id comes back in `done`
        requested = canonical_id(body["session_id"])
        record = await run_in_threadpool(app.state.chat.load, user, requested) if requested else None
        sid = requested if record else str(uuid.uuid4())
        history = record["messages"] if record else []
        now = now_iso()
        reply = await engine.answer(body["message"], app.state.search)
        assistant = {"id": str(uuid.uuid4()), "role": "assistant", "content": reply.text, "createdAt": now}
        if reply.jobs:
            assistant["jobs"] = reply.jobs
        user_msg = {"id": str(uuid.uuid4()), "role": "user", "content": body["message"], "createdAt": now}
        messages = (history + [user_msg, assistant])[-MAX_MESSAGES:]
        # persist before the first byte is streamed: a client that disconnects mid-answer keeps its history
        await run_in_threadpool(app.state.chat.save, user, sid, messages, record["created_at"] if record else None)

        if not body["stream"]:
            return {"response": reply.text, "sessionId": sid, "jobs": reply.jobs}

        async def events():
            for chunk in token_frames(reply.text):
                yield chunk
            if reply.error:
                yield frame("error", reply.error)
            if reply.jobs:
                yield frame("jobs", reply.jobs)
            yield frame("done", {"sessionId": sid, "done": True})

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


# run with: uvicorn app.main:create_app --factory --port 6000
