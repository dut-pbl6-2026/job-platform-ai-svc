"""Pure ASGI request-body cap. Runs before routing, form parsing and auth.

Starlette spools every multipart upload to disk before a route can look at its size, and `request.json()` reads
the whole body into memory. This middleware rejects a body over the limit by Content-Length, and counts the bytes
of chunked bodies as they arrive, so neither path can write or buffer more than `max_bytes`.
"""
import json

PROBLEM_TYPE = b"application/problem+json"


class _BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: int, detail: str):
        self.app = app
        self.max_bytes = max_bytes
        self.detail = detail

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = next((v for k, v in scope.get("headers", []) if k == b"content-length"), None)
        if declared is not None and _as_int(declared) > self.max_bytes:
            await self._reject(send)
            return
        seen = 0
        started = False

        async def counting_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if started:
                raise  # a response is already streaming; it cannot be replaced
            await self._reject(send)

    async def _reject(self, send) -> None:
        body = json.dumps({
            "type": "https://job-platform/errors/file_too_large", "title": "File too large", "status": 413,
            "detail": self.detail, "code": "file_too_large",
        }).encode()
        await send({"type": "http.response.start", "status": 413, "headers": [
            (b"content-type", PROBLEM_TYPE), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def _as_int(raw: bytes) -> int:
    try:
        return int(raw)
    except ValueError:
        return 0
