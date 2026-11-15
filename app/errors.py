"""Typed errors mapped to RFC 7807-style responses (SPEC section 7)."""


class ApiError(Exception):
    status = 500
    code = "internal_error"
    title = "Internal error"

    def __init__(self, detail: str = ""):
        super().__init__(detail or self.title)
        self.detail = detail or self.title


class BadRequest(ApiError):
    status, code, title = 400, "bad_request", "Bad request"


class JobNotFound(ApiError):
    status, code, title = 404, "job_not_found", "Job not found"


class ScoreNotFound(ApiError):
    """Missing, expired and somebody else's score all look the same, so ids cannot be probed."""

    status, code, title = 404, "score_not_found", "Score not found"


class SessionNotFound(ApiError):
    """Missing, expired and somebody else's chat session look the same."""

    status, code, title = 404, "session_not_found", "Chat session not found"


class FileTooLarge(ApiError):
    status, code, title = 413, "file_too_large", "File too large"


class UnsupportedMedia(ApiError):
    status, code, title = 415, "unsupported_media_type", "Unsupported file type"


class CvEncrypted(ApiError):
    status, code, title = 422, "cv_encrypted", "CV is password protected"


class CvUnreadable(ApiError):
    status, code, title = 422, "cv_unreadable", "CV file is corrupt or unreadable"


class ZipBomb(CvUnreadable):
    """A DOCX whose uncompressed size or compression ratio is beyond what a CV can be."""

    title = "The DOCX file is too large to read safely"


class CvNoText(ApiError):
    status, code, title = 422, "cv_no_text", "No text could be extracted from the CV"


class RateLimited(ApiError):
    status, code, title = 429, "rate_limited", "Too many requests"


class JobServiceUnavailable(ApiError):
    status, code, title = 503, "job_service_unavailable", "Job service unavailable"


class Busy(ApiError):
    status, code, title = 503, "busy", "Service busy, retry shortly"


class SearchUnavailable(ApiError):
    """search-svc is not configured, unreachable or failing. Used inside the chat stream as an error event."""

    status, code, title = 503, "search_service_unavailable", "Search service unavailable"


class Unauthorized(ApiError):
    status, code, title = 401, "unauthorized", "Unauthorized"


class EmbedderUnavailable(ApiError):
    """The local embedding model is not loaded. Never answered with placeholder vectors."""

    status, code, title = 503, "embedder_unavailable", "Embedding model unavailable"
