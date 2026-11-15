"""Contract models for the public and service-to-service payloads of ai-svc.

These describe the wire format that web, mobile, gateway and search-svc depend on. They are test-only: the routes
still return plain dicts. Every model forbids unknown fields, so an added, renamed or retyped field fails a
contract test on purpose and has to be reflected here (and in the clients) in the same change.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- resume scoring ---------------------------------------------------------

class Meta(Strict):
    parser: Literal["text", "ocr", "mixed", "docx"]
    pages: int
    pages_processed: int
    language: str
    ruleset: str
    params: str
    model: str
    cached: bool
    elapsed_ms: int


class SkillsBlock(Strict):
    score: int | None
    matching: list[str]
    missing: list[str]
    negated: list[str]
    extra: list[str]
    evidence: dict[str, str]


class TopMatch(Strict):
    requirement: str
    cv_excerpt: str
    similarity: float


class SemanticBlock(Strict):
    score: int
    title_fit: float
    top_matches: list[TopMatch]


class ExperienceBlock(Strict):
    score: int | None
    cv_years: float
    required_years: float


class EducationBlock(Strict):
    score: int | None
    cv_level: str
    required_level: str


class Breakdown(Strict):
    skills: SkillsBlock
    semantic: SemanticBlock
    experience: ExperienceBlock
    education: EducationBlock


class Components(Strict):
    skills: float | None
    semantic: float | None
    experience: float | None
    education: float | None


class ScoreResponse(Strict):
    """POST /api/ai/score-resume (and GET /api/ai/score-resume/{scoreId}, which returns the same body)."""

    overall_score: int = Field(ge=0, le=100)
    band: str
    breakdown: Breakdown
    suggestions: list[str]
    notes: list[str]
    components: Components
    raw_semantic: float
    lang_pair: str
    relevance: float
    warnings: list[str]
    meta: Meta
    score_id: str


# --- parsing ----------------------------------------------------------------

class ParsedCvContract(Strict):
    """Mirror of app.parsing.types.ParsedCv (the value handed from the parser to scoring)."""

    text: str
    parser: Literal["text", "ocr", "mixed", "docx"]
    pages: int
    pages_processed: int
    warnings: list[str]
    hidden_removed: int


# --- search-svc client ------------------------------------------------------

class SemanticSearchItem(Strict):
    """One item of search-svc GET /api/search/jobs/semantic, as ai-svc reads it."""

    score: float
    job: dict[str, Any]


class SemanticSearchResponse(Strict):
    items: list[SemanticSearchItem]


class JobHit(Strict):
    """What ai-svc keeps per hit (SearchClient.find_jobs) and what chat sends to clients."""

    id: str
    title: str
    company: str


# --- chat -------------------------------------------------------------------

class ChatRequest(Strict):
    message: str = Field(min_length=1, max_length=2000)
    sessionId: str | None = None
    stream: bool = True


class ChatJsonResponse(Strict):
    """POST /api/ai/chat with stream=false."""

    response: str
    sessionId: str
    jobs: list[JobHit]


class TokenEvent(Strict):
    delta: str = Field(min_length=1)


class ErrorEvent(Strict):
    code: str
    message: str


class DoneEvent(Strict):
    sessionId: str
    done: Literal[True]


class SessionCreated(Strict):
    sessionId: str
    createdAt: str


class ChatMessage(Strict):
    id: str
    role: Literal["user", "assistant"]
    content: str
    createdAt: str
    jobs: list[JobHit] | None = None


class SessionView(Strict):
    sessionId: str
    createdAt: str
    updatedAt: str
    messages: list[ChatMessage]


# --- errors -----------------------------------------------------------------

class ProblemDetails(Strict):
    """RFC 7807 body used for every error response."""

    type: str
    title: str
    status: int
    detail: str | None = None
    code: str
    # FastAPI's own 422 bodies carry "detail" as a list and no "code"; only ai-svc errors are checked here.


SSE_MODELS: dict[str, type[BaseModel]] = {"token": TokenEvent, "error": ErrorEvent, "jobs": list, "done": DoneEvent}
