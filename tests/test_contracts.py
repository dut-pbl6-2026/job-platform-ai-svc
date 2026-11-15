"""Contract tests: request/response models for chat, resume scoring, parsing, the search client and errors.

Two sides are checked against the same models in tests/contracts/models.py:
  1. the JSON fixtures in tests/contracts/fixtures (the documented examples), and
  2. what the code actually produces (TestClient responses, SSE frames, the parser dataclass, the search client
     reading a canned search-svc response). No network, no model, no Redis.
"""
import dataclasses
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.chat.search import SearchClient
from app.config import Settings
from app.main import create_app
from app.parsing.parse import parse_cv
from app.parsing.types import ParsedCv
from tests.conftest import CV_TEXT, make_text_pdf
from tests.contracts.models import (
    SSE_MODELS,
    ChatJsonResponse,
    ChatMessage,
    ChatRequest,
    DoneEvent,
    JobHit,
    ParsedCvContract,
    ProblemDetails,
    ScoreResponse,
    SemanticSearchResponse,
    SessionCreated,
    SessionView,
)
from tests.test_api import JOB_ID, FakeJobs, post
from tests.test_chat import search_with
from tests.test_chat_api import FakeSearch

FIXTURES = Path(__file__).parent / "contracts" / "fixtures"


def fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --- 1. fixtures match the models -------------------------------------------

@pytest.mark.parametrize(
    "name,model",
    [
        ("chat_request.json", ChatRequest),
        ("chat_json_response.json", ChatJsonResponse),
        ("session_view.json", SessionView),
        ("semantic_search_response.json", SemanticSearchResponse),
        ("parsed_cv.json", ParsedCvContract),
        ("problem_details.json", ProblemDetails),
        ("score_response.json", ScoreResponse),
    ],
)
def test_fixture_matches_model(name, model):
    model.model_validate(fixture(name))


def test_sse_fixture_matches_the_event_models():
    events = fixture("sse_events.json")
    assert [e["event"] for e in events] == ["token", "token", "error", "jobs", "done"]
    for event in events:
        if event["event"] == "jobs":
            [JobHit.model_validate(h) for h in event["data"]]
        else:
            SSE_MODELS[event["event"]].model_validate(event["data"])


@pytest.mark.parametrize(
    "name,model,mutate",
    [
        ("score_response.json", ScoreResponse, lambda d: d.update(overallScore=d.pop("overall_score"))),  # renamed field
        ("score_response.json", ScoreResponse, lambda d: d.update(overall_score="high")),  # wrong type (numeric strings are coerced by pydantic)
        ("score_response.json", ScoreResponse, lambda d: d["meta"].update(parser="pdf")),  # value outside the contract
        ("chat_json_response.json", ChatJsonResponse, lambda d: d["jobs"][0].update(company_name="Acme")),
        ("parsed_cv.json", ParsedCvContract, lambda d: d.update(hidden_removed="none")),
        ("problem_details.json", ProblemDetails, lambda d: d.pop("code")),
    ],
)
def test_drift_from_the_contract_is_caught(name, model, mutate):
    data = fixture(name)
    mutate(data)
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_request_limits_are_part_of_the_contract():
    ChatRequest.model_validate({"message": "x" * 2000})
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "x" * 2001})
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "hi", "unexpected": True})


# --- 2. the code produces the contract ----------------------------------------

@pytest.fixture
def client(settings, emb):
    with TestClient(create_app(settings, emb, FakeJobs(), search_client=FakeSearch())) as c:
        yield c


def test_score_post_and_get_match_the_contract(client):
    first = post(client, make_text_pdf(CV_TEXT))
    assert first.status_code == 200
    body = ScoreResponse.model_validate(first.json())
    again = client.get(f"/api/ai/score-resume/{body.score_id}", headers={"X-User-Id": "u1"})
    assert ScoreResponse.model_validate(again.json()).overall_score == body.overall_score


def test_parser_output_matches_the_contract_dataclass():
    contract = {name: field.annotation for name, field in ParsedCvContract.model_fields.items()}
    assert [f.name for f in dataclasses.fields(ParsedCv)] == list(contract)
    parsed = parse_cv(make_text_pdf(CV_TEXT), Settings(max_pages=5))
    ParsedCvContract.model_validate(dataclasses.asdict(parsed))


def test_chat_json_reply_matches_the_contract(client):
    r = client.post("/api/ai/chat", json=fixture("chat_request.json"), headers={"X-User-Id": "alice"})
    ChatJsonResponse.model_validate(r.json())
    assert r.json()["jobs"] == [fixture("chat_json_response.json")["jobs"][0]]


def test_chat_stream_frames_match_the_contract(client):
    r = client.post("/api/ai/chat", json={"message": "tìm việc Java ở Hà Nội"}, headers={"X-User-Id": "alice"})
    frames = []
    for block in r.text.split("\n\n"):
        if block.strip():
            event, data = block.split("\n")
            frames.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    assert frames[-1][0] == "done"
    for event, data in frames:
        if event == "jobs":
            [JobHit.model_validate(h) for h in data]
        else:
            SSE_MODELS[event].model_validate(data)
    DoneEvent.model_validate(frames[-1][1])


def test_session_endpoints_match_the_contract(client):
    headers = {"X-User-Id": "alice"}
    created = SessionCreated.model_validate(client.post("/api/ai/session", headers=headers).json())
    client.post("/api/ai/chat", json={"message": "tìm việc Java ở Hà Nội", "sessionId": created.sessionId, "stream": False}, headers=headers)
    view = SessionView.model_validate(client.get(f"/api/ai/session/{created.sessionId}", headers=headers).json())
    assert [m.role for m in view.messages] == ["user", "assistant"]
    assert all(isinstance(ChatMessage.model_validate(m.model_dump()), ChatMessage) for m in view.messages)


def test_error_responses_match_the_problem_contract(settings, emb):
    big = Settings(**{**settings.model_dump(), "max_upload_bytes": 1024})
    with TestClient(create_app(big, emb, FakeJobs())) as c:
        r = post(c, b"%PDF" + b"0" * 2000)
    assert r.status_code == 413
    problem = ProblemDetails.model_validate(r.json())
    assert problem.code == "file_too_large" and problem.status == 413
    with TestClient(create_app(settings, emb, FakeJobs())) as c:
        missing = c.get("/api/ai/session/77777777-7777-7777-7777-777777777777", headers={"X-User-Id": "alice"})
    assert ProblemDetails.model_validate(missing.json()).code == "session_not_found"


async def test_search_client_reads_the_contract_and_returns_job_hits():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=fixture("semantic_search_response.json"))

    hits = await search_with(handler).find_jobs("Java tại Hà Nội")
    SemanticSearchResponse.model_validate(fixture("semantic_search_response.json"))
    assert seen["path"] == "/api/search/jobs/semantic" and seen["params"]["size"] == "3"
    assert [JobHit.model_validate(h).model_dump() for h in hits] == [fixture("chat_json_response.json")["jobs"][0]]


async def test_search_client_drops_items_that_break_the_contract():
    payload = {"items": [{"score": 0.5, "job": {"title": "no id"}}, "not-an-object",
                         {"score": 0.4, "job": {"id": JOB_ID, "title": "ok", "companyName": None}}]}
    client = SearchClient("http://search.test", 1.0)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))
    hits = await client.find_jobs("Java")
    assert [JobHit.model_validate(h).id for h in hits] == [JOB_ID]
    assert hits[0]["company"] == ""
