import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.errors import JobNotFound
from app.main import create_app
from tests.conftest import CV_TEXT, make_docx, make_image_pdf, make_text_pdf

JOB_ID = "11111111-1111-1111-1111-111111111111"
PY_JOB = {"id": JOB_ID, "title": "Python Backend Developer", "updatedAt": "2026-01-01",
          "requirements": "- At least 3 years of experience with Python and Django\n- Experience with PostgreSQL and Docker\n- Bachelor degree in Computer Science\n- Kubernetes is a plus",
          "description": "Build and maintain REST APIs for our platform.", "experienceLevel": "Mid"}
NURSE_CV = """Mary Jones
Registered Nurse

SKILLS
Patient care, wound care, medication administration, vital signs, triage

EXPERIENCE
Staff Nurse, City Hospital  2012 - 2024
- Provided patient care and emergency care to 30 patients per shift.

EDUCATION
Bachelor of Science in Nursing, 2011
"""


class FakeJobs:
    def __init__(self):
        self.calls = 0

    async def get(self, job_id):
        self.calls += 1
        if job_id == JOB_ID:
            return PY_JOB
        raise JobNotFound("nope")

    async def close(self):
        pass


@pytest.fixture
def client(settings, emb):
    app = create_app(settings, emb, FakeJobs())
    with TestClient(app) as c:
        yield c


def post(client, data, job_id=JOB_ID, name="cv.pdf", headers=None):
    return client.post("/api/ai/score-resume", files={"cv_file": (name, data)}, data={"job_id": job_id}, headers={"X-User-Id": "u1", **(headers or {})})


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["models_loaded"]


def test_relevant_cv_outscores_irrelevant(client):
    good = post(client, make_text_pdf(CV_TEXT)).json()
    bad = post(client, make_text_pdf(NURSE_CV)).json()
    assert good["overall_score"] > bad["overall_score"] + 25, (good["overall_score"], bad["overall_score"])
    assert {"python", "django", "postgresql", "docker"} <= set(good["breakdown"]["skills"]["matching"])
    assert good["breakdown"]["skills"]["missing"] == [] and good["breakdown"]["experience"]["cv_years"] >= 5
    assert bad["breakdown"]["skills"]["matching"] == [] and bad["overall_score"] <= 60


def test_response_shape_and_meta(client):
    j = post(client, make_text_pdf(CV_TEXT)).json()
    assert set(j) >= {"overall_score", "band", "breakdown", "suggestions", "warnings", "meta"}
    assert 0 <= j["overall_score"] <= 100 and j["meta"]["parser"] == "text" and j["meta"]["cached"] is False
    assert "@" not in str(j) and "phone" not in j  # no contact data in evidence


def test_scanned_cv_scores_like_text_cv(client):
    a = post(client, make_text_pdf(CV_TEXT)).json()["overall_score"]
    b = post(client, make_image_pdf(CV_TEXT)).json()
    assert b["meta"]["parser"] == "ocr" and abs(a - b["overall_score"]) <= 12


def test_docx_supported(client):
    r = post(client, make_docx([(CV_TEXT, {})]), name="cv.docx")
    assert r.status_code == 200 and r.json()["meta"]["parser"] == "docx"


def test_keyword_stuffing_does_not_help(client):
    def stuff(page):
        page.insert_textbox(pymupdf.Rect(50, 600, 560, 780), "python django postgresql docker kubernetes " * 20, fontsize=10, color=(1, 1, 1))

    plain = post(client, make_text_pdf(NURSE_CV)).json()["overall_score"]
    stuffed = post(client, make_text_pdf(NURSE_CV, stuff)).json()
    assert stuffed["overall_score"] == plain and any(w.startswith("hidden_text_removed") for w in stuffed["warnings"])


def test_negated_skill_not_credited(client):
    cv = CV_TEXT.replace("Python, Django, PostgreSQL, Docker, AWS, Git", "Django, PostgreSQL, AWS, Git. I do not know Docker.").replace("Deployed with Docker on AWS", "Deployed on AWS")
    j = post(client, make_text_pdf(cv)).json()
    assert "docker" in j["breakdown"]["skills"]["negated"] and "docker" not in j["breakdown"]["skills"]["matching"]
    assert any("docker" in s.lower() for s in j["suggestions"])


def test_cache_hit(client):
    d = make_text_pdf(CV_TEXT)
    assert post(client, d).json()["meta"]["cached"] is False
    assert post(client, d).json()["meta"]["cached"] is True


@pytest.mark.parametrize("data,job,name,status,code", [
    (b"plain text file", JOB_ID, "cv.txt", 415, "unsupported_media_type"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"0" * 50, JOB_ID, "cv.doc", 415, "unsupported_media_type"),
    (b"%PDF-1.4 garbage", JOB_ID, "cv.pdf", 422, "cv_unreadable"),
    (b"", JOB_ID, "cv.pdf", 400, "bad_request"),
    (None, "not-a-uuid", "cv.pdf", 400, "bad_request"),
    (None, "22222222-2222-2222-2222-222222222222", "cv.pdf", 404, "job_not_found"),
])
def test_errors(client, data, job, name, status, code):
    r = post(client, data if data is not None else make_text_pdf(CV_TEXT), job_id=job, name=name)
    assert r.status_code == status and r.json()["code"] == code


def test_blank_pdf_422(client):
    blank = pymupdf.open()
    blank.new_page()
    r = post(client, blank.tobytes())
    assert r.status_code == 422 and r.json()["code"] == "cv_no_text"


def test_too_large_413(client, settings):
    r = post(client, b"%PDF" + b"0" * (settings.max_upload_bytes + 10))
    assert r.status_code == 413


def test_requires_identity(client):
    r = client.post("/api/ai/score-resume", files={"cv_file": ("a.pdf", make_text_pdf(CV_TEXT))}, data={"job_id": JOB_ID})
    assert r.status_code == 401


def test_rate_limit(settings, emb):
    settings.rate_limit_per_min = 2
    with TestClient(create_app(settings, emb, FakeJobs())) as c:
        codes = [post(c, make_text_pdf(CV_TEXT)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_vietnamese_cv_gets_vietnamese_suggestions(client):
    vi = """Nguyễn Văn A
Lập trình viên

KỸ NĂNG
Java, Spring Boot, MySQL

KINH NGHIỆM LÀM VIỆC
Lập trình viên, Công ty ABC  2020 - 2024
- Phát triển các dịch vụ cho hệ thống của công ty và tham gia cải tiến quy trình làm việc của nhóm.

HỌC VẤN
Đại học Bách Khoa, cử nhân Công nghệ thông tin
"""
    j = post(client, make_text_pdf(vi)).json()
    assert j["meta"]["language"] == "vi" and any("Công việc" in s or "CV" in s for s in j["suggestions"])


def test_log_level_accepts_dotnet_names():
    from app.main import log_level

    assert [log_level(x) for x in ("Information", "WARNING", "debug", "nonsense")] == ["INFO", "WARNING", "DEBUG", "INFO"]


def test_or_requirement_is_satisfied_by_either(client):
    # PY_JOB asks for "Django or Flask"? -> use a dedicated job
    from app.scoring.scorer import score
    from app.scoring.specs import build_cv_spec, build_job_spec

    job = build_job_spec({"id": "x", "title": "Dev", "requirements": "- Experience with Django or Flask\n- PostgreSQL or MySQL", "experienceLevel": "Entry"})
    assert job.groups == [["django", "flask"], ["postgresql", "mysql"]]
    cv = build_cv_spec("SKILLS\nDjango, MySQL\nEXPERIENCE\nDev at X 2020 - 2024\nBuilt web apps with Django and MySQL for many customers and teams.")
    r = score(cv, job, client.app.state.embedder)
    assert r["breakdown"]["skills"]["score"] == 100 and r["breakdown"]["skills"]["missing"] == []


def test_concurrent_requests_are_safe(client):
    from concurrent.futures import ThreadPoolExecutor

    data = [make_text_pdf(CV_TEXT), make_text_pdf(NURSE_CV)]
    with ThreadPoolExecutor(6) as ex:
        codes = list(ex.map(lambda i: post(client, data[i % 2], headers={"X-User-Id": f"user-{i}"}).status_code, range(12)))
    assert codes == [200] * 12


def test_lang_param_overrides_suggestion_language_and_cache(client):
    d = make_text_pdf(NURSE_CV)
    en = post(client, d).json()
    r = client.post("/api/ai/score-resume", files={"cv_file": ("cv.pdf", d)}, data={"job_id": JOB_ID, "lang": "vi"}, headers={"X-User-Id": "u1"}).json()
    assert r["meta"]["cached"] is False  # lang is part of the cache key
    assert any("Công việc" in s for s in r["suggestions"]) and not any("Công việc" in s for s in en["suggestions"])


def test_breakdown_shows_ungated_experience_and_education(client):
    j = post(client, make_text_pdf(CV_TEXT)).json()["breakdown"]
    assert j["education"]["score"] == 100 and j["experience"]["score"] == 100  # CV meets both requirements
