import io

import pymupdf
import pytest
from docx import Document
from docx.shared import Pt, RGBColor

from app.config import Settings
from app.embed import HashingEmbedder

CV_TEXT = """John Smith
Senior Python Developer

SUMMARY
Backend engineer with 6 years of experience building REST APIs.

SKILLS
Python, Django, PostgreSQL, Docker, AWS, Git

EXPERIENCE
Backend Developer, Acme Corp  Jan 2019 - Dec 2024
- Built REST API services with Django and PostgreSQL serving 2 million users.
- Deployed with Docker on AWS and reduced latency by 30%.

EDUCATION
Bachelor of Science in Computer Science, 2018
"""


@pytest.fixture
def settings() -> Settings:
    return Settings(job_service_url="http://job.test", ocr_concurrency=1, jwt_secret="", redis_url="")


@pytest.fixture
def emb() -> HashingEmbedder:
    return HashingEmbedder()


def make_text_pdf(text: str, extra=None, pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 560, 780), text, fontsize=10, color=(0, 0, 0))
        if extra:
            extra(page)
    return doc.tobytes()


def make_image_pdf(text: str) -> bytes:
    src = pymupdf.open()
    sp = src.new_page(width=595, height=842)
    sp.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text, fontsize=11, color=(0, 0, 0))
    png = sp.get_pixmap(dpi=150).tobytes("jpeg", jpg_quality=80)
    out = pymupdf.open()
    page = out.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=png)
    return out.tobytes()


def make_docx(paragraphs: list[tuple[str, dict]]) -> bytes:
    d = Document()
    for text, fmt in paragraphs:
        run = d.add_paragraph().add_run(text)
        if fmt.get("white"):
            run.font.color.rgb = RGBColor(255, 255, 255)
        if fmt.get("hidden"):
            run.font.hidden = True
        if fmt.get("tiny"):
            run.font.size = Pt(1)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()
