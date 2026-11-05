import io

import pymupdf
import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from lxml import etree

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

def _add_dark_shading(paragraph, fill: str = "1F3864") -> None:
    """Shade the paragraph behind it, as dark-sidebar CV templates do."""
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def make_docx_dark_sidebar(skills_line: str) -> bytes:
    """White text on a dark paragraph fill (a very common CV template)."""
    d = Document()
    p = d.add_paragraph()
    run = p.add_run(skills_line)
    run.font.color.rgb = RGBColor(255, 255, 255)
    _add_dark_shading(p)
    d.add_paragraph().add_run("Skills: Python, Django, PostgreSQL and Docker for backend APIs. 5 years experience.")
    d.add_paragraph().add_run("Experience: Backend Developer at Acme Corp, built REST APIs with Django and PostgreSQL.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def make_docx_hyperlink(text: str) -> bytes:
    """Text nested inside a hyperlink (w:hyperlink) and a text box (w:txbxContent)."""
    d = Document()
    p = d.add_paragraph()
    p.add_run("See my project: ")
    hlink = OxmlElement("w:hyperlink")
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    r.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    r.append(t)
    hlink.append(r)
    p._p.append(hlink)

    # a text box (mc:AlternateContent with a w:txbxContent child), built with raw lxml so
    # that the mc/wps namespaces are actually registered
    txbx_xml = (
        '<mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'mc:Ignorable="w14 w15"><mc:Choice Requires="wps"><w:txbxContent><w:p><w:r>'
        '<w:t>Boxed skills: Kubernetes, Redis, AWS</w:t></w:r></w:p></w:txbxContent>'
        '</mc:Choice><mc:Fallback><w:pict><v:shape xmlns:v="urn:schemas-microsoft-com:vml">'
        '<v:textbox><w:txbxContent><w:p><w:r><w:t>Boxed skills: Kubernetes, Redis, AWS</w:t>'
        '</w:r></w:p></w:txbxContent></v:textbox></v:shape></w:pict></mc:Fallback>'
        '</mc:AlternateContent>'
    )
    d.add_paragraph()._p.append(etree.fromstring(txbx_xml))

    d.add_paragraph().add_run("Experience: Backend Developer, built REST APIs with Python and Django for 5 years.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()

