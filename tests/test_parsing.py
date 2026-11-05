import pymupdf
import pytest

from app.errors import (
    CvEncrypted,
    CvNoText,
    CvUnreadable,
    FileTooLarge,
    UnsupportedMedia,
    ZipBomb,
)
from app.parsing.ocr import ocr_available
from app.parsing.parse import parse_cv
from tests.conftest import (
    CV_TEXT,
    make_docx,
    make_docx_dark_sidebar,
    make_docx_hyperlink,
    make_image_pdf,
    make_text_pdf,
)

needs_ocr = pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")


def test_text_pdf(settings):
    p = parse_cv(make_text_pdf(CV_TEXT), settings)
    assert p.parser == "text" and "PostgreSQL" in p.text and p.warnings == []


@needs_ocr
def test_scanned_pdf_uses_ocr(settings):
    p = parse_cv(make_image_pdf(CV_TEXT), settings)
    assert p.parser == "ocr" and "python" in p.text.lower() and "postgresql" in p.text.lower()


@needs_ocr
def test_mixed_pdf_decided_per_page(settings):
    a, b = pymupdf.open(stream=make_text_pdf(CV_TEXT)), pymupdf.open(stream=make_image_pdf(CV_TEXT))
    a.insert_pdf(b)
    p = parse_cv(a.tobytes(), settings)
    assert p.parser == "mixed" and p.pages == 2


def test_white_text_stuffing_removed(settings):
    def stuff(page):
        page.insert_textbox(pymupdf.Rect(50, 600, 560, 780), "kubernetes terraform golang rust scala haskell " * 6, fontsize=10, color=(1, 1, 1))

    p = parse_cv(make_text_pdf(CV_TEXT, stuff), settings)
    assert "kubernetes" not in p.text.lower() and "Django" in p.text
    assert any(w.startswith("hidden_text_removed") for w in p.warnings)


def test_white_text_on_dark_background_is_kept(settings):
    def dark_header(page):
        page.draw_rect(pymupdf.Rect(0, 0, 595, 45), fill=(0.1, 0.1, 0.3), color=None)
        page.insert_text((60, 30), "JOHN SMITH - PYTHON DEVELOPER", fontsize=14, color=(1, 1, 1))

    p = parse_cv(make_text_pdf(CV_TEXT, dark_header), settings)
    assert "PYTHON DEVELOPER" in p.text and not any(w.startswith("hidden") for w in p.warnings)


def test_tiny_offpage_and_covered_text_removed(settings):
    def tricks(page):
        page.insert_text((60, 790), "tinystuff rustlang cobol fortran", fontsize=1.5, color=(0, 0, 0))
        page.insert_text((2000, 2000), "offpagestuff scala erlang", fontsize=10, color=(0, 0, 0))
        page.insert_text((60, 700), "coveredstuff haskell ocaml", fontsize=10, color=(0, 0, 0))
        page.draw_rect(pymupdf.Rect(50, 685, 400, 710), fill=(1, 1, 1), color=None)

    low = parse_cv(make_text_pdf(CV_TEXT, tricks), settings).text.lower()
    assert not any(w in low for w in ("tinystuff", "offpagestuff", "coveredstuff")) and "django" in low


def test_docx_hidden_runs_removed(settings):
    data = make_docx([("Skills: Python, Django, PostgreSQL and Docker for backend APIs. 5 years experience.", {}), ("kubernetes terraform", {"white": True}),
                      ("golang rust", {"hidden": True}), ("scala erlang", {"tiny": True}), ("Education: Bachelor of Science in Computer Science", {})])
    p = parse_cv(data, settings)
    low = p.text.lower()
    assert p.parser == "docx" and "python" in low and not any(w in low for w in ("kubernetes", "golang", "scala"))
    assert any(w.startswith("hidden_text_removed") for w in p.warnings)


def test_page_cap_warns(settings):
    p = parse_cv(make_text_pdf(CV_TEXT, pages=7), settings)
    assert p.pages == 7 and p.pages_processed == settings.max_pages and any(w.startswith("pages_truncated") for w in p.warnings)


def test_encrypted_pdf(settings):
    doc = pymupdf.open(stream=make_text_pdf(CV_TEXT))
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    with pytest.raises(CvEncrypted):
        parse_cv(data, settings)


def test_error_cases(settings):
    with pytest.raises(UnsupportedMedia):
        parse_cv(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"0" * 100, settings)  # legacy .doc
    with pytest.raises(UnsupportedMedia):
        parse_cv(b"hello world, plain text", settings)
    with pytest.raises(CvUnreadable):
        parse_cv(b"%PDF-1.4 truncated garbage", settings)
    with pytest.raises(FileTooLarge):
        parse_cv(b"%PDF" + b"0" * (settings.max_upload_bytes + 1), settings)
    blank = pymupdf.open()
    blank.new_page()
    with pytest.raises(CvNoText):
        parse_cv(blank.tobytes(), settings)  # nothing to read, nothing to OCR


def test_extension_is_ignored_content_decides(settings):
    p = parse_cv(make_docx([(CV_TEXT, {})]), settings)  # a DOCX regardless of the file name
    assert p.parser == "docx"

def test_docx_white_text_on_dark_fill_is_kept(settings):
    # A dark-sidebar template: white text on a dark fill is visible ink, not stuffing.
    p = parse_cv(make_docx_dark_sidebar("SKILLS: Python, Django, PostgreSQL, Docker, Kubernetes"), settings)
    low = p.text.lower()
    assert "python" in low and "kubernetes" in low
    assert not any(w.startswith("hidden") for w in p.warnings)


def test_docx_hyperlink_and_textbox_text_is_not_lost(settings):
    p = parse_cv(make_docx_hyperlink("github project using FastAPI and Redis"), settings)
    low = p.text.lower()
    assert "fastapi" in low and "redis" in low
    assert "kubernetes" in low  # text box content


def test_docx_zip_bomb_rejected(settings):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "x" * 2_000_000)  # ~2 MB that compresses to almost nothing
        z.writestr("[Content_Types].xml", "<x/>")
    data = buf.getvalue()
    assert len(data) < settings.max_upload_bytes  # small on the wire, huge when expanded
    with pytest.raises(ZipBomb):
        parse_cv(data, settings)


def test_huge_pdf_page_does_not_oom(settings):
    # 14400x14400pt page: the probe render must stay capped, and the page is not OCR'd away.
    doc = pymupdf.open()
    page = doc.new_page(width=14400, height=14400)
    page.insert_textbox(pymupdf.Rect(200, 200, 4000, 900), CV_TEXT, fontsize=22)
    p = parse_cv(doc.tobytes(), settings)
    assert "postgresql" in p.text.lower()

