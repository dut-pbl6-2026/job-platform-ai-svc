"""OCR limits (unit): every tesseract call is bounded by OCR_TIMEOUT_SECONDS. No tesseract binary needed."""

from app.parsing import ocr, pdf
from tests.conftest import make_image_pdf


def test_every_tesseract_call_gets_the_timeout(monkeypatch, settings):
    seen = []

    def fake_data(img, **kwargs):
        seen.append(("data", kwargs["timeout"]))
        return {"text": [], "conf": [], "block_num": [], "par_num": [], "line_num": []}

    def fake_osd(img, **kwargs):
        seen.append(("osd", kwargs["timeout"]))
        raise RuntimeError("no OSD")

    monkeypatch.setattr(ocr.pytesseract, "image_to_data", fake_data)
    monkeypatch.setattr(ocr.pytesseract, "image_to_osd", fake_osd)
    from PIL import Image

    png = Image.new("L", (50, 50), 255)
    import io

    buf = io.BytesIO()
    png.save(buf, format="PNG")
    ocr.ocr_png(buf.getvalue(), "eng", 7.5)
    assert seen and all(timeout == 7.5 for _, timeout in seen)


def test_a_hung_tesseract_fails_the_page_instead_of_hanging_the_request(monkeypatch, settings):
    def hung(*args, **kwargs):
        assert kwargs["timeout"] == settings.ocr_timeout_seconds
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(pdf.ocr, "ocr_available", lambda: True)
    monkeypatch.setattr(pdf.ocr, "ocr_png", lambda png, langs, timeout: hung(png, langs, timeout=timeout))
    parsed = pdf.parse_pdf(make_image_pdf("Python developer with Django and PostgreSQL experience, five years."), settings)
    assert "ocr_failed:page1" in parsed.warnings and parsed.text == ""


def test_timeout_setting_defaults_to_a_bounded_value():
    from app.config import Settings

    assert 0 < Settings().ocr_timeout_seconds <= 60
