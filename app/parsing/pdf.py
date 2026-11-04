"""PDF parsing: per-page text layer with hidden-text filtering, OCR fallback per page."""
import numpy as np
import pymupdf

from app.config import Settings
from app.errors import CvEncrypted, CvUnreadable
from app.parsing import hidden, ocr
from app.parsing.types import ParsedCv

MIN_PAGE_CHARS = 40
MAX_RENDER_SIDE_PX = 2600
SCAN_SCALE = 1.0  # scale used for the hidden-text probe render (1 px per pt)


def _page_text(page: "pymupdf.Page") -> tuple[str, int]:
    """Visible text of a text-layer page and the number of characters removed as hidden."""
    d = page.get_text("dict")
    pix_obj = page.get_pixmap(matrix=pymupdf.Matrix(SCAN_SCALE, SCAN_SCALE), colorspace=pymupdf.csRGB, alpha=False)
    pix = np.frombuffer(pix_obj.samples, dtype=np.uint8).reshape(pix_obj.height, pix_obj.width, 3)
    kept_lines: list[str] = []
    removed = 0
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            parts = []
            for span in line.get("spans", []):
                if hidden.is_hidden_span(span, page.rect, pix, SCAN_SCALE):
                    removed += len(span.get("text", "").strip())
                else:
                    parts.append(span["text"])
            joined = "".join(parts).strip()
            if joined:
                kept_lines.append(joined)
    return "\n".join(kept_lines), removed


def _render_png(page: "pymupdf.Page", dpi: int) -> bytes:
    w, h = page.rect.width, page.rect.height
    scale = min(dpi / 72.0, MAX_RENDER_SIDE_PX / max(w, h))
    return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False).tobytes("png")


def parse_pdf(data: bytes, settings: Settings) -> ParsedCv:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise CvUnreadable("The PDF could not be opened.") from exc
    if doc.needs_pass:
        raise CvEncrypted("The PDF is password protected; remove the password and upload again.")
    total = doc.page_count
    if total == 0:
        raise CvUnreadable("The PDF has no pages.")
    n = min(total, settings.max_pages)
    warnings: list[str] = []
    if total > n:
        warnings.append(f"pages_truncated:{n}/{total}")

    texts: list[str] = []
    used_text = used_ocr = removed_total = 0
    for i in range(n):
        try:
            page = doc[i]
            text, removed = _page_text(page)
        except Exception:  # noqa: BLE001
            text, removed = "", 0
            page = None
        removed_total += removed
        if len(text) >= MIN_PAGE_CHARS:
            texts.append(text)
            used_text += 1
            continue
        if page is None or not ocr.ocr_available():
            if page is not None:
                warnings.append("ocr_unavailable")
            continue
        try:
            otext, conf, rotated = ocr.ocr_png(_render_png(page, settings.ocr_dpi), settings.ocr_langs)
        except Exception:  # noqa: BLE001
            warnings.append(f"ocr_failed:page{i + 1}")
            continue
        if rotated:
            warnings.append(f"page_rotated:{i + 1}")
        if conf and conf < 55:
            warnings.append(f"low_ocr_confidence:{round(conf)}")
        if otext.strip():
            texts.append(otext)
            used_ocr += 1
    if removed_total:
        warnings.append(f"hidden_text_removed:{removed_total}")
    parser = "ocr" if used_ocr and not used_text else "text" if used_text and not used_ocr else "mixed"
    return ParsedCv("\n\n".join(texts), parser, total, n, warnings, removed_total)
