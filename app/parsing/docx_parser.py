"""DOCX parsing: all body text (paragraphs + tables, in document order), including
text nested in hyperlinks, text boxes and content controls. Drops hidden / white-on-light
/ tiny runs (keyword-stuffing defence), but keeps white text on a dark fill (CV templates)."""
import io

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.errors import CvUnreadable
from app.parsing.types import ParsedCv

MIN_FONT_PT = 3.0
W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _hex_luminance(fill: str) -> float | None:
    """Luminance of a hex colour (e.g. "1F3864"), None when unparsable."""
    try:
        r, g, b = (int(fill[i:i + 2], 16) for i in (0, 2, 4))
        return r * 0.299 + g * 0.587 + b * 0.114
    except (ValueError, IndexError):
        return None


def _dark_fill(r_el) -> bool:
    """True if any ancestor paints a dark fill behind this run element.

    A w:shd on w:pPr (paragraph), w:rPr (run), w:tcPr (cell) or w:tblPrEx (table) gives the
    background in front of a run; `w:fill` is a hex colour in the default RGB space
    (`auto` = no fill). lxml `getparent()` only walks ancestors, so the run itself is checked
    (rPr) and then its paragraph's pPr via the parent element.
    """
    el = r_el
    while el is not None:
        shd = None
        if el.tag == qn("w:rPr"):
            shd = el.find(qn("w:shd"))
        elif el.tag == qn("w:p"):
            ppr = el.find(qn("w:pPr"))
            shd = ppr.find(qn("w:shd")) if ppr is not None else None
        elif el.tag == qn("w:tcPr") or el.tag == qn("w:tblPrEx"):
            shd = el.find(qn("w:shd"))
        if shd is not None and shd.get(qn("w:fill")):
            fill = shd.get(qn("w:fill"))
            if fill != "auto":
                lum = _hex_luminance(fill)
                if lum is not None and lum < 160:
                    return True
        el = el.getparent()
    return False


def _run_hidden(run_el) -> bool:
    rpr = run_el.find(qn("w:rPr"))
    if rpr is not None:
        vanish = rpr.find(qn("w:vanish"))
        if vanish is not None and vanish.get(qn("w:val")) != "0":
            return True
        color = rpr.find(qn("w:color"))
        if color is not None and color.get(qn("w:val")):
            val = color.get(qn("w:val"))
            if val != "auto":
                try:
                    r, g, b = (int(val[i:i + 2], 16) for i in (0, 2, 4))
                    near_white = r >= 0xF4 and g >= 0xF4 and b >= 0xF4
                except ValueError:
                    near_white = False
                if near_white and not _dark_fill(run_el):
                    return True
        sz = rpr.find(qn("w:sz"))
        if sz is not None and sz.get(qn("w:val")):
            try:
                if int(sz.get(qn("w:val"))) / 2.0 < MIN_FONT_PT:
                    return True
            except ValueError:
                pass
    return False


def _para_text(p: Paragraph) -> tuple[str, int]:
    """All text of a paragraph, including runs inside w:hyperlink / w:sdt / text boxes."""
    kept, removed = [], 0
    for t in p._p.findall(".//w:t", W_NS):
        run_el = t.getparent()  # w:r (w:t is a direct child of w:r)
        text = t.text or ""
        if _run_hidden(run_el):
            removed += len(text.strip())
        else:
            kept.append(text)
    return "".join(kept).strip(), removed


def _cell_text(cell_el) -> tuple[str, int]:
    """All text inside a table cell, including nested tables."""
    parts, removed = [], 0
    for t in cell_el.findall(".//w:t", W_NS):
        run_el = t.getparent()
        text = t.text or ""
        if _run_hidden(run_el):
            removed += len(text.strip())
        else:
            parts.append(text)
    return "".join(parts).strip(), removed


def parse_docx(data: bytes) -> ParsedCv:
    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        raise CvUnreadable("The DOCX could not be opened.") from exc
    lines: list[str] = []
    removed = 0
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            text, r = _para_text(Paragraph(child, doc))
            removed += r
            if text:
                lines.append(text)
        elif child.tag == qn("w:tbl"):
            for row in Table(child, doc).rows:
                cells = []
                for cell in row.cells:
                    t, r = _cell_text(cell._tc)
                    removed += r
                    if t:
                        cells.append(t)
                if cells:
                    lines.append(" | ".join(dict.fromkeys(cells)))
    warnings = [f"hidden_text_removed:{removed}"] if removed else []
    return ParsedCv("\n".join(lines), "docx", 1, 1, warnings, removed)
