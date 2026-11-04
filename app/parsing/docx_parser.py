"""DOCX parsing (body paragraphs + tables, in document order). Drops hidden / white / tiny runs."""
import io

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.errors import CvUnreadable
from app.parsing.types import ParsedCv

MIN_FONT_PT = 3.0


def _run_hidden(run) -> bool:
    rpr = run._r.rPr
    if rpr is not None and rpr.find(qn("w:vanish")) is not None:
        return True
    try:
        rgb = run.font.color.rgb if run.font.color is not None and run.font.color.type is not None else None
    except Exception:  # noqa: BLE001
        rgb = None
    if rgb is not None and all(c >= 0xF4 for c in (rgb[0], rgb[1], rgb[2])):
        return True
    return run.font.size is not None and run.font.size.pt < MIN_FONT_PT


def _para_text(p: Paragraph) -> tuple[str, int]:
    kept, removed = [], 0
    for run in p.runs:
        if _run_hidden(run):
            removed += len(run.text.strip())
        else:
            kept.append(run.text)
    return "".join(kept).strip(), removed


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
                    ctext = []
                    for p in cell.paragraphs:
                        t, r = _para_text(p)
                        removed += r
                        if t:
                            ctext.append(t)
                    if ctext:
                        cells.append(" ".join(ctext))
                if cells:
                    lines.append(" | ".join(dict.fromkeys(cells)))
    warnings = [f"hidden_text_removed:{removed}"] if removed else []
    return ParsedCv("\n".join(lines), "docx", 1, 1, warnings, removed)
