"""Entry point: validate by content (not extension), dispatch to the right parser."""
import io
import zipfile

from app.config import Settings
from app.errors import CvNoText, FileTooLarge, UnsupportedMedia, ZipBomb
from app.parsing.docx_parser import parse_docx
from app.parsing.pdf import parse_pdf
from app.parsing.types import ParsedCv

MIN_TOTAL_CHARS = 80
MAX_DOCX_UNCOMPRESSED = 50 * 1024 * 1024  # 50 MB of extracted XML is far beyond any CV
MAX_DOCX_RATIO = 100  # compressed archive -> uncompressed XML (bomb guard)


def _check_docx_size(z: zipfile.ZipFile, data: bytes) -> None:
    """Reject a DOCX zip bomb: a small archive can expand to gigabytes of XML for the parser."""
    total = 0
    for info in z.infolist():
        total += info.file_size
        if total > MAX_DOCX_UNCOMPRESSED:
            raise ZipBomb("A DOCX with more than 50 MB of extracted content cannot be a CV.")
    if len(data) > 0 and total / len(data) > MAX_DOCX_RATIO:
        raise ZipBomb("The DOCX compresses more than 100x, which is not a real CV.")


def detect_type(data: bytes) -> str:
    if b"%PDF" in data[:1024]:
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if "word/document.xml" in z.namelist():
                    _check_docx_size(z, data)  # may raise ZipBomb (zip bomb)
                    return "docx"
        except zipfile.BadZipFile:
            pass  # a zip that is not a DOCX falls through to "zip"
        except ZipBomb:
            raise  # do not mask the bomb guard as an unsupported type
        return "zip"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "doc"
    return "unknown"


def parse_cv(data: bytes, settings: Settings) -> ParsedCv:
    if len(data) > settings.max_upload_bytes:
        raise FileTooLarge(f"Maximum upload size is {settings.max_upload_bytes // (1024 * 1024)} MB.")
    kind = detect_type(data)
    if kind == "doc":
        raise UnsupportedMedia("Legacy .doc files are not supported. Save the CV as PDF or DOCX.")
    if kind not in ("pdf", "docx"):
        raise UnsupportedMedia("Only PDF and DOCX files are supported.")
    parsed = parse_pdf(data, settings) if kind == "pdf" else parse_docx(data)
    if len(parsed.text.strip()) < MIN_TOTAL_CHARS:
        raise CvNoText("No readable text was found. Upload a clearer or text-based file.")
    return parsed
