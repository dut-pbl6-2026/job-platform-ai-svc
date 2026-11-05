"""Text normalisation: NFC, de-hyphenation, whitespace; folded (lower, diacritic-free) copy for matching."""
import re
import unicodedata

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_URL = re.compile(r"(?:https?://|www\.|linkedin\.com/|github\.com/)\S+", re.IGNORECASE)
_BULLETS = "•·●▪■◦▶►➢✓✔❖*–—-"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace(" ", " ").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fold(text: str) -> str:
    """Lower-case and strip diacritics (đ -> d). Keeps length of ASCII text; used only for matching."""
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d").replace("Đ", "d"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def redact(text: str) -> str:
    """Remove contact data from any excerpt that leaves the service."""
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[link]", text)
    return _PHONE.sub("[phone]", text)


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in text.split("\n"):
        line = line.strip().lstrip(_BULLETS + " ")
        if not line:
            continue
        for p in re.split(r"(?<=[.;!?])\s+|\s[|•·●▪■◦]\s", line):
            p = p.strip()
            if len(p) > 1:
                parts.append(p)
    return parts
