"""Education level lexicon (EN + VI)."""
import re

from app.nlp.normalize import fold

LEVELS = ["none", "diploma", "bachelor", "master", "phd"]
_PATTERNS = {
    "phd": r"\b(?:ph\.? ?d|doctorate|doctoral|tien si)\b",
    "master": r"\b(?:master|msc|m\.sc|m\.s\.|mba|m\.tech|mtech|m\.eng|m\.a\.|thac si|postgraduate)\b",
    "bachelor": r"\b(?:bachelor|b\.sc|bsc|b\.s\.|b\.tech|btech|b\.e\.|b\.eng|b\.a\.|b\.com|bcom|bba|undergraduate degree|university degree|degree in|cu nhan|ky su|dai hoc|tot nghiep dai hoc|b\.?s in)\b",
    "diploma": r"\b(?:diploma|associate degree|associate of|cao dang|trung cap|vocational|certificate iii|a\.a\.)\b",
}


def levels_in(text: str) -> list[str]:
    f = fold(text)
    return [lvl for lvl, pat in _PATTERNS.items() if re.search(pat, f)]


def highest_level(text: str) -> str:
    found = levels_in(text)
    return max(found, key=LEVELS.index) if found else "none"


def required_level(text: str) -> str:
    """Lowest level mentioned in a job posting ('Bachelor or Master' => bachelor); 'none' when not stated."""
    found = levels_in(text)
    return min(found, key=LEVELS.index) if found else "none"


def rank(level: str) -> int:
    return LEVELS.index(level)
