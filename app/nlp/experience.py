"""Years of experience (SPEC section 4.5) and required years from a job posting."""
import re
from datetime import date

from app.nlp.normalize import fold

_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_PRESENT = r"(?:present|current|now|today|ongoing|hien tai|nay|den nay)"
_DATE = rf"(?:(?:{_MONTH_RE}|thang\s*\d{{1,2}}|\d{{1,2}})\s*[,/.\-]?\s*)?(?:19|20)\d{{2}}"
_RANGE = re.compile(rf"({_DATE})\s*(?:-|–|—|to|until|den|toi|~)\s*({_DATE}|{_PRESENT})", re.IGNORECASE)
_EXPLICIT_EN = re.compile(r"(\d{1,2})(?:\.\d)?\s*\+?\s*(?:plus\s*)?(?:years?|yrs?)(?:\s+of)?(?:\s+(?:relevant|professional|industry|work|total|hands-on|proven|solid))*\s+(?:experience|exp)\b", re.IGNORECASE)
_EXPLICIT_VI = re.compile(r"(\d{1,2})\s*\+?\s*nam\s*(?:kinh nghiem)?", re.IGNORECASE)
_LEVEL_YEARS = {"intern": 0, "fresher": 0, "entry": 0, "junior": 1, "mid": 3, "middle": 3, "intermediate": 3, "senior": 5, "lead": 7, "manager": 7, "principal": 8, "director": 10}


def _to_month(token: str, today: date) -> int | None:
    t = fold(token).strip()
    if re.fullmatch(_PRESENT, t):
        return today.year * 12 + today.month
    y = re.search(r"(19|20)\d{2}", t)
    if not y:
        return None
    year = int(y.group(0))
    rest = t[: y.start()]
    mm = re.search(_MONTH_RE, rest)
    if mm:
        month = _MONTHS.get(mm.group(0)[:3], 1)
    else:
        n = re.search(r"\d{1,2}", rest)
        month = int(n.group(0)) if n and 1 <= int(n.group(0)) <= 12 else 1
    return year * 12 + month


def date_range_years(text: str, today: date | None = None) -> float:
    """Union of dated ranges (so overlapping jobs are not double counted), in years."""
    today = today or date.today()
    now = today.year * 12 + today.month
    spans: list[tuple[int, int]] = []
    for m in _RANGE.finditer(fold(text)):
        a, b = _to_month(m.group(1), today), _to_month(m.group(2), today)
        if a is None or b is None or b < a or a > now + 1:
            continue
        b = min(b, now)
        if (b - a) > 12 * 45 or a < 1970 * 12:
            continue
        spans.append((a, b))
    spans.sort()
    total, cur_s, cur_e = 0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return round(total / 12, 1)


def explicit_years(text: str) -> float:
    f = fold(text)
    vals = [int(m.group(1)) for m in _EXPLICIT_EN.finditer(f)]
    vals += [int(m.group(1)) for m in _EXPLICIT_VI.finditer(f) if "kinh nghiem" in m.group(0)]
    vals = [v for v in vals if 0 < v <= 40]
    return float(max(vals)) if vals else 0.0


def cv_years(experience_text: str, whole_text: str, today: date | None = None) -> float:
    """Dated ranges come from the experience section (or whole CV if absent); explicit claims from the whole CV."""
    dated = date_range_years(experience_text or whole_text, today)
    claimed = explicit_years(whole_text)
    return max(dated, claimed)


_REQ_RANGE = re.compile(r"(\d{1,2})\s*(?:-|–|to|den|toi)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?|nam)", re.IGNORECASE)
_REQ_SINGLE = re.compile(r"(?:at least|minimum(?: of)?|min\.?|toi thieu|it nhat|tu|over|more than|tren)?\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?|nam)", re.IGNORECASE)


def required_years(requirements: str, description: str, experience_level: str | None) -> float | None:
    for text in (requirements, description):
        f = fold(text or "")
        m = _REQ_RANGE.search(f)
        if m:
            return float(m.group(1))
        for m in _REQ_SINGLE.finditer(f):
            ctx = f[max(0, m.start() - 40) : m.end() + 40]
            if re.search(r"experience|exp|kinh nghiem|work|lam viec|background", ctx) and 0 < int(m.group(1)) <= 20:
                return float(m.group(1))
    lvl = fold(experience_level or "").strip()
    for key, yrs in _LEVEL_YEARS.items():
        if key in lvl:
            return float(yrs) if yrs else None
    return None
