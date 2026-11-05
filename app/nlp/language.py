"""Tiny script/stop-word language detector: en | vi | mixed."""
import re

from app.nlp.normalize import fold

_VI_MARKS = re.compile(r"[ăâêôơưđắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợứừửữựáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", re.IGNORECASE)
_VI_WORDS = {"va", "cua", "la", "cac", "nhung", "duoc", "cho", "trong", "voi", "mot", "nam", "kinh", "nghiem", "ky", "nang", "cong", "ty", "nhan", "vien", "hoc", "van", "viec", "lam", "co", "khong", "den", "tu", "tai"}
_EN_WORDS = {"the", "and", "of", "to", "in", "with", "for", "experience", "skills", "work", "team", "years", "on", "at", "a", "is", "as", "by"}


def detect_language(text: str) -> str:
    sample = text[:6000]
    words = re.findall(r"[a-zA-ZÀ-ỹ]+", sample)
    if not words:
        return "en"
    marked = sum(1 for w in words if _VI_MARKS.search(w))
    folded = [fold(w) for w in words]
    vi = sum(1 for w in folded if w in _VI_WORDS) + marked
    en = sum(1 for w in folded if w in _EN_WORDS)
    if vi >= 8 and en >= 8 and 0.3 < vi / max(vi + en, 1) < 0.7:
        return "mixed"
    return "vi" if vi > en else "en"
