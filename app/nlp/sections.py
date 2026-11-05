"""CV section splitting by header lexicon (EN + VI)."""
import re

from app.nlp.normalize import fold

HEADERS: dict[str, list[str]] = {
    "summary": ["summary", "professional summary", "career summary", "profile", "professional profile", "objective", "career objective", "about me", "personal statement", "overview", "muc tieu", "muc tieu nghe nghiep", "gioi thieu", "gioi thieu ban than", "tom tat"],
    "skills": ["skills", "technical skills", "key skills", "core skills", "core competencies", "competencies", "skills and abilities", "skills summary", "areas of expertise", "expertise", "technical proficiency", "technologies", "tools", "strengths", "ky nang", "ky nang chuyen mon", "chuyen mon"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "work history", "employment", "career history", "relevant experience", "internship", "kinh nghiem", "kinh nghiem lam viec", "qua trinh lam viec"],
    "education": ["education", "academic background", "academic qualifications", "qualifications", "education and training", "academics", "hoc van", "trinh do hoc van", "dao tao"],
    "projects": ["projects", "personal projects", "academic projects", "key projects", "du an", "cac du an"],
    "certifications": ["certifications", "certificates", "licenses", "licenses and certifications", "courses", "training", "chung chi", "khoa hoc"],
    "other": ["awards", "honors", "achievements", "interests", "hobbies", "references", "languages", "volunteer", "activities", "publications", "so thich", "giai thuong", "ngoai ngu"],
}
_LOOKUP = {fold(h): sec for sec, hs in HEADERS.items() for h in hs}


def _header_key(line: str) -> str | None:
    s = re.sub(r"[^a-z0-9& ]+", " ", fold(line)).strip()
    s = re.sub(r"\s+", " ", s)
    if not s or len(s.split()) > 5:
        return None
    s = s.replace(" & ", " and ")
    return _LOOKUP.get(s)


def split_sections(text: str) -> dict[str, str]:
    out: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.split("\n"):
        key = _header_key(line)
        if key:
            current = key
            out.setdefault(current, [])
            continue
        out.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in out.items() if "".join(v).strip()}
