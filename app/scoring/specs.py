"""Structured views of a CV and a job posting (what the scorer compares)."""
import re
from dataclasses import dataclass, field
from datetime import date

from app.nlp import education, experience
from app.nlp.chunks import make_chunks
from app.nlp.language import detect_language
from app.nlp.normalize import fold, normalize, split_sentences
from app.nlp.sections import split_sections
from app.nlp.skills import SkillResult, alternative_groups, extract_skills, find_hits

_NICE_CUES = re.compile(r"\b(?:preferred|a plus|an advantage|advantage|nice to have|desirable|bonus|is a plus|are a plus|uu tien|loi the|diem cong|co them|will be considered)\b")
_SKILL_SECTION_MAX_SHORT = 8  # words: a short line in the experience section looks like a job title


@dataclass
class CvSpec:
    text: str
    language: str
    sections: dict[str, str]
    skills: SkillResult
    years: float
    level: str
    chunks: list[str]
    title_candidates: list[str]
    has_skills_section: bool
    has_numbers: bool
    cv_like: bool


@dataclass
class JobSpec:
    job_id: str
    title: str
    required: list[str] = field(default_factory=list)
    nice: list[str] = field(default_factory=list)
    groups: list[list[str]] = field(default_factory=list)  # required alternatives: any one member satisfies the group
    lines: list[str] = field(default_factory=list)
    required_years: float | None = None
    required_level: str = "none"
    category: str = ""
    version: str = ""
    language: str = "en"


def build_cv_spec(raw_text: str, today: date | None = None) -> CvSpec:
    text = normalize(raw_text)
    sections = split_sections(text)
    skills = extract_skills(text)
    exp_text = sections.get("experience", "")
    years = experience.cv_years(exp_text, "\n".join(v for k, v in sections.items() if k != "education") or text, today)
    edu_text = sections.get("education", "") or text
    level = education.highest_level(edu_text)
    chunks = make_chunks("\n".join(v for k, v in sections.items() if k != "header") or text)
    head = [ln.strip() for ln in sections.get("header", "").split("\n")[:5] if ln.strip()]
    summary = split_sentences(sections.get("summary", ""))[:2]
    short_exp = [ln.strip() for ln in exp_text.split("\n") if 0 < len(ln.split()) <= _SKILL_SECTION_MAX_SHORT][:12]
    title_candidates = (head + summary + short_exp)[:20] or chunks[:3]
    cv_like = bool(set(sections) & {"skills", "experience", "education"}) or len(skills.present) >= 2
    return CvSpec(text, detect_language(text), sections, skills, years, level, chunks, title_candidates,
                  "skills" in sections, bool(re.search(r"\d+\s*%|\$\s?\d|\d+\s*(?:users|customers|clients|projects|million|k\b)", text, re.IGNORECASE)), cv_like)


def build_job_spec(job: dict) -> JobSpec:
    """`job` uses the job-svc JobDetailDto field names (camelCase JSON)."""
    title = (job.get("title") or "").strip()
    requirements = normalize(job.get("requirements") or "")
    description = normalize(job.get("description") or "")
    required: dict[str, None] = {}
    nice: dict[str, None] = {}
    groups: list[list[str]] = []
    req_sents = split_sentences(requirements) if requirements else []
    desc_sents = split_sentences(description)
    for sentences, primary in ((req_sents, True), (desc_sents, not req_sents)):
        for s in sentences:
            nice_sentence = bool(_NICE_CUES.search(fold(s)))
            hits = find_hits(s)
            for h in hits:
                if h.negated:
                    continue
                (required if primary and not nice_sentence else nice)[h.skill] = None
            if primary and not nice_sentence:
                groups += alternative_groups(hits)
    for h in find_hits(title):
        if not h.negated:
            required[h.skill] = None
    for s in list(nice):
        if s in required:
            del nice[s]
    lines = [s for s in req_sents if len(s.split()) >= 3][:12]
    if len(lines) < 3:
        lines += [s for s in desc_sents if len(s.split()) >= 4][: 12 - len(lines)]
    return JobSpec(
        job_id=str(job.get("id", "")),
        title=title,
        required=list(required),
        nice=list(nice),
        groups=[g for g in groups if all(m in required for m in g)],
        lines=lines,
        required_years=experience.required_years(requirements, description, job.get("experienceLevel")),
        required_level=education.required_level(requirements or description),
        category=job.get("categoryName") or "",
        version=str(job.get("updatedAt", "")),
        language=detect_language(f"{title}\n{requirements}\n{description}"),
    )
