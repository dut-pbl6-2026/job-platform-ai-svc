"""Intent classification and parameter extraction by rules.

Input is folded (lower case, no diacritics) so "Hà Nội", "ha noi" and "HA NOI" match the same pattern.
Order matters: an explicit job request wins over a CV/interview question, and off-topic wins over everything.
"""
import re
from dataclasses import dataclass

from app.nlp.normalize import fold

GREETING = "greeting"
HELP = "help"
CV = "cv"
INTERVIEW = "interview"
JOB_SEARCH = "job_search"
OFF_TOPIC = "off_topic"
UNKNOWN = "unknown"


def _re(alternatives: str) -> re.Pattern[str]:
    # word-ish boundaries that also work for terms like "c#", "node.js" and "tp.hcm"
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])")


_OFF_TOPIC = _re(r"hack\w*|vu khi|weapon|bom|bomb|ma tuy|drugs?|sex|nsfw|khieu dam|lua dao|scam")
_GREETING = re.compile(r"^(?:xin )?(?:chao|hello|hi|hey|alo|good (?:morning|afternoon|evening))(?: (?:ban|bot|em|anh|chi|ad))?$")
_HELP = _re(r"giup|help|lam duoc gi|co the lam gi|what can you do|huong dan|su dung")
_JOB_WORD = _re(r"tim|viec|jobs?|vi tri|tuyen|cong viec|hiring|opening|position|tim kiem|search|apply|ung tuyen")
_CV = _re(r"cv|resume|ho so|curriculum")
_INTERVIEW = _re(r"phong van|interview|tra loi|chuan bi")

# (canonical name shown to the user and sent to search-svc, folded alternatives)
_SKILLS: list[tuple[str, str]] = [
    ("Python", r"python"),
    ("Java", r"java"),
    ("JavaScript", r"javascript"),
    ("TypeScript", r"typescript"),
    ("React", r"react|reactjs|react\.js"),
    ("Vue", r"vue|vuejs|vue\.js"),
    ("Angular", r"angular"),
    ("Node.js", r"node\.?js|node js"),
    (".NET", r"\.net|dotnet|asp\.net|c#"),
    ("C++", r"c\+\+"),
    ("Go", r"golang"),
    ("PHP", r"php"),
    ("Laravel", r"laravel"),
    ("Django", r"django"),
    ("Flask", r"flask"),
    ("Spring", r"spring boot|spring"),
    ("Docker", r"docker"),
    ("Kubernetes", r"kubernetes|k8s"),
    ("AWS", r"aws|amazon web services"),
    ("SQL", r"sql"),
    ("PostgreSQL", r"postgresql|postgres"),
    ("MySQL", r"mysql"),
    ("Flutter", r"flutter"),
    ("Kotlin", r"kotlin"),
    ("Swift", r"swift"),
    ("Android", r"android"),
    ("iOS", r"ios"),
    ("Excel", r"excel"),
    ("Figma", r"figma"),
]
_SKILL_PATTERNS = [(name, _re(alts)) for name, alts in _SKILLS]

# (role shown to the user; sent as the search keyword, folded alternatives)
_ROLES: list[tuple[str, str]] = [
    ("developer", r"lap trinh vien|lap trinh|developer|programmer|software engineer|ky su phan mem|engineer|dev"),
    ("kế toán", r"ke toan|accountant|accounting"),
    ("designer", r"designer|thiet ke|ui\/ux|ux|ui"),
    ("marketing", r"marketing"),
    ("tester", r"tester|qa|kiem thu"),
    ("data analyst", r"data analyst|phan tich du lieu"),
    ("sales", r"sales|nhan vien ban hang|ban hang"),
    ("thực tập sinh", r"thuc tap|intern"),
]
_ROLE_PATTERNS = [(name, _re(alts)) for name, alts in _ROLES]

_LOCATIONS: list[tuple[str, str]] = [
    ("Hà Nội", r"ha noi|hanoi|hn"),
    ("Hồ Chí Minh", r"ho chi minh|tp\.? ?hcm|tphcm|hcm|sai gon|saigon"),
    ("Đà Nẵng", r"da nang|danang"),
    ("Cần Thơ", r"can tho"),
    ("Hải Phòng", r"hai phong"),
    ("Bình Dương", r"binh duong"),
    ("Đồng Nai", r"dong nai"),
]
_LOCATION_PATTERNS = [(name, _re(alts)) for name, alts in _LOCATIONS]

@dataclass(frozen=True)
class Intent:
    name: str
    keyword: str | None = None  # role, sent as the search keyword
    location: str | None = None
    skills: tuple[str, ...] = ()

    @property
    def has_criteria(self) -> bool:
        return bool(self.keyword or self.location or self.skills)


def _roles(text: str) -> str | None:
    return next((name for name, pat in _ROLE_PATTERNS if pat.search(text)), None)


def _location(text: str) -> str | None:
    return next((name for name, pat in _LOCATION_PATTERNS if pat.search(text)), None)


def _skills(text: str) -> tuple[str, ...]:
    return tuple(name for name, pat in _SKILL_PATTERNS if pat.search(text))


def classify(message: str) -> Intent:
    """Return the intent and the search parameters found in ``message``. Never raises on odd input."""
    text = re.sub(r"\s+", " ", fold(message or "")).strip()
    if not text:
        return Intent(UNKNOWN)
    if _OFF_TOPIC.search(text):
        return Intent(OFF_TOPIC)
    if _GREETING.match(text.strip(" !.?,")):
        return Intent(GREETING)

    keyword = _roles(text)
    location = _location(text)
    skills = _skills(text)
    criteria = bool(keyword or location or skills)
    job_word = bool(_JOB_WORD.search(text))

    if job_word:
        return Intent(JOB_SEARCH, keyword, location, skills)
    if _CV.search(text):
        return Intent(CV)
    if _INTERVIEW.search(text):
        return Intent(INTERVIEW)
    if criteria:
        return Intent(JOB_SEARCH, keyword, location, skills)
    if _HELP.search(text):
        return Intent(HELP)
    return Intent(UNKNOWN)
