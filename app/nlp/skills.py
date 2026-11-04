"""Skill taxonomy matching with sentence-scoped negation (SPEC section 4.3-4.4)."""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import yaml

from app.nlp.normalize import fold, redact, split_sentences

DATA = Path(__file__).resolve().parents[2] / "data" / "skills.yaml"
NEG_WINDOW = 4
NEGATION_ENABLED = True  # switched off only by the ablation study (eval/run.py)
_TOKEN = re.compile(r"[a-z0-9+#./']+")
# Negation cues on folded text. 'limited' only counts when it qualifies a knowledge noun (not "XYZ Limited").
_CUES = re.compile(
    r"\b(?:no|not|never|without|lack of|lacking|hate|dislike|cannot|unable to|khong|chua tung|chua|han che|ghet|thieu)\b"
    r"|\blimited (?:experience|knowledge|exposure|skills?|proficiency|understanding|familiarity|use|in|with)\b"
)
_EXCEPTIONS = re.compile(r"\b(?:not only|khong chi|not limited|khong han che|not just|khong chi la|no longer than)\b")
_POST_CUES = re.compile(r"^\W*(?:\w+\s+){0,2}?(?:han che|chua thanh thao|chua gioi|chua vung|chua nhieu|khong thanh thao|limited|not proficient|no experience)\b|^\W*(?:\(|-)?\s*(?:han che|limited)\b")
_RESETS = {"but", "however", "although", "though", "while", "whereas", "nhung", "tuy", "tuy nhien", "song", "va co"}


@dataclass
class Hit:
    skill: str
    negated: bool
    sentence: str
    start: int = 0
    end: int = 0
    folded: str = ""


@dataclass
class SkillIndex:
    alias_to_canon: dict[str, str]
    groups: dict[str, str]
    pattern: re.Pattern


@lru_cache
def load_index() -> SkillIndex:
    raw = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    alias_to_canon: dict[str, str] = {}
    groups: dict[str, str] = {}
    for group, skills in raw.items():
        for canon, aliases in skills.items():
            groups[canon] = group
            for a in {canon, *(aliases or [])}:
                alias_to_canon.setdefault(fold(str(a)), canon)
    alts = sorted(alias_to_canon, key=len, reverse=True)
    body = "|".join(re.escape(a) for a in alts)
    pattern = re.compile(rf"(?<![a-z0-9+#.]){'(?:' + body + ')'}(?![a-z0-9+#]|\.[a-z0-9])")
    return SkillIndex(alias_to_canon, groups, pattern)


def is_negated(folded_sentence: str, start: int, end: int | None = None) -> bool:
    if not NEGATION_ENABLED:
        return False
    prefix = folded_sentence[:start].rsplit(":", 1)[-1]  # a colon opens a new clause ("No problems: built X")
    if end is not None and _POST_CUES.search(folded_sentence[end : end + 40]):
        return True
    tokens = _TOKEN.findall(prefix)[-NEG_WINDOW:]
    if not tokens:
        return False
    window = " ".join(tokens)
    if any(r in tokens for r in _RESETS):
        window = " ".join(tokens[max(i for i, t in enumerate(tokens) if t in _RESETS) + 1 :])
    if not window:
        return False
    window = _EXCEPTIONS.sub(" ", window)
    return bool(_CUES.search(window + " "))


def find_hits(text: str) -> list[Hit]:
    idx = load_index()
    hits: list[Hit] = []
    for sent in split_sentences(text):
        folded = fold(sent)
        # fold() is length-preserving for Latin text except for stray combining marks; match on folded only.
        for m in idx.pattern.finditer(folded):
            canon = idx.alias_to_canon[m.group(0)]
            hits.append(Hit(canon, is_negated(folded, m.start(), m.end()), redact(sent)[:160], m.start(), m.end(), folded))
    return hits


@dataclass
class SkillResult:
    present: dict[str, str] = field(default_factory=dict)  # skill -> evidence sentence
    negated: dict[str, str] = field(default_factory=dict)  # skill -> evidence sentence (only negated mentions)
    counts: dict[str, int] = field(default_factory=dict)


def extract_skills(text: str) -> SkillResult:
    res = SkillResult()
    for h in find_hits(text):
        res.counts[h.skill] = res.counts.get(h.skill, 0) + 1
        if h.negated:
            res.negated.setdefault(h.skill, h.sentence)
        else:
            res.present.setdefault(h.skill, h.sentence)
    for s in list(res.negated):
        if s in res.present:
            del res.negated[s]
    return res


def is_generic(skill: str) -> bool:
    return load_index().groups.get(skill) == "generic"


_ALT = re.compile(r"^[\s,(]*(?:or|hoac|/|\|)[\s,)]*$")


def alternative_groups(hits: list[Hit]) -> list[list[str]]:
    """Skills joined by 'or' / 'hoặc' / '/' inside one sentence are alternatives ("Django or Flask").
    Plain lists ("Python, Django") are not. Returns groups of >= 2 distinct skills."""
    hits = sorted((h for h in hits if not h.negated), key=lambda h: h.start)
    groups: list[list[str]] = []
    cur: list[str] = []
    for prev, nxt in pairwise(hits):
        if _ALT.match(prev.folded[prev.end : nxt.start]) and prev.folded is nxt.folded:
            cur = cur or [prev.skill]
            if nxt.skill not in cur:
                cur.append(nxt.skill)
        else:
            if len(cur) >= 2:
                groups.append(cur)
            cur = []
    if len(cur) >= 2:
        groups.append(cur)
    return groups
