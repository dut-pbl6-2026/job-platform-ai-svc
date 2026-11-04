from datetime import date

import pytest

from app.nlp.education import highest_level, required_level
from app.nlp.experience import date_range_years, explicit_years, required_years
from app.nlp.sections import split_sections
from app.nlp.skills import extract_skills

TODAY = date(2025, 1, 1)


@pytest.mark.parametrize("text,skill", [
    ("Proficient in ReactJS and Node.js", "react"), ("Proficient in ReactJS and Node.js", "node.js"),
    ("Developed apps in C# and .NET Core", "c#"), ("Developed apps in C# and .NET Core", "asp.net"),
    ("Wrote C++ firmware", "c++"), ("Experienced with k8s and Docker", "kubernetes"),
    ("Kinh nghiệm lập trình Python", "python"), ("Sử dụng Postgres và Redis", "postgresql"),
])
def test_alias_matching(text, skill):
    assert skill in extract_skills(text).present


def test_java_is_not_javascript():
    r = extract_skills("Strong JavaScript skills")
    assert "javascript" in r.present and "java" not in r.present


def test_generic_words_do_not_match():
    r = extract_skills("I go to the office. It was a swift response in spring. Education: bachelor")
    assert not r.present


@pytest.mark.parametrize("text", [
    "I do not know Python", "No experience with Python", "Tôi không biết Python", "Toi chua tung dung Python",
    "Kinh nghiệm Python hạn chế", "I have limited experience with Python", "Lacking Python skills", "Chưa có kinh nghiệm Python",
    "without any experience in Python",
])
def test_negation_detected(text):
    r = extract_skills(text)
    assert "python" in r.negated and "python" not in r.present, text


@pytest.mark.parametrize("text", [
    "Not only Python but also Java", "Worked at Acme Limited using Python", "Không chỉ Python mà còn Java",
    "Not limited to Python", "No problems: built Python services", "Skills: Python, Java. Never missed a deadline.",
    "No Java but strong Python",
])
def test_negation_false_positives_avoided(text):
    assert "python" in extract_skills(text).present, text


def test_present_wins_over_negated_mention():
    r = extract_skills("I do not know Python well.\nSkills: Python, Django")
    assert "python" in r.present and "python" not in r.negated


def test_date_ranges_merge_overlaps():
    txt = "Dev, Jan 2018 - Dec 2020\nLead, Jun 2019 - Dec 2021\nOps 2015 - 2016"
    assert date_range_years(txt, TODAY) == pytest.approx(5.0, abs=0.2)  # 2015-16 (1y) + Jan2018-Dec2021 (4y)


def test_present_range_and_vietnamese_dates():
    assert date_range_years("2022 - Present", TODAY) == pytest.approx(3.0, abs=0.2)
    assert date_range_years("Tháng 3/2020 - Tháng 3/2022", TODAY) == pytest.approx(2.0, abs=0.2)


def test_explicit_years():
    assert explicit_years("Engineer with 8+ years of experience in data") == 8
    assert explicit_years("hơn 5 năm kinh nghiệm") == 5
    assert explicit_years("Graduated 2 years ago") == 0


def test_required_years():
    assert required_years("At least 3 years of experience with Java", "", None) == 3
    assert required_years("Yêu cầu 2 năm kinh nghiệm", "", None) == 2
    assert required_years("3-5 years experience", "", None) == 3
    assert required_years("", "", "Senior") == 5
    assert required_years("", "", "Entry") is None


def test_education():
    assert highest_level("Master of Science, 2020. Bachelor of Arts 2018") == "master"
    assert highest_level("Tốt nghiệp Đại học Bách Khoa, cử nhân") == "bachelor"
    assert highest_level("High school") == "none"
    assert required_level("Bachelor's or Master's degree preferred") == "bachelor"


def test_sections():
    s = split_sections("Jane Doe\nSKILLS\nPython\nWORK EXPERIENCE\nDev at X\nEducation\nBSc")
    assert set(s) >= {"header", "skills", "experience", "education"}


def test_alternative_groups():
    from app.nlp.skills import alternative_groups, find_hits

    g = lambda t: alternative_groups(find_hits(t))
    assert g("Experience with Django or Flask") == [["django", "flask"]]
    assert g("PostgreSQL / MySQL experience") == [["postgresql", "mysql"]]
    assert g("Kinh nghiệm với PostgreSQL hoặc MySQL") == [["postgresql", "mysql"]]
    assert g("Python, Django and Docker") == []
    assert g("Django or Flask and building REST APIs") == [["django", "flask"]]
