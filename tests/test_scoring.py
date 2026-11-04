"""Scorer internals: skill matching edge cases, the cosine window and the calibrated score map."""
import pytest

from app.nlp.skills import extract_skills
from app.scoring.scorer import Params, apply_label_map, band

JOB = {"id": "job-1", "title": "Security Engineer", "updatedAt": "2026-01-01",
       "requirements": "- Hands-on experience with firewalls, VPN and IDS/IPS\n- Knowledge of TCP/IP and routing",
       "description": "Protect the platform."}


@pytest.mark.parametrize("text,skill", [
    ("Configured firewalls and VPN tunnels", "security"),          # plural alias
    ("Reconciliations and month-end closing", "accounting"),        # plural alias
    ("Delivered construction projects and site supervision", "civil engineering"),  # plural alias
    ("Ran social media campaigns for clients", "marketing"),                       # plural alias
])
def test_plural_aliases_are_matched(text, skill):
    assert skill in extract_skills(text).present


def test_short_aliases_do_not_match_plurals():
    """'hrs' means hours, not human resources."""
    r = extract_skills("Worked 40 hrs per week for 3 years")
    assert "human resources" not in r.present


def test_project_management_terms_are_not_accounting_signals():
    """'risk management' / 'budgeting' belong to project management, not auditing / financial analysis."""
    r = extract_skills("Led risk management, budgeting and stakeholder management for 5 years")
    assert "project management" in r.present
    assert "auditing" not in r.present
    assert "financial analysis" not in r.present


def test_label_map_linear_is_monotone():
    p = Params({"skills": 1.0}, {"same": (0.0, 1.0), "cross": (0.0, 1.0)}, 0.3, "t", {"type": "linear", "a": 1.1, "b": -5.0})
    assert apply_label_map(0.0, p) == pytest.approx(-5.0)
    assert apply_label_map(50.0, p) == pytest.approx(50.0)
    assert apply_label_map(100.0, p) == pytest.approx(105.0)


def test_label_map_isotonic_interpolates_and_clamps():
    knots = [[10.0, 5.0], [50.0, 40.0], [90.0, 90.0]]
    p = Params({"skills": 1.0}, {"same": (0.0, 1.0), "cross": (0.0, 1.0)}, 0.3, "t", {"type": "isotonic", "knots": knots})
    assert apply_label_map(0.0, p) == pytest.approx(5.0)     # clamped to the first knot
    assert apply_label_map(100.0, p) == pytest.approx(90.0)  # clamped to the last knot
    assert apply_label_map(30.0, p) == pytest.approx(22.5)   # interpolated


def test_no_label_map_keeps_the_score():
    p = Params({"skills": 1.0}, {"same": (0.0, 1.0), "cross": (0.0, 1.0)}, 0.3, "t")
    assert apply_label_map(42.0, p) == 42.0


def test_bands_are_unchanged():
    assert band(80) == "Strong" and band(60) == "Good" and band(40) == "Partial" and band(10) == "Weak"
