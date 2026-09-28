"""Oracle fallback-matrix tests — the REAL local scorer (no dummy `return 0.0`)."""
import pytest

import oracle


def test_negative_news_scores_negative():
    s = oracle.fallback_score("Company hit by massive fraud scandal and lawsuit", "TCS", "TECH")
    assert s < 0


def test_positive_news_scores_positive():
    s = oracle.fallback_score("Company reports record profit and wins major contract", "TCS", "TECH")
    assert s > 0


def test_neutral_news_near_zero():
    s = oracle.fallback_score("Company holds routine annual general meeting", "TCS", "TECH")
    assert abs(s) < 0.4


def test_score_is_clamped():
    s = oracle.fallback_score(
        "crash fraud bankruptcy war scandal recall lawsuit hack breach default", "TCS", "TECH")
    assert -1.0 <= s <= 1.0


def test_deterministic():
    a = oracle.fallback_score("Company faces sanction and downgrade", "TCS", "BANK")
    b = oracle.fallback_score("Company faces sanction and downgrade", "TCS", "BANK")
    assert a == b
