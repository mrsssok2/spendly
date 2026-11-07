"""The eval harness's own logic (no Ollama): ground truth, periods, grading, the case file."""
import json
import os
import sys
from datetime import date

import pytest

import database.db as db

EVALS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals")
sys.path.insert(0, EVALS)

import grading  # noqa: E402
from ground_truth import GroundTruth, period_range, placeholders, resolve_date  # noqa: E402
from seed import seed  # noqa: E402

TODAY = date(2026, 11, 7)   # a Saturday


@pytest.fixture
def truth(fresh_database):
    user, _ = seed(db, TODAY)
    t = GroundTruth(fresh_database, user, TODAY)
    yield t
    t.close()


def test_periods():
    assert period_range("none", TODAY) == (None, None)
    assert period_range("day_before_yesterday", TODAY) == ("2026-11-05", "2026-11-05")
    assert period_range("this_week", TODAY) == ("2026-11-02", "2026-11-08")
    assert period_range("last_week", TODAY) == ("2026-10-26", "2026-11-01")
    assert period_range("last_month", date(2026, 1, 15)) == ("2025-12-01", "2025-12-31")
    assert period_range("empty_month", TODAY) == ("2026-03-01", "2026-03-31")
    assert placeholders(TODAY)["last_month_label"] == "October 2026"
    assert resolve_date("yesterday", TODAY) == "2026-11-06"


def test_ground_truth_numbers(truth):
    assert truth.value({"agg": "count"}) == 20
    assert truth.value({"agg": "sum", "period": "today"}) == 360
    assert truth.value({"agg": "count", "gt": 280}) != truth.value({"agg": "count", "ge": 280})
    assert truth.value({"agg": "count", "eq": 280}) == 2
    assert truth.value({"agg": "avg", "period": "empty_month"}) is None
    assert truth.value({"agg": "sum", "period": "empty_month"}) == 0
    assert truth.top_names({"group": "category", "by": "count"}) == ["Food", "Transport"]   # the tie
    assert [r[3] for r in truth.rows({"lt": 100})] == ["bus", "medicines", "chai"]
    assert truth.value({"agg": "diff", "a": {"agg": "sum", "period": "today"},
                        "b": {"agg": "sum", "period": "yesterday"}}) == 185


def test_other_users_rows_are_never_in_the_truth(truth):
    assert truth.value({"agg": "max"}) == 2000      # not the other user's 77777


@pytest.mark.parametrize("text, expected, ok", [
    ("Average amount per expense: ₹289.50 (12 expenses).", 289.5, True),
    ("You spent ₹1,23,456.00 in total.", 123456, True),
    ("About ₹290 per expense.", 289.5, True),             # rounded to a whole rupee
    ("₹579.00 per expense", 289.5, False),
    ("You spent ₹80 on 05 Nov 2026.", 5, False),           # a date is not a number
    ("You have no expenses today.", 0, True),
])
def test_has_number(text, expected, ok):
    assert grading.has_number(text, expected) is ok


def test_previews_must_match_exactly():
    pending = {"a": {"type": "add", "expenses": [{"amount": 150.0, "category": "Food", "date": "2026-11-07",
                                                 "description": "chai"}]}}
    previews = grading.flatten_pending(pending)
    want = [{"type": "add", "amount": 150, "category": "Food", "date": "2026-11-07"}]
    assert grading.grade_previews(previews, want)[0]
    assert not grading.grade_previews(previews, [{**want[0], "category": "Other"}])[0]
    assert not grading.grade_previews(previews, [])[0]          # an unexpected preview fails


def test_case_file_is_up_to_date_and_valid():
    import make_cases
    with open(os.path.join(EVALS, "cases.json"), encoding="utf-8") as f:
        assert json.load(f) == make_cases.CASES, "run: python evals/make_cases.py"
    ids = [c["id"] for c in make_cases.CASES]
    assert len(ids) == len(set(ids)) and len(ids) >= 80
    assert any("regression" in c["tags"] for c in make_cases.CASES)


def test_insights_ground_truth_and_grading(truth):
    t = truth.insights_numbers()
    assert t["same_days_note"] == "1–7 Nov vs 1–7 Oct"
    assert t["income"] > 0 and t["rate"] is not None

    fair = f"{'▲' if t['same_days_pct'] > 0 else '▼'} {abs(t['same_days_pct'])}%"
    good = {"status": "ok", "summary": "You've spent less.", "tips": ["a", "b", "c"],
            "key_numbers": [{"label": "vs same days last month", "value": fair, "note": t["same_days_note"]}]}
    assert grading.grade_insights(good, {"same_days_change": True}, t) == []
    old = {**good, "key_numbers": [{"label": "Change", "value": "▼ 68.3%"}]}
    assert any("old 'Change'" in p for p in grading.grade_insights(old, {"same_days_change": True}, t))

    lie = {**good, "tips": ["Your savings so far are ₹22,385 (60.5% of income) — keep going.", "b", "c"]}
    assert any("no income" in p for p in grading.grade_insights(lie, {"no_income_words": True}, t))
    problems = grading.grade_insights(lie, {"savings_labels": True}, t)
    assert any("₹22,385.00 is not the savings" in p for p in problems)
    assert any("60.5% is not the savings rate" in p for p in problems)

    no_forecast_tip = grading.grade_insights(good, {"forecast_tip": True}, t, [("Food", 10000)])
    assert no_forecast_tip and "Food" in no_forecast_tip[0]
    with_tip = {**good, "tips": ["Food is on track to exceed your ₹10,000 budget around 27 Nov — plan.", "b", "c"]}
    assert grading.grade_insights(with_tip, {"forecast_tip": True}, t, [("Food", 10000)]) == []
