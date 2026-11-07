"""Monthly budgets: table and queries, the Budgets section, alerts, warnings after adding
an expense (form and AI), and the AI budget tools. Dates are relative to today."""
import sqlite3
from datetime import date, timedelta

import pytest

import app as spendly
import database.db as db
from conftest import ask_ai, tool_call

TODAY = date.today()
MONTH_FROM, MONTH_TO = spendly.this_month_range(TODAY)
LAST_MONTH_DAY = (spendly.month_start(TODAY, 0) - timedelta(days=1)).isoformat()


def status(user_id):
    return {row["category"]: row for row in db.get_budget_status(user_id, MONTH_FROM, MONTH_TO)}


def add_form_expense(client, amount, category="Food", day=None):
    return client.post("/expenses/add", data={
        "amount": str(amount), "category": category,
        "date": day or TODAY.isoformat(), "description": "test"}, follow_redirects=True)


# ---------- Table and queries ----------

def test_set_budget_creates_then_updates_one_row(user):
    db.set_budget(user, None, 2000)
    db.set_budget(user, None, 2500)          # overall again: updates, never a second row
    db.set_budget(user, "Food", 1000)
    db.set_budget(user, "Food", 1200)
    rows = db.get_budget_status(user, MONTH_FROM, MONTH_TO)
    assert [(r["category"], r["monthly_limit"]) for r in rows] == [(None, 2500.0), ("Food", 1200.0)]


def test_database_rejects_zero_or_negative_budget(user):
    for bad in (0, -50):
        with pytest.raises(sqlite3.IntegrityError):
            db.set_budget(user, "Food", bad)


def test_database_rejects_two_overall_budgets(user):
    db.set_budget(user, None, 2000)
    conn = db.get_db()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO budgets (user_id, category, monthly_limit) VALUES (?, NULL, 5)", (user,))
    finally:
        conn.close()


def test_budget_status_counts_only_this_users_this_months_matching_spending(user, other_user):
    db.set_budget(user, None, 2000)
    db.set_budget(user, "Food", 400)
    db.add_expense(user, 300, "Food", TODAY.isoformat(), "lunch")
    db.add_expense(user, 1200, "Bills", TODAY.isoformat(), "rent")
    db.add_expense(user, 999, "Food", LAST_MONTH_DAY, "last month")          # not this month
    db.add_expense(other_user, 5000, "Food", TODAY.isoformat(), "Ravi's")     # not this user

    s = status(user)
    assert (s[None]["spent"], s[None]["percent"], s[None]["remaining"]) == (1500.0, 75.0, 500.0)
    assert (s["Food"]["spent"], s["Food"]["percent"], s["Food"]["remaining"]) == (300.0, 75.0, 100.0)


def test_users_only_see_their_own_budgets(user, other_user):
    db.set_budget(other_user, "Food", 9999)
    assert db.get_budget_status(user, MONTH_FROM, MONTH_TO) == []
    assert db.get_budget(user, "Food") is None


# ---------- Budgets section on the profile page ----------

def test_save_budget_from_profile(client, user):
    response = client.post("/budgets", data={"category": "Food", "amount": "1000"}, follow_redirects=True)
    assert b"Food budget set to" in response.data
    assert db.get_budget(user, "Food") == 1000.0

    client.post("/budgets", data={"category": "", "amount": "2000"})
    assert db.get_budget(user, None) == 2000.0


@pytest.mark.parametrize("amount", ["0", "-5", "abc", "", "nan"])
def test_invalid_budget_amount_is_rejected(client, user, amount):
    response = client.post("/budgets", data={"category": "Food", "amount": amount}, follow_redirects=True)
    assert b"Budget must be" in response.data
    assert db.get_budget(user, "Food") is None


def test_invalid_budget_category_is_rejected(client, user):
    client.post("/budgets", data={"category": "Crypto", "amount": "100"})
    assert db.get_budget_status(user, MONTH_FROM, MONTH_TO) == []


def test_budget_routes_require_login(client):
    assert "/login" in client.post("/budgets", data={"category": "", "amount": "1"}).headers["Location"]
    assert "/login" in client.post("/budgets/delete", data={"category": ""}).headers["Location"]


def test_remove_budget_only_affects_own(client, user, other_user):
    db.set_budget(user, "Food", 1000)
    db.set_budget(other_user, "Food", 500)
    client.post("/budgets/delete", data={"category": "Food"})
    assert db.get_budget(user, "Food") is None
    assert db.get_budget(other_user, "Food") == 500.0
    assert client.post("/budgets/delete", data={"category": "Food"}).status_code == 404


@pytest.mark.parametrize("spent, colour", [(500, "budget-ok"), (1400, "budget-warning"),
                                           (1980, "budget-warning"), (2000, "budget-over"),
                                           (2500, "budget-over")])
def test_progress_bar_colour(client, user, spent, colour):
    db.set_budget(user, None, 2000)
    db.add_expense(user, spent, "Food", TODAY.isoformat(), "x")
    page = client.get("/profile").get_data(as_text=True)
    assert f"bar-fill {colour}" in page


def test_banner_for_75_percent_used(client, user):
    db.set_budget(user, None, 2000)
    db.add_expense(user, 1500, "Food", TODAY.isoformat(), "x")
    page = client.get("/profile").get_data(as_text=True)
    assert "⚠️ You&#39;ve used 75% of your ₹2,000 monthly budget" in page


def test_banner_for_exceeded_category_budget(client, user):
    db.set_budget(user, "Food", 1000)
    db.add_expense(user, 1050, "Food", TODAY.isoformat(), "x")
    page = client.get("/profile").get_data(as_text=True)
    assert "🚨 Food budget exceeded by ₹50" in page


def test_no_banner_under_70_percent(client, user):
    db.set_budget(user, None, 2000)
    db.add_expense(user, 100, "Food", TODAY.isoformat(), "x")
    page = client.get("/profile").get_data(as_text=True)
    assert "⚠️" not in page and "🚨" not in page


# ---------- Warnings after adding an expense with the form ----------

def test_form_expense_crossing_80_percent_warns(client, user):
    db.set_budget(user, "Food", 1000)
    db.add_expense(user, 700, "Food", TODAY.isoformat(), "before")
    page = add_form_expense(client, 150).get_data(as_text=True)
    assert "⚠️ You&#39;ve now used 85% of your ₹1,000 Food budget" in page


def test_form_expense_crossing_100_percent_warns(client, user):
    db.set_budget(user, None, 1000)
    db.add_expense(user, 900, "Food", TODAY.isoformat(), "before")
    page = add_form_expense(client, 150).get_data(as_text=True)
    assert "🚨 Monthly budget exceeded by ₹50" in page


def test_no_warning_when_no_threshold_is_crossed(client, user):
    db.set_budget(user, "Food", 1000)
    page = add_form_expense(client, 100).get_data(as_text=True)
    assert "now used" not in page and "exceeded" not in page


def test_expense_dated_last_month_does_not_warn(client, user):
    db.set_budget(user, "Food", 100)
    page = add_form_expense(client, 500, day=LAST_MONTH_DAY).get_data(as_text=True)
    assert "exceeded" not in page


def test_other_category_does_not_warn(client, user):
    db.set_budget(user, "Food", 100)
    page = add_form_expense(client, 500, category="Transport").get_data(as_text=True)
    assert "exceeded" not in page


# ---------- AI: warnings, budget questions and setting budgets ----------

def test_ai_add_crossing_budget_shows_warning_after_confirm(client, user, fake_ai):
    db.set_budget(user, "Food", 1000)
    db.add_expense(user, 900, "Food", TODAY.isoformat(), "before")
    fake_ai(tool_call("add_expense_tool", amount=150, category="Food", date=TODAY.isoformat(),
                      description="lunch"), "Ready.")
    _, data = ask_ai(client, "spent 150 on lunch")
    result = client.post("/ai/confirm", json={"action_id": data["cards"][0]["id"]}).get_json()
    assert result["kind"] == "warning"
    assert "✅ Added ₹150" in result["message"]
    assert "🚨 Food budget exceeded by ₹50" in result["message"]


def test_ai_budget_question_answers_from_database(client, user, fake_ai):
    db.set_budget(user, None, 2000)
    db.add_expense(user, 1500, "Food", TODAY.isoformat(), "x")
    fake_ai(tool_call("get_budget_status"), "You have ₹99999 left!")      # AI lies
    status_code, data = ask_ai(client, "how much budget do I have left?")
    assert status_code == 200 and data["cards"] == []
    assert "₹1,500 spent of ₹2,000 (75%), ₹500 left" in data["reply"]
    assert "99999" not in data["reply"]


def test_ai_budget_question_without_tool_call_uses_code_fallback(client, user, fake_ai):
    db.set_budget(user, "Food", 1000)
    fake_ai("You're doing great!", "Still great!")                        # never calls a tool
    _, data = ask_ai(client, "how much budget do I have left?")
    assert "Food budget: ₹0 spent of ₹1,000" in data["reply"]


def test_ai_set_budget_needs_confirm(client, user, fake_ai):
    db.set_budget(user, "Food", 800)
    fake_ai(tool_call("set_budget_tool", amount=1000, category="food"), "Ready for review.")
    _, data = ask_ai(client, "set my food budget to 1000")
    card = data["cards"][0]
    assert card == {"id": card["id"], "type": "budget", "name": "Food budget",
                    "before": "₹800", "after": "₹1,000"}
    assert db.get_budget(user, "Food") == 800.0                          # not saved yet

    result = client.post("/ai/confirm", json={"action_id": card["id"]}).get_json()
    assert result["message"] == "💰 Food budget set to ₹1,000 (was ₹800)"
    assert db.get_budget(user, "Food") == 1000.0


def test_ai_set_budget_cancel_changes_nothing(client, user, fake_ai):
    fake_ai(tool_call("set_budget_tool", amount=20000), "Ready.")
    _, data = ask_ai(client, "set my monthly budget to 20000")
    client.post("/ai/cancel", json={"action_id": data["cards"][0]["id"]})
    assert db.get_budget(user, None) is None


def test_ai_cannot_invent_a_budget_amount(client, user, fake_ai):
    fake_ai(tool_call("set_budget_tool", amount=5000, category="Food"), "Done.", "Done.")
    _, data = ask_ai(client, "set my food budget to 1000")               # AI used 5000
    assert data["cards"] == []
    assert db.get_budget(user, "Food") is None


@pytest.mark.parametrize("message", ["set my food budget to 0", "set my food budget to -500"])
def test_zero_or_negative_budget_rejected_before_ai(client, user, ai_must_not_be_called, message):
    status_code, data = ask_ai(client, message)
    assert status_code == 400 and "greater than 0" in data["error"]
    assert ai_must_not_be_called == []


def test_ai_set_budget_only_for_logged_in_user(client, user, other_user, fake_ai):
    db.set_budget(other_user, "Food", 500)
    fake_ai(tool_call("set_budget_tool", amount=1000, category="Food"), "Ready.")
    _, data = ask_ai(client, "set my food budget to 1000")
    assert data["cards"][0]["before"] == "not set"                       # Ravi's ₹500 isn't seen
    client.post("/ai/confirm", json={"action_id": data["cards"][0]["id"]})
    assert db.get_budget(user, "Food") == 1000.0
    assert db.get_budget(other_user, "Food") == 500.0


def test_ai_set_budget_detects_change_since_preview(client, user, fake_ai):
    fake_ai(tool_call("set_budget_tool", amount=1000, category="Food"), "Ready.")
    _, data = ask_ai(client, "set my food budget to 1000")
    db.set_budget(user, "Food", 700)                                      # changed in another tab
    response = client.post("/ai/confirm", json={"action_id": data["cards"][0]["id"]})
    assert response.status_code == 400
    assert db.get_budget(user, "Food") == 700.0


def test_question_cannot_set_a_budget(client, user, fake_ai):
    fake_ai(tool_call("set_budget_tool", amount=1000, category="Food"), "Done.")
    _, data = ask_ai(client, "how much budget do I have left?")
    assert data["cards"] == []
    assert db.get_budget(user, "Food") is None
