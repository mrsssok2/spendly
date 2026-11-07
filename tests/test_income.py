"""Income tracking and savings rate: table + queries, add/edit/delete routes, profile cards,
analytics chart, AI assistant (add income needs Confirm, savings questions) and AI insights."""
import json
import re
import sqlite3
from datetime import date, timedelta

import pytest

import ai_agent
import ai_insights
import app as spendly
import database.db as db
from conftest import ask_ai, tool_call

TODAY = date.today()
THIS_START = TODAY.replace(day=1)
LAST_MONTH_DAY = (THIS_START - timedelta(days=1)).isoformat()


def iso(d):
    return d.isoformat()


def income_count(user_id):
    return len(db.get_income_for_user(user_id))


def add_income_form(client, amount="30000", source="Salary", day=None, description="November salary"):
    return client.post("/income/add", data={"amount": amount, "source": source,
                                            "date": day or iso(TODAY), "description": description})


# ---------- Table and queries ----------

def test_database_rejects_zero_or_negative_income(user):
    for bad in (0, -500):
        with pytest.raises(sqlite3.IntegrityError):
            db.add_income(user, bad, "Salary", iso(TODAY), "x")


def test_money_summary_is_calculated_by_sql(user, other_user):
    db.add_income(user, 30000, "Salary", iso(TODAY), "salary")
    db.add_expense(user, 7000, "Bills", iso(TODAY), "rent")
    db.add_expense(user, 5000, "Food", iso(TODAY), "groceries")
    db.add_income(other_user, 99999, "Salary", iso(TODAY), "Ravi's")          # not this user
    assert db.get_money_summary(user) == {"income": 30000.0, "income_count": 1, "expenses": 12000.0,
                                          "expense_count": 2, "balance": 18000.0, "savings_rate": 60.0}


def test_money_summary_negative_balance_and_no_income(user):
    db.add_expense(user, 1500, "Food", iso(TODAY), "x")
    assert db.get_money_summary(user)["savings_rate"] is None                 # no income: can't divide
    db.add_income(user, 1000, "Gift", iso(TODAY), "y")
    s = db.get_money_summary(user)
    assert (s["balance"], s["savings_rate"]) == (-500.0, -50.0)


def test_money_summary_respects_the_date_range(user):
    db.add_income(user, 30000, "Salary", iso(TODAY), "this month")
    db.add_income(user, 20000, "Salary", LAST_MONTH_DAY, "last month")
    assert db.get_money_summary(user, iso(THIS_START), iso(TODAY))["income"] == 30000.0
    assert db.get_money_summary(user)["income"] == 50000.0


# ---------- Add / edit / delete (same rules and security as expenses) ----------

def test_income_pages_require_login(client):
    assert "/login" in client.get("/income/add").headers["Location"]
    assert "/login" in add_income_form(client).headers["Location"]


def test_add_income(client, user):
    assert add_income_form(client).status_code == 302
    row = db.get_income_for_user(user)[0]
    assert (row["amount"], row["source"], row["date"], row["description"]) == \
           (30000.0, "Salary", iso(TODAY), "November salary")


@pytest.mark.parametrize("field, value", [
    ("amount", "0"), ("amount", "-5"), ("amount", "abc"), ("amount", "nan"),
    ("source", "Lottery"), ("source", "Food"),          # an expense category is not an income source
    ("date", "2026-02-30"), ("description", "x" * 201),
])
def test_invalid_income_is_rejected(client, user, field, value):
    data = {"amount": "100", "source": "Salary", "date": iso(TODAY), "description": "ok", field: value}
    assert client.post("/income/add", data=data).status_code == 400
    assert income_count(user) == 0


def test_user_id_comes_from_the_session(client, user, other_user):
    client.post("/income/add", data={"amount": "100", "source": "Gift", "date": iso(TODAY),
                                     "description": "sneaky", "user_id": str(other_user)})
    assert income_count(user) == 1 and income_count(other_user) == 0


def test_edit_and_delete_own_income(client, user):
    add_income_form(client)
    income_id = db.get_income_for_user(user)[0]["id"]
    client.post(f"/income/{income_id}/edit", data={"amount": "32000", "source": "Salary",
                                                   "date": iso(TODAY), "description": "after raise"})
    assert db.get_income_for_user(user)[0]["amount"] == 32000.0
    assert client.post(f"/income/{income_id}/delete").status_code == 302
    assert income_count(user) == 0


def test_cannot_touch_another_users_income(client, user, other_user):
    income_id = db.add_income(other_user, 50000, "Salary", iso(TODAY), "Ravi's salary")
    assert client.get(f"/income/{income_id}/edit").status_code == 404
    assert client.post(f"/income/{income_id}/edit", data={"amount": "1", "source": "Gift",
                                                          "date": iso(TODAY), "description": "x"}).status_code == 404
    assert client.post(f"/income/{income_id}/delete").status_code == 404
    assert db.get_income(income_id, other_user)["amount"] == 50000.0
    assert "Ravi&#39;s salary" not in client.get("/profile").get_data(as_text=True)


# ---------- Profile cards ----------

def card_value(page, label):
    match = re.search(rf'<p class="card-label">{label}</p>\s*<p class="card-value([^"]*)"[^>]*>\s*([^<]+?)\s*</p>', page)
    return match.group(1).strip(), match.group(2)


def test_profile_cards_show_income_balance_and_savings_rate(client, user):
    db.add_income(user, 30000, "Salary", iso(TODAY), "salary")
    db.add_expense(user, 12000, "Bills", iso(TODAY), "rent")
    page = client.get("/profile").get_data(as_text=True)
    assert card_value(page, "Income")[1] == "₹30,000.00"
    assert card_value(page, "Expenses")[1] == "₹12,000.00"
    assert card_value(page, "Balance") == ("money-positive", "₹18,000.00")       # green
    assert card_value(page, "Savings rate") == ("money-positive", "60.0%")
    assert "+₹30,000.00" in page                                                  # income table row


def test_negative_balance_is_red(client, user):
    db.add_income(user, 1000, "Gift", iso(TODAY), "gift")
    db.add_expense(user, 1500, "Food", iso(TODAY), "x")
    page = client.get("/profile").get_data(as_text=True)
    assert card_value(page, "Balance") == ("money-negative", "−₹500.00")
    assert card_value(page, "Savings rate") == ("money-negative", "-50.0%")


def test_negative_balance_stays_on_one_line(client, user):
    db.add_expense(user, 3394, "Food", iso(TODAY), "x")
    page = client.get("/profile").get_data(as_text=True)
    assert card_value(page, "Balance")[1] == "−₹3,394.00"                       # minus right next to ₹
    import os
    css = open(os.path.join(os.path.dirname(spendly.__file__), "static", "css", "profile.css"), encoding="utf-8").read()
    rule = re.search(r"\.money-cards \.card-value\s*\{([^}]*)\}", css)
    assert rule and "white-space: nowrap" in rule.group(1)                     # never wraps after "−"


def test_no_income_shows_a_hint(client, user):
    db.add_expense(user, 100, "Food", iso(TODAY), "x")
    page = client.get("/profile").get_data(as_text=True)
    assert card_value(page, "Savings rate")[1] == "—" and "Add income to see it" in page


def test_cards_follow_the_date_filter(client, user):
    db.add_income(user, 30000, "Salary", iso(TODAY), "this month")
    db.add_income(user, 20000, "Salary", LAST_MONTH_DAY, "last month")
    db.add_expense(user, 3000, "Food", iso(TODAY), "x")
    page = client.get(f"/profile?date_from={iso(THIS_START)}&date_to={iso(TODAY)}").get_data(as_text=True)
    assert card_value(page, "Income")[1] == "₹30,000.00"
    assert card_value(page, "Savings rate")[1] == "90.0%"
    assert "last month" not in page                                              # filtered out of the table


# ---------- Analytics ----------

def test_analytics_income_vs_expenses_chart(client, user):
    db.add_income(user, 30000, "Salary", iso(TODAY), "salary")
    db.add_expense(user, 12000, "Bills", iso(TODAY), "rent")
    page = client.get("/analytics").get_data(as_text=True)
    data = json.loads(re.search(r'id="analytics-data">(.*?)</script>', page, re.DOTALL).group(1))
    chart = data["income_vs_expenses"]
    assert chart["income"][-1] == 30000.0 and chart["expenses"][-1] == 12000.0
    assert chart["has_income"] is True and 'id="chart-income"' in page


# ---------- AI assistant ----------

def test_ai_adds_income_only_after_confirm(client, user, fake_ai):
    fake_ai(tool_call("add_income_tool", amount=30000, source="salary", date=iso(TODAY),
                      description="November salary"), "Ready for review.")
    status, data = ask_ai(client, "got salary 30000 today")
    card = data["cards"][0]
    assert status == 200 and card["type"] == "add_income"
    assert card["rows"][0]["amount"] == "₹30,000" and card["rows"][0]["source"] == "Salary"
    assert income_count(user) == 0                                               # not saved yet

    result = client.post("/ai/confirm", json={"action_id": card["id"]}).get_json()
    assert result["message"].startswith("💵 Added income ₹30,000 · Salary")
    assert income_count(user) == 1
    assert db.get_expense_summary(user)["count"] == 0                            # NOT an expense


def test_ai_income_cancel_changes_nothing(client, user, fake_ai):
    fake_ai(tool_call("add_income_tool", amount=5000, source="Freelance", date=iso(TODAY)), "Ready.")
    _, data = ask_ai(client, "received 5000 for freelance work")
    client.post("/ai/cancel", json={"action_id": data["cards"][0]["id"]})
    assert income_count(user) == 0


def test_ai_cannot_invent_an_income_amount(client, user, fake_ai):
    fake_ai(tool_call("add_income_tool", amount=300000, source="Salary", date=iso(TODAY)), "Done.", "Done.")
    _, data = ask_ai(client, "got salary 30000 today")
    assert data["cards"] == [] and income_count(user) == 0


# The AI asked "how would you like to describe it?" instead of showing the preview card.
# The description is optional: code prepares the preview itself (the preview IS the confirm step).
ASKS_FOR_DESCRIPTION = ("This seems to be income. Would you like to record this? "
                        "If so, how would you like to describe it?")


def test_mummy_ne_2000_diye_shows_gift_income_preview(client, user, fake_ai):
    fake_ai("Mom gave me 2000",                                          # the translator
            ASKS_FOR_DESCRIPTION)                                        # the agent: no tool call
    status, data = ask_ai(client, "mummy ne 2000 diye")
    assert status == 200
    assert data["understood_as"] == '🌐 Understood as: "Mom gave me 2000"'
    card = data["cards"][0]
    assert card["type"] == "add_income"
    assert card["rows"][0]["amount"] == "₹2,000" and card["rows"][0]["source"] == "Gift"
    assert data["reply"].startswith("Add this income? ₹2,000 · Gift")
    assert "describe" not in data["reply"]
    assert income_count(user) == 0                                               # needs Confirm

    client.post("/ai/confirm", json={"action_id": card["id"]})
    [row] = db.get_income_for_user(user)
    assert (row["amount"], row["source"], row["date"], row["description"]) == (
        2000, "Gift", iso(TODAY), "from mom")
    assert db.get_expense_summary(user)["count"] == 0                            # NOT an expense


def test_kal_freelance_se_5000_mile_shows_freelance_income_for_yesterday(client, user, fake_ai):
    yesterday = iso(TODAY - timedelta(days=1))
    fake_ai("Received 5000 from freelance yesterday", ASKS_FOR_DESCRIPTION)
    status, data = ask_ai(client, "kal freelance se 5000 mile")
    assert status == 200
    card = data["cards"][0]
    assert card["type"] == "add_income"
    assert card["rows"][0]["amount"] == "₹5,000" and card["rows"][0]["source"] == "Freelance"
    assert data["reply"].startswith("Add this income? ₹5,000 · Freelance")
    assert income_count(user) == 0

    client.post("/ai/confirm", json={"action_id": card["id"]})
    [row] = db.get_income_for_user(user)
    assert (row["amount"], row["source"], row["date"], row["description"]) == (
        5000, "Freelance", yesterday, "freelance")


@pytest.mark.parametrize("english, original, expected", [
    ("Mom gave me 2000", "mummy ne 2000 diye", ("Gift", "from mom")),
    ("Received 5000 from freelance yesterday", "kal freelance se 5000 mile", ("Freelance", "freelance")),
    ("got salary 30000 today", "got salary 30000 today", ("Salary", "salary")),
    ("My friend paid me back 500", "My friend paid me back 500", None),  # friend + refund: unclear
    ("Got a refund of 499", "Got a refund of 499", ("Refund", "refund")),
    ("Papa sent me 1000", "Papa sent me 1000", ("Gift", "from dad")),
])
def test_income_source_mapping(english, original, expected):
    result = ai_agent.income_from_message(english, original, TODAY)
    assert (result and (result["source"], result["description"])) == expected


@pytest.mark.parametrize("english, original", [
    ("Gave 2000 to mom", "mummy ko 2000 diye"),          # money going OUT
    ("I gave mom 2000", "I gave mom 2000"),
    ("Mom gave me money", "Mom gave me money"),          # no amount: the AI may ask
    ("Mom gave me 2000 and 500", "Mom gave me 2000 and 500"),   # two amounts: unclear
    ("Spent 250 on auto", "Spent 250 on auto"),          # an expense
])
def test_unclear_or_outgoing_money_is_not_auto_income(english, original):
    assert ai_agent.income_from_message(english, original, TODAY) is None


def test_ai_may_still_ask_when_the_amount_is_missing(client, user, fake_ai):
    fake_ai("How much did your mom give you?")
    _, data = ask_ai(client, "mom gave me some money")
    assert data["cards"] == [] and data["reply"] == "How much did your mom give you?"


@pytest.mark.parametrize("message", ["got salary -30000 today", "got salary 0 today"])
def test_zero_or_negative_income_rejected_before_ai(client, user, ai_must_not_be_called, message):
    status, data = ask_ai(client, message)
    assert status == 400 and "greater than 0" in data["error"]


def test_a_question_cannot_add_income(client, user, fake_ai):
    fake_ai(tool_call("add_income_tool", amount=30000, source="Salary", date=iso(TODAY)), "Done.")
    _, data = ask_ai(client, "how much did I save this month?")
    assert data["cards"] == [] and income_count(user) == 0


def test_savings_question_answered_from_the_database(client, user, fake_ai):
    db.add_income(user, 30000, "Salary", iso(TODAY), "salary")
    db.add_expense(user, 12000, "Bills", iso(TODAY), "rent")
    fake_ai(tool_call("get_savings_summary", period="this month"), "You saved a lot!")
    _, data = ask_ai(client, "how much did I save this month?")
    assert data["reply"] == ("You earned ₹30,000 and spent ₹12,000 this month, so you saved ₹18,000: "
                             "a savings rate of 60%.")


def test_savings_question_without_tool_call_uses_code_fallback(client, user, fake_ai):
    db.add_income(user, 30000, "Salary", iso(TODAY), "salary")
    db.add_expense(user, 12000, "Bills", iso(TODAY), "rent")
    fake_ai("Great job!", "Still great!")                                        # never calls a tool
    _, data = ask_ai(client, "what's my savings rate this month?")
    assert "savings rate of 60%" in data["reply"]
    assert "You spent ₹12,000 this month across" not in data["reply"]           # not the spending answer


def test_savings_question_with_no_income(client, user, fake_ai):
    db.add_expense(user, 500, "Food", iso(TODAY), "x")
    fake_ai(tool_call("get_savings_summary"), "?")
    _, data = ask_ai(client, "what's my savings rate?")
    assert "haven't recorded any income" in data["reply"] and "You spent ₹500" in data["reply"]


def test_savings_question_when_overspent(client, user, fake_ai):
    db.add_income(user, 1000, "Gift", iso(TODAY), "gift")
    db.add_expense(user, 1500, "Food", iso(TODAY), "x")
    fake_ai(tool_call("get_savings_summary"), "?")
    _, data = ask_ai(client, "what's my balance?")
    assert "₹500 more than you earned" in data["reply"]


# ---------- AI insights ----------

def insights_state(user):
    state = ai_insights.fetch_data({"user_id": user, "today": TODAY})
    return ai_insights.analyse(state)["facts"]


def test_insights_facts_include_the_savings_rate(user, other_user):
    db.add_income(user, 30000, "Salary", iso(THIS_START), "salary")
    db.add_income(other_user, 99999, "Salary", iso(THIS_START), "Ravi's")
    for amount in (6000, 4000, 2000):
        db.add_expense(user, amount, "Food", iso(THIS_START), "groceries")
    facts = insights_state(user)
    text = "\n".join(facts["lines"])
    assert "[income] Income this month so far: ₹30,000." in text
    assert "[savings] Savings so far (income minus spending): ₹18,000." in text
    assert "[savings_rate] Savings rate: 60% of income." in text
    assert "savings" in facts["amount_labels"][18000.0] and "savings_rate_pct" in facts["percent_labels"][60.0]
    assert {"label": "Savings rate", "value": "60%"} in facts["key_numbers"]
    assert "savings" in facts["names"] and 18000.0 in facts["allowed_amounts"]
    assert "99,999" not in text


def test_insights_fallback_mentions_savings_and_gives_a_tip_when_low(user):
    db.add_income(user, 10000, "Salary", iso(THIS_START), "salary")
    for amount in (5000, 3000, 1500):
        db.add_expense(user, amount, "Food", iso(THIS_START), "groceries")
    facts = insights_state(user)
    plain = ai_insights.plain_summary(facts)
    assert "You've saved ₹500 of your ₹10,000 income so far (5%)." in plain["summary"]
    assert any("Your savings so far are ₹500 (5% of income)" in tip for tip in plain["tips"])


def test_insights_without_income_say_so(user):
    for amount in (100, 200, 300):
        db.add_expense(user, amount, "Food", iso(THIS_START), "x")
    facts = insights_state(user)
    text = "\n".join(facts["lines"]).lower()
    assert "income" not in text and "saving" not in text       # no income: no income/savings facts at all
    assert all(n["label"] != "Savings rate" for n in facts["key_numbers"])
