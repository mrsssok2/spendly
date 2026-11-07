"""Regression tests for question routing (from the first full eval run, 2026-11-07).

- totals: "how much ...?" got a LIST of rows (find_expenses with a description the AI made up);
- periods: "today" / a misspelled "this month" were dropped and all-time numbers shown;
- SQL: count / weekday / misspelled-average questions ended in "I couldn't answer that safely."
The real model is never called: `fake_ai` scripts it, `ollama_down` forbids it."""
from datetime import date, timedelta

import pytest

import ai_agent
import database.db as db
from conftest import ask_ai, tool_call

TODAY = date.today()
LONG_AGO = TODAY - timedelta(days=400)


def add(user_id, amount, category, day, description):
    db.add_expense(user_id, amount, category, str(day), description)


# ---------- Typos, fixed in code ----------

@pytest.mark.parametrize("text, fixed", [
    ("how much did i spnd this mnth?", "how much did i spend this month?"),
    ("avrage expense amout", "average expense amount"),
    ("averge spend ths wk", "average spend this week"),
    ("chai 20 todya", "chai 20 today"),
    ("yesterdy spending", "yesterday spending"),
    ("Spent 250 on auto", "Spent 250 on auto"),          # nothing to fix
])
def test_fix_typos(text, fixed):
    assert ai_agent.fix_typos(text) == fixed


@pytest.mark.parametrize("text, label", [
    ("how much did i spnd this mnth?", "this month"),
    ("spending todya", "today"),
    ("how much yesterdy", "yesterday"),
    ("spent on toddy today", "today"),                  # "toddy" is a drink, not a typo
])
def test_period_read_through_typos(text, label):
    assert ai_agent.detect_period(text, date(2026, 11, 7))[2] == label


@pytest.mark.parametrize("text", ["how much this mnoth?", "how much tooday?", "how much this weak?"])
def test_unknown_time_word_is_unclear_not_all_time(text):
    with pytest.raises(ai_agent.UnclearPeriod):
        ai_agent.detect_period(text, date(2026, 11, 7))


def test_unclear_time_word_asks_instead_of_using_all_data(client, user, ollama_down):
    add(user, 500, "Food", LONG_AGO, "dinner")
    status, data = ask_ai(client, "how much did I spend this mnoth?")
    assert status == 200
    assert data["reply"].startswith('Which period do you mean by "this mnoth"?')
    assert "₹" not in data["reply"]


# ---------- "How much ...?" is the SUM for the period, never a list ----------

def test_total_question_answered_with_list_gets_the_sum(client, user, fake_ai):
    add(user, 280, "Transport", TODAY, "auto")
    add(user, 80, "Food", TODAY, "chai")
    add(user, 999, "Food", LONG_AGO, "chai")
    # Real qwen2.5: a list, then a search for 'chai' (a word the user never typed).
    fake_ai(tool_call("find_expenses", period="today", limit=5),
            tool_call("find_expenses", description="chai", category="Food"),
            "Here are your expenses.")
    status, data = ask_ai(client, "how much did I spend today?")
    assert status == 200
    assert data["reply"].startswith("You spent ₹360 today across 2 expenses.")
    assert "•" not in data["reply"]                                 # not a list of rows


def test_total_without_question_word_is_a_sum(client, user, fake_ai):
    add(user, 280, "Transport", TODAY, "cab")
    add(user, 500, "Transport", TODAY, "petrol")
    fake_ai(tool_call("find_expenses", category="Transport", period="this month"), "Here they are.")
    status, data = ask_ai(client, "transport spending this month")
    assert status == 200
    assert data["reply"] == "You spent ₹780 on Transport this month across 2 expenses."


def test_total_for_one_day_ignores_the_ais_limit(client, user, fake_ai):
    day = TODAY - timedelta(days=2)
    add(user, 60, "Transport", day, "bus")
    add(user, 1200, "Bills", day, "electricity")
    fake_ai(tool_call("find_expenses", date_from=str(day), date_to=str(day), limit=1), "Done.")
    status, data = ask_ai(client, "how much did I spend day before yesterday?")
    assert data["reply"].startswith("You spent ₹1,260 day before yesterday across 2 expenses.")


def test_how_much_on_an_item_the_user_named_is_still_a_search(client, user, fake_ai):
    add(user, 80, "Food", TODAY, "chai")
    add(user, 300, "Food", TODAY, "lunch")
    fake_ai(tool_call("find_expenses", description="chai"), "You spent ₹80 on chai.")
    status, data = ask_ai(client, "how much did I spend on chai?")
    assert "₹80" in data["reply"] and "₹380" not in data["reply"]


# ---------- The user's period always wins; an empty period says so ----------

def test_period_is_never_dropped_for_an_empty_day(client, user, fake_ai):
    add(user, 1000, "Shopping", LONG_AGO, "shirt")
    # Real qwen2.5: an unknown `date=` argument (ignored) and then no period at all.
    fake_ai(tool_call("find_expenses", category="Shopping", date=str(TODAY), limit=1),
            tool_call("find_expenses", description="shirt", category="Shopping"),
            "You bought a shirt.")
    status, data = ask_ai(client, "how much did I spend on shopping today?")
    assert status == 200
    assert data["reply"] == "You have no Shopping expenses today."
    assert "1,000" not in data["reply"]


def test_misspelled_month_is_this_month_not_all_time(client, user, fake_ai):
    add(user, 300, "Food", TODAY, "lunch")
    add(user, 9000, "Bills", LONG_AGO, "rent")
    fake_ai(tool_call("get_spending_summary", period="this month"),
            tool_call("get_spending_summary", period=""),     # the AI drops the period
            "Done.")
    status, data = ask_ai(client, "how much did i spnd this mnth?")
    assert data["reply"] == "You spent ₹300 this month across 1 expense."


def test_ais_period_cannot_replace_the_users(client, user, fake_ai):
    add(user, 300, "Food", TODAY, "lunch")
    add(user, 9000, "Bills", LONG_AGO, "rent")
    fake_ai(tool_call("get_spending_summary", period="all time"), "Done.")
    status, data = ask_ai(client, "how much did I spend today?")
    assert data["reply"] == "You spent ₹300 today across 1 expense."


# ---------- Count template ----------

def test_count_question_uses_the_count_template_even_after_bad_sql(client, user, fake_ai):
    add(user, 80, "Food", TODAY, "chai")
    add(user, 280, "Transport", TODAY, "auto")
    add(user, 500, "Bills", LONG_AGO, "phone")
    # Real qwen2.5: date literals in the SQL (blocked). The count template answers instead.
    fake_ai(tool_call("run_sql_query", sql="SELECT COUNT(*) FROM my_expenses WHERE date >= date('now')",
                      period="this week", measure=""),
            "You have expenses.")
    status, data = ask_ai(client, "how many expenses this week?")
    assert status == 200
    assert data["reply"].startswith("Number of expenses this week: 2")


@pytest.mark.parametrize("question, sql, expected", [
    ("how many expenses of 280 or more?", "SELECT COUNT(*) AS n FROM my_expenses WHERE amount >= 280", "2"),
    ("how many expenses between 100 and 500?",
     "SELECT COUNT(*) AS n FROM my_expenses WHERE amount >= 100 AND amount <= 500", "2"),
    ("how many expenses of exactly 280?", "SELECT COUNT(*) AS n FROM my_expenses WHERE amount = 280", "1"),
])
def test_count_with_an_amount_condition_uses_the_ais_sql_not_the_template(client, user, fake_ai,
                                                                           question, sql, expected):
    """Regression (Groq + qwen2.5 evals): the right SQL was replaced by the plain count template."""
    add(user, 80, "Food", TODAY, "chai")
    add(user, 280, "Transport", TODAY, "auto")
    add(user, 500, "Bills", LONG_AGO, "phone")
    fake_ai(tool_call("run_sql_query", sql=sql, period="", measure=""), f"You have {expected}.")
    status, data = ask_ai(client, question)
    assert status == 200
    assert "Number of expenses: 3" not in data["reply"]
    assert expected in data["reply"]


def test_count_question_without_any_sql_runs_the_template(client, user, fake_ai):
    add(user, 80, "Food", TODAY, "chai")
    fake_ai("You have a few expenses.")
    status, data = ask_ai(client, "how many expenses this week?")
    assert data["reply"].startswith("Number of expenses this week: 1")


@pytest.mark.parametrize("text, measure", [
    ("how many expenses this week?", "count"),
    ("how many food expenses today", "count"),
    ("number of transactions in october", "count"),
    ("how many expenses above 250", None),             # a threshold: free-form SQL
    # Regression (evals, both qwen2.5 and gpt-oss): these got the plain count template (20).
    ("how many expenses of 280 or more?", None),
    ("how many expenses of ₹280 or less", None),
    ("how many expenses between 100 and 500?", None),
    ("how many expenses between ₹100 to ₹500", None),
    ("how many expenses of exactly 280?", None),
    ("how many transactions of 280", None),
    ("how many expenses of 5 days", "count"),          # a time span, not an amount
    ("how many expenses in each category", None),      # grouped: free-form SQL
    ("avrage expense amout", "average_per_expense"),
    ("averge spend per day", "average_per_day"),
])
def test_template_measure(text, measure):
    assert ai_agent.template_measure(ai_agent.fix_typos(text)) == measure


# ---------- Misspelled "average" ----------

def test_misspelled_average_gets_the_average_template(client, user, fake_ai):
    add(user, 100, "Food", TODAY, "lunch")
    add(user, 300, "Food", LONG_AGO, "dinner")
    # Real qwen2.5 picked average_per_day for "today" (the user named no period).
    fake_ai(tool_call("run_sql_query", measure="average_per_day", period="today"), "Done.")
    status, data = ask_ai(client, "avrage expense amout")
    assert status == 200
    assert "₹200.00" in data["reply"] and "2 expenses" in data["reply"]


# ---------- Spending by day of the week (SQL, all rows, ties shown) ----------
# 2026-11-07 is a Saturday, 2026-11-01 a Sunday, 2026-11-02 a Monday.

def test_weekday_most_with_a_tie(client, user, ollama_down):
    add(user, 300, "Food", "2026-11-07", "lunch")
    add(user, 200, "Food", "2026-10-31", "dinner")      # also a Saturday: 500 in total
    add(user, 500, "Shopping", "2026-11-01", "shirt")   # Sunday: 500
    add(user, 100, "Transport", "2026-11-02", "auto")   # Monday: 100
    status, data = ask_ai(client, "which day of the week do I spend the most?")
    assert status == 200
    reply = data["reply"]
    assert reply.startswith("You spend the most on Sunday and Saturday (a tie: ₹500.00 each).")
    assert "- Monday: ₹100.00 (1 expense)" in reply
    assert "- Saturday: ₹500.00 (2 expenses)" in reply


def test_weekday_least_and_category(client, user, ollama_down):
    add(user, 300, "Food", "2026-11-07", "lunch")
    add(user, 50, "Food", "2026-11-02", "chai")
    add(user, 10, "Transport", "2026-11-03", "bus")     # not Food
    status, data = ask_ai(client, "which day of the week do I spend the least on food?")
    assert data["reply"].startswith("You spend the least on Food on Monday (₹50.00).")
    assert "Tuesday" not in data["reply"]


def test_weekday_with_no_expenses_says_so(client, user, ollama_down):
    status, data = ask_ai(client, "which day of the week do I spend the most?")
    assert data["reply"] == "You have no expenses."


def test_weekday_sql_is_parameterized_and_per_user(user, other_user):
    db.add_expense(user, 100, "Food", "2026-11-07", "lunch")
    db.add_expense(other_user, 77777, "Food", "2026-11-07", "secret")
    rows = db.get_spending_by_weekday(user)
    assert rows == [{"weekday": 6, "count": 1, "total": 100.0}]
