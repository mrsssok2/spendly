"""AI insights workflow (ai_insights.py). The real Ollama model is never called: a scripted
fake plays the model. Checks: numbers come from code, the AI's ₹ amounts are verified,
retry then fallback, not-enough-data, Ollama down, logged-in user only, and read-only."""
import json
from datetime import date, timedelta

import httpx
import pytest
from langchain_core.messages import AIMessage

import ai_insights
import database.db as db
from conftest import FakeToolModel

TODAY = date.today()
THIS_START = TODAY.replace(day=1).isoformat()                 # always within this month, <= today
LAST_START = (TODAY.replace(day=1) - timedelta(days=1)).replace(day=1).isoformat()

GOOD = {"summary": "You've spent ₹1,300 so far this month, ₹300 more than last month. "
                   "Your Food budget has ₹200 left (83.3% used).",
        "tips": ["Your groceries (₹600) were your biggest expense — buy staples in bulk.",
                 "Your Food budget has only ₹200 left — cook at home for the rest of the month.",
                 "Your auto ride (₹300) was your only Transport cost — try sharing rides."]}
INVENTED = {"summary": "You spent ₹5,000 this month!", "tips": ["a", "b", "c"]}
GENERIC = {"summary": "You've spent ₹1,300 so far this month.",
           "tips": ["Plan your meals to save money.", "Use public transport.", "Track your spending."]}


def sample_data(user, other_user=None):
    """This month ₹1,300 (Food ₹1,000, Transport ₹300); last month ₹1,000; Food budget ₹1,200."""
    db.add_expense(user, 600, "Food", THIS_START, "groceries")
    db.add_expense(user, 400, "Food", THIS_START, "dinner")
    db.add_expense(user, 300, "Transport", THIS_START, "auto")
    db.add_expense(user, 500, "Food", LAST_START, "groceries")
    db.add_expense(user, 500, "Bills", LAST_START, "electricity")
    db.set_budget(user, "Food", 1200)
    if other_user:
        db.add_expense(other_user, 77777, "Food", THIS_START, "Ravi's party")


@pytest.fixture
def fake_insights_ai(monkeypatch):
    """Script the insights model's replies (dicts become JSON). Returns a call counter."""
    calls = []

    def _script(*replies):
        messages = iter([AIMessage(content=r if isinstance(r, str) else json.dumps(r)) for r in replies])

        def _get_llm(*args, **kwargs):
            calls.append(1)
            return FakeToolModel(messages=messages)
        monkeypatch.setattr(ai_insights, "get_chat_model", _get_llm)
        return calls
    return _script


def get_insights(client):
    response = client.post("/analytics/insights")
    return response.status_code, response.get_json()


def snapshot():
    conn = db.get_db()
    try:
        return (conn.execute("SELECT * FROM expenses ORDER BY id").fetchall(),
                conn.execute("SELECT * FROM budgets ORDER BY id").fetchall())
    finally:
        conn.close()


# ---------- The route ----------

def test_insights_require_login(client):
    assert client.post("/analytics/insights").status_code == 401


def test_good_ai_output_is_shown(client, user, fake_insights_ai):
    sample_data(user)
    calls = fake_insights_ai(GOOD)
    status, data = get_insights(client)
    assert status == 200
    assert data["status"] == "ok" and data["source"] == "ai"
    assert data["summary"] == GOOD["summary"]
    assert data["tips"] == GOOD["tips"]
    assert {n["label"]: n["value"] for n in data["key_numbers"]} == {
        "This month so far": "₹1,300", "Last month": "₹1,000", "vs same days last month": "▲ 30.0%",
        "Biggest category": "Food ₹1,000"}
    assert len(calls) == 1


def test_invented_amount_triggers_one_retry(client, user, fake_insights_ai):
    sample_data(user)
    calls = fake_insights_ai(INVENTED, GOOD)
    _, data = get_insights(client)
    assert data["source"] == "ai" and data["summary"] == GOOD["summary"]
    assert len(calls) == 2                                   # wrote, rejected, rewrote


def test_invented_amounts_twice_falls_back_to_plain_summary(client, user, fake_insights_ai):
    sample_data(user)
    calls = fake_insights_ai(INVENTED, INVENTED, GOOD)       # third reply must never be used
    _, data = get_insights(client)
    assert data["status"] == "ok" and data["source"] == "fallback"
    assert "₹5,000" not in data["summary"]
    assert "₹1,300" in data["summary"] and "Food" in data["summary"]
    assert len(data["tips"]) == 3
    assert len(calls) == 2                                   # only one retry


def test_generic_tips_are_rejected_then_retried(client, user, fake_insights_ai):
    sample_data(user)
    calls = fake_insights_ai(GENERIC, GOOD)
    _, data = get_insights(client)
    assert data["source"] == "ai" and data["tips"] == GOOD["tips"]
    assert len(calls) == 2


def test_generic_tips_twice_fall_back_to_specific_tips(client, user, fake_insights_ai):
    sample_data(user)
    fake_insights_ai(GENERIC, GENERIC)
    _, data = get_insights(client)
    assert data["source"] == "fallback"
    names = ["groceries", "dinner", "auto", "Food", "Transport"]
    for tip in data["tips"]:
        assert "₹" in tip and any(n in tip for n in names), tip   # specific, not generic
    assert "Food budget has only ₹200 left (83.3% used)" in data["tips"][0]   # most urgent first
    assert "Your groceries (₹600) was your biggest expense" in data["tips"][1]
    assert "83.3% used, ₹200 left" in data["summary"]


def test_facts_include_top_three_expenses_and_budget_status(user):
    sample_data(user)
    db.add_expense(user, 50, "Food", THIS_START, "chai")                    # 4th: not in the top 3
    state = ai_insights.fetch_data({"user_id": user, "today": TODAY})
    facts = ai_insights.analyse(state)["facts"]
    text = "\n".join(facts["lines"])
    nice_date = date.fromisoformat(THIS_START).strftime("%d %b %Y")
    assert (f"Biggest single expense this month (one purchase, not a category): groceries ₹600 "
            f"(Food category, {nice_date}).") in text
    assert (f"Top single expenses this month: 1. groceries ₹600 (Food, {nice_date}); "
            f"2. dinner ₹400 (Food, {nice_date}); 3. auto ₹300 (Transport, {nice_date}).") in text
    assert "chai" not in text
    assert "Food budget is NEAR its limit: ₹1,050 spent of ₹1,200, 87.5% used, ₹150 left." in text
    assert {"groceries", "dinner", "auto", "Food", "Transport"} <= set(facts["names"])


@pytest.mark.parametrize("bad", ["not json at all", {"summary": "ok"},
                                 {"summary": "ok", "tips": ["only one"]}, {"tips": ["a", "b", "c"]}])
def test_malformed_ai_output_is_never_shown(client, user, fake_insights_ai, bad):
    sample_data(user)
    fake_insights_ai(bad, bad)
    _, data = get_insights(client)
    assert data["source"] == "fallback" and len(data["tips"]) == 3


def test_not_enough_data_says_so_without_calling_the_ai(client, user, monkeypatch):
    db.add_expense(user, 100, "Food", THIS_START, "only one")
    monkeypatch.setattr(ai_insights, "get_chat_model", lambda *a, **k: pytest.fail("AI must not be called"))
    status, data = get_insights(client)
    assert status == 200 and data["status"] == "not_enough"
    assert "Not enough data yet" in data["message"] and "1 expense" in data["message"]


def test_ollama_down_shows_insights_unavailable(client, user, monkeypatch):
    sample_data(user)

    class Down(FakeToolModel):
        def _generate(self, *args, **kwargs):
            raise httpx.ConnectError("refused")
    monkeypatch.setattr(ai_insights, "get_chat_model", lambda *a, **k: Down(messages=iter([])))
    status, data = get_insights(client)
    assert status == 503
    assert data["error"] == "Insights unavailable" and data["status"] == "unavailable"
    assert client.get("/analytics").status_code == 200       # the page still works


def test_insights_are_read_only(client, user, fake_insights_ai):
    sample_data(user)
    before = snapshot()
    fake_insights_ai(INVENTED, INVENTED)
    get_insights(client)
    assert snapshot() == before


def test_only_the_logged_in_users_data_is_used(user, other_user, fake_insights_ai):
    sample_data(user, other_user)
    state = ai_insights.fetch_data({"user_id": user, "today": TODAY})
    facts = ai_insights.analyse(state)["facts"]
    assert "77,777" not in "\n".join(facts["lines"])
    assert facts["this_total"] == 1300.0


def test_analytics_page_has_the_insights_button(client, user):
    db.add_expense(user, 100, "Food", THIS_START, "x")
    page = client.get("/analytics").get_data(as_text=True)
    assert 'id="insights-button"' in page and "insights.js" in page


# ---------- The nodes ----------

def test_analyse_computes_the_numbers_in_code(user):
    state = {"data": {
        "this_month": {"total": 3000.0, "count": 4}, "last_month": {"total": 2000.0, "count": 3},
        "this_categories": [{"category": "Shopping", "total": 2400.0, "count": 1},
                            {"category": "Food", "total": 600.0, "count": 3}],
        "last_categories": [{"category": "Food", "total": 800.0, "count": 3}],
        "top_expenses": [{"amount": 2400.0, "category": "Shopping", "date": THIS_START, "description": "phone"},
                         {"amount": 300.0, "category": "Food", "date": THIS_START, "description": None}],
        "usual": {"average": 400.0, "count": 9},
        "budgets": [{"category": None, "monthly_limit": 2800.0, "spent": 3000.0, "percent": 107.1, "remaining": -200.0},
                    {"category": "Food", "monthly_limit": 700.0, "spent": 600.0, "percent": 85.7, "remaining": 100.0}],
        "this_label": "This Month", "last_label": "Last Month", "day_of_month": 10, "days_in_month": 30,
    }}
    facts = ai_insights.analyse(state)["facts"]
    text = "\n".join(facts["lines"])
    assert facts["change"] == 50.0                                      # (3000 - 2000) / 2000
    assert "50.0% more (₹1,000 more)" in text            # no same-days data given: falls back to the months
    assert "Biggest category this month (total of all its expenses): Shopping category, ₹2,400 (80.0%" in text
    assert "Biggest single expense this month (one purchase, not a category): phone ₹2,400" in text
    assert "2. Food payment ₹300" in text                      # no description: "payment", not "expense"
    assert facts["biggest"] == {"name": "phone", "amount": 2400.0}
    assert "Food: ₹600 this month so far (nothing in the same days of last month)" in text
    assert "Overall monthly budget is OVER its limit" in text and "over by ₹200" in text
    assert "Food budget is NEAR its limit" in text
    assert "Unusually large expense: ₹2,400" in text and "6.0x" in text  # 2400 / 400
    assert text.count("Unusually large expense") == 1                  # ₹300 is not unusual
    assert 1000.0 in facts["allowed_amounts"] and 2400.0 in facts["allowed_amounts"]


def test_output_check_rules():
    allowed = [1300.0, 300.0]
    problem = ai_insights._problem_with
    assert problem({"summary": "₹1,300 spent", "tips": ["a", "b", "c"]}, allowed) is None
    assert "₹999" in problem({"summary": "₹999 spent", "tips": ["a", "b", "c"]}, allowed)
    assert "₹50" in problem({"summary": "ok", "tips": ["save Rs 50", "b", "c"]}, allowed)
    assert "3 tips" in problem({"summary": "ok", "tips": ["a", "b"]}, allowed)
    assert "JSON" in problem(None, allowed)

    names = ["Food", "uber"]
    specific = ["Your uber (₹300) was big.", "Food hit ₹1,300.", "Food again at ₹300."]
    assert problem({"summary": "ok", "tips": specific}, allowed, names) is None
    no_amount = ["Your uber was big.", *specific[1:]]
    assert "tip 1 is too generic" in problem({"summary": "ok", "tips": no_amount}, allowed, names)
    no_name = [specific[0], "Spend ₹300 less somewhere.", specific[2]]
    assert "tip 2 is too generic" in problem({"summary": "ok", "tips": no_name}, allowed, names)
    assert problem({"summary": "ok", "tips": ["Seafood ₹300", *specific[1:]]}, allowed, names)  # whole words only
    promise = ["Your uber (₹300) was big — take an auto to save ₹300.", *specific[1:]]
    assert "savings amount" in problem({"summary": "ok", "tips": promise}, allowed, names)  # ₹300 IS in facts


def test_a_category_is_never_called_the_biggest_expense():
    """Facts: Transport category ₹680; biggest single expense uber ₹340."""
    allowed, names = [680.0, 340.0, 220.0], ["Transport", "Food", "uber"]
    biggest = {"name": "uber", "amount": 340.0}
    problem = ai_insights._problem_with
    good = ["Your biggest single expense was uber (₹340) — try an auto next time.",
            "Your Transport category is at ₹680 this month — walk short distances.",
            "Your Food category is at ₹220 — keep planning meals."]
    assert problem({"summary": "ok", "tips": good}, allowed, names, biggest) is None

    category_as_expense = ["Your Transport (₹680) was your biggest expense — walk more.", *good[1:]]
    assert "category, not an expense" in problem(
        {"summary": "ok", "tips": category_as_expense}, allowed, names, biggest)
    summary_mistake = {"summary": "Transport is your biggest expense at ₹680.", "tips": good}
    assert "category, not an expense" in problem(summary_mistake, allowed, names, biggest)

    no_uber = ["Your Transport category is at ₹680 — walk more.", *good[1:]]
    assert "biggest single expense, uber (₹340)" in problem(
        {"summary": "ok", "tips": no_uber}, allowed, names, biggest)


def test_graph_has_the_four_nodes_in_order():
    graph = ai_insights.INSIGHTS_GRAPH.get_graph()
    assert {"fetch_data", "analyse", "write_insights", "check_output"} <= set(graph.nodes)
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("__start__", "fetch_data") in edges and ("fetch_data", "analyse") in edges
    assert ("analyse", "write_insights") in edges and ("write_insights", "check_output") in edges
    assert ("check_output", "write_insights") in edges                  # the retry loop


# ---------- Regression: numbers called the wrong thing / unfair % change (ml_demo, Nov 2026) ----------

FIXED_TODAY = date(2026, 11, 7)


def demo_month(user, income=None):
    """1–7 Nov: ₹17,385 (rent ₹15,000, groceries ₹2,000, auto ₹385). October: ₹45,000 in all, but
    only ₹15,000 (rent) in 1–7 Oct. Overall budget ₹28,000 (62.1% used)."""
    db.add_expense(user, 15000, "Bills", "2026-11-01", "rent")
    db.add_expense(user, 2000, "Food", "2026-11-03", "groceries")
    db.add_expense(user, 385, "Transport", "2026-11-05", "auto")
    db.add_expense(user, 15000, "Bills", "2026-10-01", "rent")
    db.add_expense(user, 30000, "Shopping", "2026-10-20", "laptop")
    db.set_budget(user, None, 28000)
    if income:
        db.add_income(user, income, "Salary", "2026-11-01", "salary")


def demo_facts(user):
    return ai_insights.analyse(ai_insights.fetch_data({"user_id": user, "today": FIXED_TODAY}))["facts"]


SAVINGS_LIE = {"summary": "You've spent ₹17,385 so far this month.",
               "tips": ["Your savings so far are ₹17,385 (62.1% of income) — keep it up.",
                        "Your biggest single expense was rent (₹15,000) — compare rents nearby.",
                        "Your Food category is at ₹2,000 this month — cook at home more."]}


def test_no_income_means_no_income_or_savings_facts(user):
    demo_month(user)
    facts = demo_facts(user)
    text = "\n".join(facts["lines"]).lower()
    assert "income" not in text and "saving" not in text            # not even "no income recorded"
    assert all(line.startswith("[") for line in facts["lines"])      # every fact has a fixed label
    assert facts["lines"][0].startswith("[spent_so_far]")
    assert "spent_so_far" in facts["amount_labels"][17385.0]
    assert "budget_spent" in facts["amount_labels"][17385.0]
    assert facts["percent_labels"][62.1] == ["budget_used_pct"]
    prompt = ai_insights.write_prompt(facts)
    assert "Your savings so far" not in prompt and "NO income" in prompt


def test_spending_called_savings_is_rejected_then_falls_back(user, fake_insights_ai):
    """The bug: 'your savings so far are ₹22,385 (60.5% of income)' for a user with NO income."""
    demo_month(user)
    calls = fake_insights_ai(SAVINGS_LIE, SAVINGS_LIE)
    result = ai_insights.run_insights(user, today=FIXED_TODAY)
    assert result["source"] == "fallback" and len(calls) == 2
    shown = (result["summary"] + " ".join(result["tips"])).lower()
    assert "saving" not in shown and "income" not in shown


def test_spending_called_savings_with_income_is_rejected_by_label(user, fake_insights_ai):
    demo_month(user, income=50000)                     # savings ₹32,615, rate 65.2%
    facts = demo_facts(user)
    problem = ai_insights._problem_with(SAVINGS_LIE, facts["allowed_amounts"], facts["names"],
                                        facts["biggest"], labels=facts)
    assert 'calls ₹17,385 "savings"' in problem and "spent_so_far" in problem
    good = {**SAVINGS_LIE, "tips": ["Your savings so far are ₹32,615 (65.2% of income) — move some to a deposit.",
                                    *SAVINGS_LIE["tips"][1:]]}
    assert ai_insights._problem_with(good, facts["allowed_amounts"], facts["names"],
                                     facts["biggest"], labels=facts) is None
    calls = fake_insights_ai(SAVINGS_LIE, good)
    result = ai_insights.run_insights(user, today=FIXED_TODAY)
    assert result["source"] == "ai" and result["tips"] == good["tips"] and len(calls) == 2


def test_label_check_rules():
    facts = ai_insights.Facts()
    facts.add("spent_so_far", "Spent ₹22,385.", spent_so_far=22385)
    facts.add("budget", "Overall: ₹22,385 spent of ₹37,000, 60.5% used.", budget_spent=22385,
              budget_limit=37000, budget_used_pct=60.5)
    d = facts.as_dict()

    def check(text, income=False):
        return ai_insights.label_problem(text, d["amount_labels"], d["percent_labels"], income)

    assert check("You've spent ₹22,385 so far, and your monthly budget is 60.5% used.") is None
    assert check("Your monthly budget is 61% used.") is None                    # rounded is fine
    assert "70% is not a percentage" in check("Your monthly budget is 70% used.")
    assert "NO income" in check("Your savings so far are ₹22,385.")
    assert "NO income" in check("That's 60.5% of income.")
    assert "is [budget_used_pct]" in check("Your savings rate is 60.5%.", income=True)
    assert 'calls ₹22,385 "income"' in check("Your income is ₹22,385.", income=True)
    assert check("No income was recorded, so focus on spending.") is None       # no number: fine


def test_change_compares_the_same_days_of_last_month(user):
    """7 days of November vs ALL of October showed '▼ 61.4%'; the fair figure is +15.9% vs 1–7 Oct."""
    demo_month(user)
    facts = demo_facts(user)
    assert facts["change"] == 15.9                                    # (17385 - 15000) / 15000
    numbers = {n["label"]: n for n in facts["key_numbers"]}
    assert "Change" not in numbers
    assert numbers["vs same days last month"]["value"] == "▲ 15.9%"
    assert numbers["vs same days last month"]["note"] == "1–7 Nov vs 1–7 Oct"
    assert numbers["Last month"]["value"] == "₹45,000"                  # the full month is still shown
    text = "\n".join(facts["lines"])
    assert "61.4" not in text
    assert "Change vs the same days of last month (1–7 Nov vs 1–7 Oct): 15.9% more (₹2,385 more)" in text
    assert "Bills: ₹15,000 this month so far vs ₹15,000 in the same days of last month (0.0%)" in text
    summary = ai_insights.plain_summary(facts)["summary"]
    assert "15.9% more than the same days last month (₹15,000 on 1–7 Oct 2026)" in summary


def test_same_days_last_month_dates():
    from forecast import same_days_label, same_days_last_month
    assert same_days_last_month(date(2026, 11, 7)) == (date(2026, 10, 1), date(2026, 10, 7))
    assert same_days_last_month(date(2027, 3, 31)) == (date(2027, 2, 1), date(2027, 2, 28))
    assert same_days_last_month(date(2027, 1, 15)) == (date(2026, 12, 1), date(2026, 12, 15))
    assert same_days_label(date(2026, 11, 7)) == "1–7 Nov vs 1–7 Oct"


def forecast_state(user):
    """demo_month's data plus a (fake) forecast: Food likely over its ₹10,000 budget around 27 Nov."""
    demo_month(user)
    db.set_budget(user, "Food", 10000)
    state = ai_insights.fetch_data({"user_id": user, "today": FIXED_TODAY})
    food = {"status": "ok", "category": "Food", "expected": 12200.0, "low": 11000.0, "high": 13900.0,
            "budget": {"limit": 10000.0, "status": "likely", "date": "2026-11-27", "chance": 0.8}}
    state["data"]["forecast"] = {
        "status": "ok", "category": None, "month_label": "November 2026", "expected": 38800.0,
        "low": 36400.0, "high": 42100.0,
        "budget": {"limit": 28000.0, "status": "over", "date": None, "chance": 1.0},
        "categories": [food], "budget_categories": []}
    return state


def test_forecast_facts_and_budget_tip(user):
    facts = ai_insights.analyse(forecast_state(user))["facts"]
    text = "\n".join(facts["lines"])
    assert "[forecast] Forecast for the end of November 2026" in text and "about ₹38,800" in text
    assert ("[budget_forecast] Food is on track to exceed its ₹10,000 budget around 27 Nov "
            "(forecast: about ₹12,200 by month-end).") in text
    assert "Overall spending is on track" not in text                    # already over: the budget fact says so
    numbers = {n["label"]: n for n in facts["key_numbers"]}
    assert numbers["Forecast vs last month"]["value"] == "▼ 13.8%"      # 38,800 vs 45,000 (both full months)
    assert "is on track to exceed your ₹<budget limit> budget" in ai_insights.write_prompt(facts)
    tips = ai_insights.plain_summary(facts)["tips"]
    assert any(t.startswith("Food is on track to exceed your ₹10,000 budget around 27 Nov") for t in tips)
    ai_tip = {"summary": "You've spent ₹17,385 so far; your monthly budget is 62.1% used.",
              "tips": ["Food is on track to exceed your ₹10,000 budget around 27 Nov — set a weekly food limit.",
                       "Your biggest single expense was rent (₹15,000) — compare rents nearby.",
                       "Your Food category is at ₹2,000 this month — cook at home more."]}
    assert ai_insights._problem_with(ai_tip, facts["allowed_amounts"], facts["names"],
                                     facts["biggest"], labels=facts) is None
