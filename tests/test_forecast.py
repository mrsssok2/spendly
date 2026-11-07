"""Month-end forecast (forecast.py), the demo seed script, the backtest/anomaly evaluation
(evals/ml_eval.py), the dashboard card, the analytics chart and the AI's forecast answers.
Never the real database (conftest), never the real model (ollama_down)."""
import json
import os
import re
import sys
from datetime import date, timedelta

import pytest

import database.db as db
import forecast
from conftest import ask_ai, login

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT, "evals"))

import ml_eval  # noqa: E402
import seed_ml_demo  # noqa: E402

TODAY = date(2026, 11, 7)


@pytest.fixture
def demo(fresh_database):
    """The ML demo user (6 months of synthetic data, fixed seed) in the TEST database."""
    user_id, spikes = seed_ml_demo.seed(db, TODAY)
    return user_id, spikes


def expenses_of(user_id, up_to=TODAY):
    return db.get_expenses_up_to(user_id, up_to.isoformat())


# ---------- The demo data ----------

def test_seed_is_repeatable_and_only_touches_the_demo_user(make_user):
    other = make_user("real@example.com")
    db.add_expense(other, 123, "Food", "2026-11-01", "real data")
    first, _ = seed_ml_demo.seed(db, TODAY)
    rows_first = [(e["amount"], e["date"], e["description"]) for e in expenses_of(first)]
    second, spikes = seed_ml_demo.seed(db, TODAY)                   # re-run: re-created
    assert [(e["amount"], e["date"], e["description"]) for e in expenses_of(second)] == rows_first
    assert db.get_user_by_email(seed_ml_demo.DEMO_EMAIL)["id"] == second
    assert db.get_expense_summary(first)["count"] == 0              # the old copy is gone
    assert db.get_expense_summary(other) == {"total": 123, "count": 1}   # other users untouched
    assert len(spikes) == 5 and len(rows_first) > 300
    assert min(e[1] for e in rows_first) == "2026-05-01" and max(e[1] for e in rows_first) <= "2026-11-07"


def test_demo_user_can_log_in(client, demo):
    response = login(client, seed_ml_demo.DEMO_EMAIL, seed_ml_demo.DEMO_PASSWORD)
    assert response.status_code == 302


# ---------- The forecast model ----------

def test_not_enough_data_never_guesses():
    rows = [{"id": i, "amount": 100, "category": "Food", "date": (TODAY - timedelta(days=i)).isoformat(),
             "description": "lunch"} for i in range(20)]
    f = forecast.forecast_month(rows, TODAY)
    assert f["status"] == "not_enough_data"
    assert f["message"] == "Not enough data for a forecast yet"
    assert "6 weeks" in f["detail"] and "30 expenses" in f["detail"]
    assert "expected" not in f


def test_forecast_on_the_demo_user(demo):
    user_id, _ = demo
    f = forecast.build_forecast(expenses_of(user_id), TODAY, forecast.current_budgets(user_id, TODAY))
    assert f["status"] == "ok"
    spent = db.get_expense_summary(user_id, "2026-11-01", "2026-11-07")["total"]
    assert f["spent"] == spent                                 # the same as SQL's total
    assert spent <= f["low"] <= f["expected"] <= f["high"]
    assert f["days_left"] == 23 and len(f["path"]["dates"]) == 24     # today is still running
    names = [c["category"] for c in f["categories"]]
    assert names[:2] == ["Bills", "Food"] and len(names) == 3 and "Shopping" not in names  # no one-off phone
    # Rent and the other monthly bills are recognised; the ones still to come are expected.
    upcoming = {u["description"]: u["date"] for u in f["upcoming"]}
    assert upcoming == {"Mobile recharge": "2026-11-12", "Internet bill": "2026-11-15",
                        "Netflix subscription": "2026-11-20"}
    # Same data -> same forecast (fixed random seed).
    again = forecast.build_forecast(expenses_of(user_id), TODAY, forecast.current_budgets(user_id, TODAY))
    assert again["expected"] == f["expected"] and again["path"] == f["path"]


def test_backtest_cannot_see_the_future(demo):
    user_id, _ = demo
    as_of = date(2026, 9, 10)
    everything = forecast.forecast_month(expenses_of(user_id), as_of, partial_today=False)
    only_past = forecast.forecast_month(expenses_of(user_id, as_of), as_of, partial_today=False)
    assert everything == only_past


def test_one_off_spikes_are_left_out_of_the_everyday_pattern(demo):
    user_id, _ = demo
    # The ₹32,000 phone (08 Oct) is in the last 90 days. Kept in, it stretched the range to ~₹73k.
    f = forecast.forecast_month(expenses_of(user_id), TODAY)
    assert f["high"] < f["expected"] * 1.25
    kept = forecast.forecast_month(expenses_of(user_id), TODAY, one_offs=frozenset())
    assert kept["high"] > f["high"]


@pytest.mark.parametrize("limit, status, text", [
    (20000, "over", "You've already gone over your ₹20,000 budget"),
    (30000, "likely", "You'll likely go over your ₹30,000 budget around "),
    (90000, "ok", "You're on track to stay within your ₹90,000 budget."),
])
def test_budget_outlook(demo, limit, status, text):
    user_id, _ = demo
    f = forecast.forecast_month(expenses_of(user_id), TODAY, budget=limit)
    assert f["budget"]["status"] == status
    assert f["budget"]["message"].startswith(text)
    if status == "likely":
        crossing = date.fromisoformat(f["budget"]["date"])
        assert TODAY <= crossing <= date(2026, 11, 30)
        assert re.search(r"around \d{2} Nov\.$", f["budget"]["message"])


def test_recurring_items_need_a_steady_monthly_pattern():
    def e(i, day, text, amount=500):
        return {"id": i, "amount": amount, "category": "Bills", "date": day, "description": text}
    rows = [e(1, "2026-08-05", "Rent", 9000), e(2, "2026-09-05", "Rent", 9000), e(3, "2026-10-04", "Rent", 9000),
            e(4, "2026-08-02", "Donation"), e(5, "2026-10-27", "Donation"),            # random days
            e(6, "2026-08-10", "Water"), e(7, "2026-08-25", "Water"), e(8, "2026-09-10", "Water")]  # twice a month
    found = forecast.find_recurring(forecast._parse(rows), TODAY)
    assert set(found) == {"rent"} and found["rent"]["day"] == 5 and found["rent"]["chance"] == 1.0


def test_baselines_for_the_backtest(demo):
    user_id, _ = demo
    rows = expenses_of(user_id)
    as_of = date(2026, 10, 10)
    month = [e for e in rows if "2026-10-01" <= e["date"] <= "2026-10-10"]
    assert forecast.run_rate_forecast(rows, as_of) == pytest.approx(sum(e["amount"] for e in month) / 10 * 31)
    lr = forecast.linear_regression_forecast(rows, as_of)
    assert lr is not None and lr > sum(e["amount"] for e in month)


# ---------- Evaluation (evals/ml_eval.py) ----------

def test_ml_evaluation_passes_on_the_demo_user(fresh_database):
    report = ml_eval.evaluate(db, TODAY, simulations=300)
    assert report["passed"], report["problems"]
    f = report["forecast"]
    assert f["forecasts"] >= 8 and f["skipped_not_enough_data"] >= 2      # never guessed early on
    ours = f["by_model"]["forecast"]["all"]["mape"]
    assert ours < f["by_model"]["run_rate"]["all"]["mape"]                # beats the simple rule
    assert report["anomalies"]["mad"]["recall"] == 1.0
    assert report["anomalies"]["mad"]["precision"] >= 0.8
    assert {"forecast", "run_rate", "linear_regression"} <= set(f["by_model"])
    assert {"mad", "iqr", "meanstd", "isolation_forest"} <= set(report["anomalies"])


# ---------- Dashboard card, analytics chart ----------

def test_dashboard_says_not_enough_data(client, user):
    db.add_expense(user, 250, "Food", date.today().isoformat(), "lunch")
    page = client.get("/profile").get_data(as_text=True)
    assert "Month-end forecast" in page and "Not enough data for a forecast yet" in page
    assert "Likely total by month-end" not in page


def test_dashboard_card_and_analytics_chart_for_the_demo_user(client):
    seed_ml_demo.seed(db, date.today())          # the app forecasts "today"
    login(client, seed_ml_demo.DEMO_EMAIL, seed_ml_demo.DEMO_PASSWORD)
    page = client.get("/profile").get_data(as_text=True)
    assert "Likely total by month-end" in page and "Probably between" in page
    assert "forecast-bar" in page and "anomaly-flag" in page
    assert "New phone" in page                    # the ₹32,000 spike has its ⚠️ badge

    page = client.get("/analytics").get_data(as_text=True)
    assert 'id="chart-forecast"' in page
    data = json.loads(re.search(r'<script type="application/json" id="analytics-data">(.*?)</script>',
                                page, re.S).group(1))
    chart = data["forecast"]
    today_i = chart["labels"].index(f"{date.today():%d %b}")
    assert chart["actual"][today_i] == chart["expected"][today_i]      # the lines meet today
    assert all(v is None for v in chart["actual"][today_i + 1:])
    assert all(lo <= mid <= hi for lo, mid, hi in zip(chart["low"][today_i:], chart["expected"][today_i:],
                                                       chart["high"][today_i:]))


def test_analytics_without_enough_data_has_no_forecast_chart(client, user):
    db.add_expense(user, 250, "Food", date.today().isoformat(), "lunch")
    page = client.get("/analytics").get_data(as_text=True)
    assert 'id="chart-forecast"' not in page and "Not enough data for a forecast yet" in page


# ---------- AI assistant: numbers only from the forecast ----------

def demo_login(client):
    seed_ml_demo.seed(db, date.today())
    login(client, seed_ml_demo.DEMO_EMAIL, seed_ml_demo.DEMO_PASSWORD)
    user_id = db.get_user_by_email(seed_ml_demo.DEMO_EMAIL)["id"]
    return user_id, forecast.get_forecast(user_id)


@pytest.mark.parametrize("question", ["how much will I spend this month?", "what's my month-end forecast?",
                                      "am I going to spend more than usual?", "predict my spending"])
def test_ai_forecast_answer_comes_from_the_model_not_the_ai(client, ollama_down, question):
    _, f = demo_login(client)
    status, data = ask_ai(client, question)
    assert status == 200
    reply = data["reply"]
    assert reply.startswith(f"By the end of {f['month_label']} you'll likely spend about ")
    for key in ("expected", "low", "high", "spent"):
        assert forecast.format_inr(f[key]) in reply


def test_ai_will_i_go_over_budget(client, ollama_down):
    _, f = demo_login(client)
    status, data = ask_ai(client, "will I go over budget?")
    assert status == 200 and data["reply"].startswith(f["budget"]["message"])


def test_ai_category_forecast(client, ollama_down):
    user_id, _ = demo_login(client)
    f = forecast.get_category_forecast(user_id, "Food")
    status, data = ask_ai(client, "how much will I spend on food this month?")
    assert forecast.format_inr(f["expected"]) in data["reply"] and "on Food" in data["reply"]


def test_ai_forecast_with_too_little_data_says_so(client, user, ollama_down):
    db.add_expense(user, 250, "Food", date.today().isoformat(), "lunch")
    status, data = ask_ai(client, "how much will I spend this month?")
    assert status == 200
    assert data["reply"].startswith("Not enough data for a forecast yet.")
    assert not re.search(r"₹\d", data["reply"])


def test_ai_only_forecasts_this_month(client, user, ollama_down):
    status, data = ask_ai(client, "how much will I spend next month?")
    assert data["reply"].startswith("I can only forecast the current month")


def test_past_questions_are_not_forecasts():
    import ai_agent
    assert not ai_agent.is_forecast_question("how much did I spend this month?")
    assert ai_agent.is_forecast_question("will I go over my food budget?")
    assert ai_agent.RECORDS_SPENDING.search("spent 500 on food today, will I go over budget?")


def test_get_forecast_tool_answers_from_code(user):
    import ai_agent
    import app as spendly
    log = {"answers": []}
    tools = {t.name: t for t in ai_agent._make_tools(user, [], spendly.validate_expense_form,
                                                      spendly.CATEGORIES, "will I go over?",
                                                      date.today(), log)}
    assert set(tools["get_forecast"].args) == {"category"}
    tools["get_forecast"].invoke({})
    assert log["answers"] == ["Not enough data for a forecast yet. Add some expenses first."]
