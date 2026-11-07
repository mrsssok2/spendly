"""Unusual-expense detection (anomaly.py): robust statistics, the "This is normal" mark, the ⚠️
badge, and the AI's "any unusual expenses?" answer (written in code, the model is never called)."""
from datetime import date, timedelta

import pytest

import anomaly
import database.db as db
from conftest import ask_ai

START = date(2026, 9, 1)


def rows(*items):
    """[(amount, category, day offset, description), ...] -> expense dicts, oldest first."""
    return [{"id": i + 1, "amount": a, "category": c, "date": (START + timedelta(days=d)).isoformat(),
             "description": text} for i, (a, c, d, text) in enumerate(items)]


def food_history(n=10, amount=150, every=4):
    return [(amount + (i % 3) * 10, "Food", i * every, "lunch") for i in range(n)]


def test_a_spike_is_flagged_with_a_plain_explanation():
    data = rows(*food_history(), (3280, "Food", 45, "party"))
    flagged = anomaly.find_anomalies(data)
    assert list(flagged) == [11]
    assert flagged[11]["explanation"] == "₹3,280 is about 20× your typical Food expense (₹160)"


def test_a_category_with_fewer_than_8_earlier_expenses_is_never_judged():
    data = rows(*food_history(n=7, every=6), (5000, "Food", 45, "party"))
    assert anomaly.find_anomalies(data) == {}


def test_nothing_is_judged_in_the_first_four_weeks():
    data = rows(*food_history(n=10, every=2), (5000, "Food", 21, "party"))
    assert anomaly.find_anomalies(data) == {}


def test_only_big_spikes_count_not_small_amounts():
    data = rows(*food_history(), (20, "Food", 45, "chai"), (420, "Food", 46, "pizza"))
    # ₹20 is unusual but cheap (only HIGH amounts are flagged); ₹420 is under 3x the typical ₹160.
    assert anomaly.find_anomalies(data) == {}


def test_expenses_sharing_a_word_are_judged_against_each_other():
    # Food mixes ₹30 chai and ₹1,000 weekly groceries: groceries are compared with groceries.
    items = [(30 + i % 5, "Food", i, "chai") for i in range(40)]
    items += [(1000 + 50 * (i % 3), "Food", 7 * i + 3, "Groceries") for i in range(5)]
    items += [(1100, "Food", 41, "Groceries"), (6000, "Food", 42, "Groceries for party")]
    flagged = anomaly.find_anomalies(rows(*items))
    assert [f["amount"] for f in flagged.values()] == [6000]
    assert "your typical “groceries” expense in Food" in next(iter(flagged.values()))["explanation"]


def test_median_mad_is_not_masked_by_an_earlier_spike_but_mean_std_is():
    """Why median + MAD: one earlier ₹9,000 spike inflates the mean and the standard deviation
    so much that the next ₹4,000 spike looks "normal" to mean/std (masking)."""
    items = food_history(n=12, every=3) + [(9000, "Food", 37, "wedding gift dinner"),
                                           (4000, "Food", 40, "party")]
    data = rows(*items)
    spike = len(data)
    assert spike in anomaly.find_anomalies(data, method="mad")
    assert spike not in anomaly.find_anomalies(data, method="meanstd")


def test_this_is_normal_stops_the_flag_until_the_amount_changes():
    data = rows(*food_history(), (3280, "Food", 45, "party"))
    assert anomaly.find_anomalies(data, dismissed={11: 3280.0}) == {}
    data[-1]["amount"] = 3500                               # edited after the mark: judged again
    assert 11 in anomaly.find_anomalies(data, dismissed={11: 3280.0})


# ---------- In the app ----------

def add_history(user_id, today):
    start = today - timedelta(days=60)
    for i in range(12):
        db.add_expense(user_id, 150 + (i % 3) * 10, "Food", (start + timedelta(days=i * 4)).isoformat(), "lunch")
    return db.add_expense(user_id, 3280, "Food", (today - timedelta(days=2)).isoformat(), "party")


def test_profile_shows_the_badge_and_this_is_normal_removes_it(client, user):
    spike = add_history(user, date.today())
    page = client.get("/profile").get_data(as_text=True)
    assert "anomaly-flag" in page and "₹3,280 is about 20× your typical Food expense (₹160)" in page

    response = client.post(f"/expenses/{spike}/normal", data={"date_from": "", "date_to": ""})
    assert response.status_code == 302
    page = client.get("/profile").get_data(as_text=True)
    assert "anomaly-flag" not in page
    assert db.get_anomaly_dismissals(user) == {spike: 3280.0}


def test_this_is_normal_only_works_on_your_own_expense(client, user, other_user):
    theirs = add_history(other_user, date.today())
    assert client.post(f"/expenses/{theirs}/normal").status_code == 404
    assert db.get_anomaly_dismissals(other_user) == {}
    assert db.get_anomaly_dismissals(user) == {}


def test_this_is_normal_needs_login(client, make_user):
    owner = make_user("x@example.com")
    spike = add_history(owner, date.today())
    response = client.post(f"/expenses/{spike}/normal")
    assert response.status_code == 302 and "/login" in response.headers["Location"]
    assert db.get_anomaly_dismissals(owner) == {}


def test_redirect_keeps_only_a_valid_date_filter(client, user):
    spike = add_history(user, date.today())
    response = client.post(f"/expenses/{spike}/normal",
                           data={"date_from": "2026-01-01", "date_to": "https://evil.example"})
    assert response.headers["Location"].endswith("/profile?date_from=2026-01-01")


def test_dismissal_is_deleted_with_the_expense(client, user):
    spike = add_history(user, date.today())
    db.dismiss_anomaly(user, spike)
    db.delete_expense(spike, user)
    assert db.get_anomaly_dismissals(user) == {}


# ---------- AI assistant ----------

@pytest.mark.parametrize("question", ["any unusual expenses?", "anything strange in my spending?",
                                      "show me outliers"])
def test_ai_lists_unusual_expenses_from_code(client, user, ollama_down, question):
    add_history(user, date.today())
    status, data = ask_ai(client, question)
    assert status == 200
    assert data["reply"].startswith("I found 1 unusual expense:")
    assert "₹3,280 is about 20× your typical Food expense (₹160)" in data["reply"]


def test_ai_says_when_nothing_is_unusual(client, user, ollama_down):
    status, data = ask_ai(client, "any unusual expenses this month?")
    assert status == 200
    assert data["reply"].startswith("No unusual expenses this month.")


def test_ai_never_shows_another_users_unusual_expenses(client, user, other_user, ollama_down):
    add_history(other_user, date.today())
    status, data = ask_ai(client, "any unusual expenses?")
    assert data["reply"].startswith("No unusual expenses") and "3,280" not in data["reply"]


def test_get_anomalies_tool_has_no_user_id_and_answers_from_code(user):
    import ai_agent
    import app as spendly
    add_history(user, date.today())
    log = {"answers": []}
    tools = {t.name: t for t in ai_agent._make_tools(user, [], spendly.validate_expense_form,
                                                      spendly.CATEGORIES, "any unusual food expenses?",
                                                      date.today(), log)}
    assert set(tools["get_anomalies"].args) == {"period", "category"}
    result = tools["get_anomalies"].invoke({"category": "Food"})
    assert "₹3,280 is about 20×" in result and log["answers"][0].startswith("I found 1 unusual Food expense")
