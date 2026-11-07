"""Analytics page: the route, the SQL queries behind it, the date filter and empty states.
Dates are relative to today, so these tests keep working in any month."""
import json
import re
from datetime import date, timedelta

import pytest

import app as spendly
import database.db as db
from forecast import same_days_label, same_days_last_month

TODAY = date.today()
THIS_START = spendly.month_start(TODAY, 0)
LAST_START = spendly.month_start(TODAY, 1)
LAST_END = THIS_START - timedelta(days=1)
SEVEN_MONTHS_AGO = spendly.month_start(TODAY, 7)


def iso(d):
    return d.isoformat()


def chart_data(response):
    """The JSON the page hands to Chart.js."""
    match = re.search(r'<script type="application/json" id="analytics-data">(.*?)</script>',
                      response.get_data(as_text=True), re.DOTALL)
    assert match, "no chart data on the page"
    return json.loads(match.group(1))


def add_sample_data(user, other_user):
    db.add_expense(user, 100, "Food", iso(THIS_START), "chai")
    db.add_expense(user, 200, "Transport", iso(TODAY), "auto")
    db.add_expense(user, 150, "Food", iso(LAST_START), "lunch")
    db.add_expense(user, 50, "Bills", iso(LAST_END), "recharge")
    db.add_expense(user, 999, "Shopping", iso(SEVEN_MONTHS_AGO), "old shoes")   # outside 6 months
    db.add_expense(other_user, 5000, "Food", iso(TODAY), "Ravi's dinner")       # someone else's


# ---------- Route basics ----------

def test_analytics_requires_login(client):
    response = client.get("/analytics")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_navbar_shows_analytics_only_when_logged_in(client, make_user):
    assert b">Analytics<" not in client.get("/").data
    make_user("asha@example.com")
    client.post("/login", data={"email": "asha@example.com", "password": "secret123"})
    assert b">Analytics<" in client.get("/").data


def test_no_expenses_shows_friendly_message_instead_of_charts(client, user):
    response = client.get("/analytics")
    assert response.status_code == 200
    assert b"No expenses yet" in response.data
    assert b"<canvas" not in response.data
    assert b"chart.js" not in response.data          # the library isn't even loaded


# ---------- SQL queries ----------

def test_month_comparison_is_calculated_by_sql(user, other_user):
    add_sample_data(user, other_user)
    result = db.get_month_comparison(user, iso(THIS_START), iso(TODAY), iso(LAST_START), iso(LAST_END))
    assert result == {"this_total": 300.0, "last_total": 200.0, "change_percent": 50.0}


def test_month_comparison_down_and_no_previous_month(user):
    db.add_expense(user, 100, "Food", iso(TODAY), "x")
    db.add_expense(user, 400, "Food", iso(LAST_START), "y")
    result = db.get_month_comparison(user, iso(THIS_START), iso(TODAY), iso(LAST_START), iso(LAST_END))
    assert result["change_percent"] == -75.0

    db.delete_expense(db.get_expenses_for_user(user, iso(LAST_START), iso(LAST_END))[0]["id"], user)
    result = db.get_month_comparison(user, iso(THIS_START), iso(TODAY), iso(LAST_START), iso(LAST_END))
    assert result["last_total"] == 0 and result["change_percent"] is None   # can't divide by 0


def test_monthly_and_daily_totals_queries(user, other_user):
    add_sample_data(user, other_user)
    monthly = db.get_monthly_totals(user, iso(LAST_START), iso(TODAY))
    assert monthly == [
        {"month": LAST_START.strftime("%Y-%m"), "total": 200.0, "count": 2},
        {"month": THIS_START.strftime("%Y-%m"), "total": 300.0, "count": 2},
    ]
    daily = {row["date"]: row["total"] for row in db.get_daily_totals(user, iso(THIS_START), iso(TODAY))}
    assert sum(daily.values()) == 300.0              # Ravi's ₹5000 is not included
    assert daily[iso(TODAY)] >= 200.0


# ---------- What the page sends to Chart.js ----------

def test_page_sends_sql_totals_as_json(client, user, other_user):
    add_sample_data(user, other_user)
    data = chart_data(client.get("/analytics"))

    monthly = data["monthly"]
    assert len(monthly["labels"]) == 6
    assert monthly["labels"][-1] == THIS_START.strftime("%b %Y")
    assert monthly["totals"][-1] == 300.0 and monthly["totals"][-2] == 200.0
    assert 999 not in monthly["totals"]              # 7 months ago is outside the window

    categories = dict(zip(data["categories"]["labels"], data["categories"]["totals"]))
    assert categories == {"Shopping": 999.0, "Food": 250.0, "Transport": 200.0, "Bills": 50.0}

    daily = data["daily"]
    assert len(daily["labels"]) == (TODAY - THIS_START).days + 1    # 1st of the month .. today
    assert sum(daily["totals"]) == 300.0
    assert all(chart["has_data"] for chart in data.values())


def test_summary_cards_show_totals_and_change(client, user, other_user):
    add_sample_data(user, other_user)
    page = client.get("/analytics").get_data(as_text=True)
    assert "₹300.00" in page and "₹200.00" in page
    # The % change is vs the SAME days of last month: lunch ₹150 on day 1 (+ the ₹50 recharge only
    # when today is on/after last month's last day).
    same_last = 150 + (50 if TODAY.day >= LAST_END.day else 0)
    change = round((300 - same_last) * 100 / same_last, 1)
    assert "vs same days last month" in page
    assert f"{abs(change)}%" in page and same_days_label(TODAY) in page


def test_change_card_never_compares_month_to_date_with_all_of_last_month(client, user):
    """Regression: on 7 Nov, ₹22,385 vs ALL of October (₹70,627) showed '▼ 68.3%'."""
    first, same_end = same_days_last_month(TODAY)
    if same_end >= LAST_END:
        pytest.skip("today is on/after last month's last day: the same days ARE the whole month")
    db.add_expense(user, 300, "Food", iso(THIS_START), "this month")
    db.add_expense(user, 200, "Food", iso(LAST_START), "same days last month")
    db.add_expense(user, 9000, "Shopping", iso(LAST_END), "after the same days")
    page = client.get("/analytics").get_data(as_text=True)
    assert "▲" in page and "50.0%" in page                       # 300 vs 200, not 300 vs 9,200
    assert "96.7%" not in page                                   # the old, misleading figure
    assert "₹9,200.00" in page                                   # the full last month is still shown


def test_date_filter_is_respected(client, user, other_user):
    add_sample_data(user, other_user)
    data = chart_data(client.get(f"/analytics?date_from={iso(LAST_START)}&date_to={iso(LAST_END)}"))
    assert data["monthly"]["labels"] == [LAST_START.strftime("%b %Y")]
    assert data["monthly"]["totals"] == [200.0]
    assert dict(zip(data["categories"]["labels"], data["categories"]["totals"])) == {"Food": 150.0, "Bills": 50.0}
    assert sum(data["daily"]["totals"]) == 200.0


def test_quick_filter_links_point_to_analytics(client, user, other_user):
    add_sample_data(user, other_user)
    page = client.get("/analytics").get_data(as_text=True)
    assert 'href="/analytics?date_from=' in page
    assert "This Month" in page


def test_invalid_filter_shows_error_and_still_works(client, user, other_user):
    add_sample_data(user, other_user)
    response = client.get("/analytics?date_from=not-a-date")
    assert response.status_code == 200
    assert b"Please enter valid dates" in response.data


def test_period_without_spending_shows_message_instead_of_chart(client, user):
    db.add_expense(user, 999, "Shopping", iso(SEVEN_MONTHS_AGO), "old shoes")   # only old data
    response = client.get("/analytics")
    page = response.get_data(as_text=True)
    assert "No spending in this period yet." in page
    assert 'id="chart-monthly"' not in page          # no empty bar chart
    assert 'id="chart-categories"' in page           # all-time categories still have data
    assert chart_data(response)["monthly"]["has_data"] is False


def test_descriptions_cannot_break_the_page(client, user):
    db.add_expense(user, 10, "Other", iso(TODAY), "</script><script>alert(1)</script>")
    page = client.get("/analytics").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in page
