import calendar
import math
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from functools import wraps

from flask import (
    Flask, abort, flash, jsonify, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash

import config  # first: loads .env and reads every setting from environment variables

import ai_agent  # ai_agent.PROVIDER is read per request (provider_text below)
from ai_agent import (
    AI_LOG,
    AIBusy,
    AIError,
    RECEIPTS_UNAVAILABLE,
    MessageRejected,
    amount_problem,
    apply_expense_changes,
    clear_chat,
    expense_snapshot,
    format_rupees as short_inr,
    format_short_date as short_date,
    get_chat_history,
    ai_health,
    record_chat,
    run_expense_assistant,
    screen_by_rules,
    vision_available,
)
from ai_insights import run_insights
from anomaly import anomalies_by_id
from forecast import get_forecast, same_days_label, same_days_last_month
from vision import MAX_UPLOAD_BYTES, ReceiptError, scan_receipt
from voice import MAX_AUDIO_BYTES, VoiceError, VoiceUnavailable, model_is_downloaded, transcribe_audio
from database.db import (
    add_expense,
    add_expenses,
    add_income,
    add_incomes,
    delete_income,
    get_income,
    get_income_for_user,
    get_money_summary,
    get_monthly_income,
    update_income,
    create_user,
    delete_expense,
    delete_budget,
    dismiss_anomaly,
    get_budget,
    get_budget_status,
    get_category_totals,
    get_daily_totals,
    get_expense,
    get_expense_summary,
    get_expenses_for_user,
    get_month_comparison,
    get_monthly_totals,
    get_user_by_email,
    get_user_by_id,
    check_database,
    init_db,
    seed_db,
    set_budget,
    update_expense,
)

app = Flask(__name__)
# Needed for sessions and flash messages. Set a real SECRET_KEY (in .env) outside development.
app.secret_key = config.SECRET_KEY
# Don't send the login cookie when another website submits a form to Spendly.
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Refuse request bodies over 6 MB before reading them (receipt photos are limited to 5 MB).
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024

# Set up the database when the app starts (both functions are safe to run repeatedly).
init_db()
seed_db()

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

CATEGORIES = ["Food", "Transport", "Bills", "Health", "Entertainment", "Shopping", "Other"]
INCOME_SOURCES = ["Salary", "Freelance", "Business", "Gift", "Refund", "Other"]
MAX_DESCRIPTION_LENGTH = 200


# ---------- Template filters ----------

@app.template_filter("inr")
def format_inr(amount):
    """2499.5 -> '₹2,499.50'"""
    return f"₹{amount:,.2f}"


CATEGORY_ICONS = {
    "Food": "🍜", "Transport": "🚕", "Bills": "💡", "Health": "💊",
    "Entertainment": "🎬", "Shopping": "🛍️", "Other": "📦",
}


@app.template_filter("cat_icon")
def category_icon(category):
    """'Food' -> '🍜' (used in tags, lists and the budget section)."""
    return CATEGORY_ICONS.get(category, "💰")


SOURCE_ICONS = {"Salary": "💼", "Freelance": "🧑‍💻", "Business": "🏪", "Gift": "🎁", "Refund": "↩️", "Other": "💵"}


@app.template_filter("source_icon")
def source_icon(source):
    """'Salary' -> '💼' (income sources)."""
    return SOURCE_ICONS.get(source, "💵")


@app.template_filter("cat_class")
def category_class(category):
    """'Food' -> 'cat-food', for the category's colour in CSS."""
    return "cat-" + (category or "other").lower()


@app.template_filter("nice_date")
def format_date(value):
    """'2026-11-04' or '2026-11-04 18:28:01' -> '04 Nov 2026'"""
    return datetime.strptime(value[:10], "%Y-%m-%d").strftime("%d %b %Y")


# ---------- Provider-dependent page text ----------
# "Local / private / no API keys" is only true with Ollama. With Groq the AI runs in the cloud,
# so pages must never claim that: every template gets the texts for the CURRENT provider.
AI_TEXT = {
    "ollama": {
        "hero": "friendly tips from an AI that runs privately on your own computer.",
        "trust_privacy": "🔒 Private by design",
        "trust_ai": "🤖 Local AI, no API keys",
        "assistant_status": "Runs privately on your computer",
        "insights_by": "your local AI",
        "auth_point": "A private AI on your own computer",
        "register_subtitle": "Free, private, and yours 🌱",
    },
    "groq": {
        "hero": "friendly tips from fast cloud AI (Groq).",
        "trust_privacy": "🔒 Only you can see your data",
        "trust_ai": "🤖 Fast cloud AI (Groq)",
        "assistant_status": "Fast cloud AI (Groq)",
        "insights_by": "AI (Groq, in the cloud)",
        "auth_point": "Fast cloud AI (Groq) for tips and questions",
        "register_subtitle": "Free, simple, and yours 🌱",
    },
}


@app.context_processor
def provider_text():
    """`ai_text` in every template: the AI wording for the provider in use (ai_agent.PROVIDER)."""
    return {"ai_text": AI_TEXT.get(ai_agent.PROVIDER, AI_TEXT["groq"])}


# ---------- Helpers ----------

def login_required(view):
    """Send visitors who aren't logged in to the login page."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/health")
def health():
    """For Docker's healthcheck (and humans): JSON, no login, no AI call (only a quick
    "is the AI provider there, which models does it have?" check, never showing the API key).
    200 when the app and database work, 503 when the database doesn't. The AI being down does
    NOT make the app unhealthy: everything except the AI still works."""
    database_ok = check_database()
    ai = ai_health()
    body = {
        "app": "ok",
        "database": "ok" if database_ok else "error",
        "ai": ai,
        "status": "ok" if database_ok and ai["reachable"] else ("degraded" if database_ok else "error"),
    }
    return jsonify(body), (200 if database_ok else 503)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", errors=[], form={})

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    errors = []
    if not name or not email or not password or not confirm_password:
        errors.append("All fields are required.")
    if email and not EMAIL_PATTERN.match(email):
        errors.append("Please enter a valid email address.")
    if password and len(password) < 6:
        errors.append("Password must be at least 6 characters.")
    if password and confirm_password and password != confirm_password:
        errors.append("Passwords do not match.")

    if not errors:
        user_id = create_user(name, email, password)
        if user_id is None:
            errors.append("An account with this email is already registered.")

    if errors:
        # Re-show the form with the user's name and email kept (never the passwords).
        form = {"name": name, "email": email}
        return render_template("register.html", errors=errors, form=form), 400

    flash("Account created! Please log in.", "success")
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None, email="")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    user = get_user_by_email(email) if email else None
    if user is None or not check_password_hash(user["password_hash"], password):
        # Same message whether the email or the password was wrong.
        return render_template(
            "login.html", error="Invalid email or password.", email=email
        ), 401

    session.clear()  # start a fresh session on every login
    session["user_id"] = user["id"]
    session["user_name"] = user["name"]
    flash(f"Welcome back, {user['name']}!", "success")
    return redirect(url_for("index"))


def is_valid_date(value):
    """True if value is a real date in YYYY-MM-DD form."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def month_start(today, months_back):
    """First day of the month `months_back` months before today's month."""
    year, month = today.year, today.month - months_back
    while month <= 0:
        month += 12
        year -= 1
    return date(year, month, 1)


def quick_filters(today):
    """The quick-filter buttons. Each range runs to the end of the current month."""
    month_end = date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
    ranges = [
        ("This Month", month_start(today, 0)),
        ("Last 3 Months", month_start(today, 2)),
        ("Last 6 Months", month_start(today, 5)),
    ]
    filters = [
        {"label": label, "date_from": start.isoformat(), "date_to": month_end.isoformat()}
        for label, start in ranges
    ]
    filters.append({"label": "All Time", "date_from": "", "date_to": ""})
    return filters


def read_date_filter():
    """Read and check the date filter from the URL (?date_from=...&date_to=...).

    Shared by the profile and analytics pages so both behave identically. Returns the
    values the templates need; on an invalid filter, active_from/active_to are None
    (= all time) but the typed values are kept to show in the inputs.
    """
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    filter_error = None
    if (date_from and not is_valid_date(date_from)) or (date_to and not is_valid_date(date_to)):
        filter_error = "Please enter valid dates. Showing all expenses instead."
    elif date_from and date_to and date_from > date_to:
        filter_error = "The From date must be on or before the To date. Showing all expenses instead."

    active_from = None if filter_error else (date_from or None)
    active_to = None if filter_error else (date_to or None)

    filters = quick_filters(date.today())
    for f in filters:
        f["active"] = (f["date_from"] or None) == active_from and (f["date_to"] or None) == active_to

    return {
        "date_from": date_from,
        "date_to": date_to,
        "filter_error": filter_error,
        "active_from": active_from,
        "active_to": active_to,
        "is_filtered": bool(active_from or active_to),
        "filters": filters,
    }


@app.route("/profile")
@login_required
def profile():
    user = get_user_by_id(session["user_id"])
    if user is None:
        # The account no longer exists (e.g. the database was reset).
        session.clear()
        flash("Please log in again.", "error")
        return redirect(url_for("login"))

    date_filter = read_date_filter()
    active_from, active_to = date_filter["active_from"], date_filter["active_to"]

    # ---- Every number below uses the same filter ----
    has_any_expenses = get_expense_summary(user["id"])["count"] > 0
    money = get_money_summary(user["id"], active_from, active_to)   # income, balance, savings rate (SQL)
    incomes = get_income_for_user(user["id"], active_from, active_to)
    has_any_income = get_money_summary(user["id"])["income_count"] > 0
    summary = get_expense_summary(user["id"], active_from, active_to)
    category_rows = get_category_totals(user["id"], active_from, active_to)
    expenses = get_expenses_for_user(user["id"], active_from, active_to)

    categories = [
        {
            "name": row["category"],
            "total": row["total"],
            "count": row["count"],
            "percent": round(row["total"] / summary["total"] * 100, 1),
        }
        for row in category_rows
    ]
    top_category = categories[0] if categories else None
    budgets = budget_rows(user["id"])  # always this month, whatever the date filter
    forecast = forecast_card(get_forecast(user["id"]))   # always this month, like budgets
    anomalies = anomalies_by_id(user["id"])               # judged on the whole history

    return render_template(
        "profile.html",
        user=user,
        summary=summary,
        categories=categories,
        top_category=top_category,
        expenses=expenses,
        has_any_expenses=has_any_expenses,
        money=money,
        incomes=incomes,
        has_any_income=has_any_income,
        budgets=budgets,
        budget_alerts=[row["message"] for row in budgets if row["message"]],
        budget_categories=CATEGORIES,
        this_month_label=f"{date.today():%B %Y}",
        forecast=forecast,
        anomalies=anomalies,
        filter_endpoint="profile",
        **date_filter,
    )


# ---------- Budgets ----------

BUDGET_ORANGE_PERCENT = 70   # progress bar turns orange and the profile banner appears
BUDGET_ALERT_PERCENT = 80    # adding an expense that crosses this shows a warning


def this_month_range(today=None):
    today = today or date.today()
    return month_start(today, 0).isoformat(), end_of_month(today).isoformat()


def budget_name(category):
    return "monthly budget" if category is None else f"{category} budget"


def nice_percent(value):
    """75.0 -> '75', 82.5 -> '82.5'"""
    return f"{value:g}"


def budget_level(percent):
    if percent >= 100:
        return "over"
    return "warning" if percent >= BUDGET_ORANGE_PERCENT else "ok"


def budget_message(row):
    """The alert text for one budget, or None if it's under 70%."""
    level = budget_level(row["percent"])
    if level == "over":
        title = "Monthly budget" if row["category"] is None else f"{row['category']} budget"
        if row["remaining"] == 0:
            return f"🚨 {title} fully used ({short_inr(row['monthly_limit'])})"
        return f"🚨 {title} exceeded by {short_inr(-row['remaining'])}"
    if level == "warning":
        return (f"⚠️ You've used {nice_percent(row['percent'])}% of your "
                f"{short_inr(row['monthly_limit'])} {budget_name(row['category'])}")
    return None


def budget_rows(user_id):
    """This month's budgets with display details. All numbers come from SQL."""
    rows = get_budget_status(user_id, *this_month_range())
    for row in rows:
        row["name"] = "Overall" if row["category"] is None else row["category"]
        row["level"] = budget_level(row["percent"])
        row["bar_width"] = min(row["percent"], 100)
        row["message"] = budget_message(row)
    return rows


def budget_crossings(before, after):
    """Warnings for budgets that a new expense pushed over 80% or over 100%."""
    previous = {row["category"]: row["percent"] for row in before}
    warnings = []
    for row in after:
        was = previous.get(row["category"], 0)
        if was < 100 <= row["percent"]:
            warnings.append(budget_message(row))
        elif was < BUDGET_ALERT_PERCENT <= row["percent"]:
            warnings.append(f"⚠️ You've now used {nice_percent(row['percent'])}% of your "
                            f"{short_inr(row['monthly_limit'])} {budget_name(row['category'])}")
    return warnings


def add_expenses_with_budget_check(user_id, expenses):
    """Save expenses (all or none) and return budget warnings they caused.
    Used by BOTH the Add Expense form and the AI assistant's Confirm."""
    month_from, month_to = this_month_range()
    affects_this_month = any(month_from <= e[2] <= month_to for e in expenses)
    before = get_budget_status(user_id, month_from, month_to) if affects_this_month else []
    add_expenses(user_id, expenses)
    if not affects_this_month:
        return []
    return budget_crossings(before, get_budget_status(user_id, month_from, month_to))


def validate_budget_form(form_data):
    """Returns (category or None for overall, amount, errors)."""
    category = form_data.get("category", "").strip() or None
    errors = []
    if category is not None and category not in CATEGORIES:
        errors.append("Please choose a category from the list.")
    amount = None
    try:
        amount = round(float(form_data.get("amount", "").strip()), 2)
    except ValueError:
        errors.append("Budget must be a number.")
    else:
        if not math.isfinite(amount) or amount <= 0:
            errors.append("Budget must be greater than 0.")
    return category, amount, errors


@app.route("/budgets", methods=["POST"])
@login_required
def save_budget():
    category, amount, errors = validate_budget_form(request.form)
    if errors:
        for error in errors:
            flash(error, "error")
        return redirect(url_for("profile"))
    set_budget(session["user_id"], category, amount)  # user id only from the session
    flash(f"{'Overall monthly' if category is None else category} budget set to {short_inr(amount)}.",
          "success")
    return redirect(url_for("profile"))


@app.route("/budgets/delete", methods=["POST"])
@login_required
def remove_budget():
    category = request.form.get("category", "").strip() or None
    if category is not None and category not in CATEGORIES:
        abort(404)
    if not delete_budget(session["user_id"], category):  # only ever this user's budget
        abort(404)
    flash("Budget removed.", "success")
    return redirect(url_for("profile"))


# ---------- Forecast & unusual expenses (forecast.py / anomaly.py) ----------

def forecast_card(f):
    """Display details for the dashboard's forecast card: where the spent-so-far, the range and
    the budget sit on one bar (percent of the bar's width). All amounts come from forecast.py."""
    if f["status"] != "ok":
        return f
    budget = f["budget"]["limit"] if f.get("budget") else None
    scale = max(f["high"], budget or 0, 1) * 1.08

    def pct(value):
        return round(min(value / scale * 100, 100), 1)

    return {**f, "bar": {
        "spent": pct(f["spent"]), "low": pct(f["low"]), "high": pct(f["high"]),
        "expected": pct(f["expected"]), "budget": pct(budget) if budget else None,
    }}


@app.route("/expenses/<int:expense_id>/normal", methods=["POST"])
@login_required
def mark_expense_normal(expense_id):
    """'This is normal': this unusual expense is never flagged again (unless it is edited)."""
    if not dismiss_anomaly(session["user_id"], expense_id):   # only this user's expense
        abort(404)
    flash("Got it: that expense won't be flagged as unusual again.", "success")
    # Back to the same date filter (checked: only real dates go into the link).
    args = {}
    for key in ("date_from", "date_to"):
        value = request.form.get(key, "").strip()
        if value and is_valid_date(value):
            args[key] = value
    return redirect(url_for("profile", **args))


def forecast_vs_last_month(f, last_total):
    """The month-end forecast vs ALL of last month (SQL total): {"expected", "percent"}, or None
    when last month had no spending. Both totals are full months, so the comparison is fair."""
    if f["status"] != "ok" or not last_total:
        return None
    return {"expected": f["expected"], "percent": round((f["expected"] - last_total) * 100 / last_total, 1)}


def forecast_chart(f, today):
    """Actual vs forecast for this month, cumulative per day. Actual comes from SQL daily totals;
    the forecast line and its 10-90% band from forecast.py. None when there is no forecast."""
    if f["status"] != "ok":
        return None
    start, end = month_start(today, 0), end_of_month(today)
    daily = {row["date"]: row["total"]
             for row in get_daily_totals(session["user_id"], start.isoformat(), today.isoformat())}
    bands = {d: i for i, d in enumerate(f["path"]["dates"])}
    labels, actual, expected, low, high = [], [], [], [], []
    running = 0.0
    day = start
    while day <= end:
        labels.append(f"{day:%d %b}")
        if day <= today:
            running = round(running + daily.get(day.isoformat(), 0), 2)
            actual.append(running)
        else:
            actual.append(None)
        i = bands.get(day.isoformat())
        if day == today:
            expected.append(running)       # the forecast line starts where the actual line ends
            low.append(running)
            high.append(running)
        elif i is not None and day > today:
            expected.append(f["path"]["p50"][i])
            low.append(f["path"]["p10"][i])
            high.append(f["path"]["p90"][i])
        else:
            expected.append(None)
            low.append(None)
            high.append(None)
        day += timedelta(days=1)
    budget = f["budget"]["limit"] if f.get("budget") else None
    return {"labels": labels, "actual": actual, "expected": expected, "low": low, "high": high,
            "budget": budget, "title": f["month_label"], "has_data": True}


# ---------- Analytics ----------

MAX_DAILY_CHART_DAYS = 186  # about 6 months of points; longer ranges show their last 186 days


def end_of_month(day):
    return date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])


def month_starts(start, end):
    """First day of every month from start's month to end's month, inclusive."""
    months, current = [], date(start.year, start.month, 1)
    while current <= end:
        months.append(current)
        current = end_of_month(current) + timedelta(days=1)
    return months


def chart_ranges(today, active_from, active_to):
    """The date ranges for the monthly and daily charts, following the page's date filter.
    No filter: last 6 months (monthly) and the current month up to today (daily)."""
    end = date.fromisoformat(active_to) if active_to else None
    start = date.fromisoformat(active_from) if active_from else None

    monthly_end = end or end_of_month(today)
    monthly_start = start or month_start(monthly_end, 5)

    if start or end:
        daily_end = end or today
        daily_start = start or date(daily_end.year, daily_end.month, 1)
    else:
        daily_start, daily_end = month_start(today, 0), today
    if daily_start <= today < daily_end:
        daily_end = today  # don't draw empty future days
    trimmed = (daily_end - daily_start).days + 1 > MAX_DAILY_CHART_DAYS
    if trimmed:
        daily_start = daily_end - timedelta(days=MAX_DAILY_CHART_DAYS - 1)
    return (monthly_start, monthly_end), (daily_start, daily_end), trimmed


def range_label(start, end):
    return f"{start:%d %b %Y} – {end:%d %b %Y}"


def filter_label(active_from, active_to):
    """'All time', '01 Oct 2026 – 30 Nov 2026', 'From 01 Oct 2026' or 'Up to 30 Nov 2026'."""
    if active_from and active_to:
        return range_label(date.fromisoformat(active_from), date.fromisoformat(active_to))
    if active_from:
        return f"From {date.fromisoformat(active_from):%d %b %Y}"
    if active_to:
        return f"Up to {date.fromisoformat(active_to):%d %b %Y}"
    return "All time"


@app.route("/analytics")
@login_required
def analytics():
    user_id = session["user_id"]
    today = date.today()
    date_filter = read_date_filter()
    active_from, active_to = date_filter["active_from"], date_filter["active_to"]

    # ---- Summary cards: this month vs last month (all three numbers come from SQL) ----
    this_start = month_start(today, 0)
    last_start = month_start(today, 1)
    comparison = get_month_comparison(
        user_id, this_start.isoformat(), end_of_month(today).isoformat(),
        last_start.isoformat(), (this_start - timedelta(days=1)).isoformat(),
    )
    # The % change compares the SAME days of both months (1–7 Nov vs 1–7 Oct), never 7 days of this
    # month with all of last month (that once showed "▼ 68.3%" on day 7). SQL does the maths.
    same_start, same_end = same_days_last_month(today)
    same_days = get_month_comparison(
        user_id, this_start.isoformat(), today.isoformat(), same_start.isoformat(), same_end.isoformat())
    same_days["label"] = same_days_label(today)

    # ---- Chart data: SQL totals; Python only fills months/days with no spending as 0 ----
    (m_start, m_end), (d_start, d_end), daily_trimmed = chart_ranges(today, active_from, active_to)

    monthly_sql = {row["month"]: row["total"]
                   for row in get_monthly_totals(user_id, m_start.isoformat(), m_end.isoformat())}
    months = month_starts(m_start, m_end)
    monthly = {
        "labels": [f"{m:%b %Y}" for m in months],
        "totals": [monthly_sql.get(f"{m:%Y-%m}", 0) for m in months],
        "title": "Last 6 months" if not date_filter["is_filtered"] else range_label(m_start, m_end),
    }

    category_rows = get_category_totals(user_id, active_from, active_to)
    categories = {
        "labels": [row["category"] for row in category_rows],
        "totals": [row["total"] for row in category_rows],
        "title": filter_label(active_from, active_to),
    }

    daily_sql = {row["date"]: row["total"]
                 for row in get_daily_totals(user_id, d_start.isoformat(), d_end.isoformat())}
    days = [d_start + timedelta(days=i) for i in range((d_end - d_start).days + 1)]
    daily = {
        "labels": [f"{d:%d %b}" for d in days],
        "totals": [daily_sql.get(d.isoformat(), 0) for d in days],
        "title": (f"{today:%B %Y}" if not date_filter["is_filtered"] else range_label(d_start, d_end))
                 + (" (last 6 months of the range)" if daily_trimmed else ""),
    }

    charts = {"monthly": monthly, "categories": categories, "daily": daily}
    for chart in charts.values():
        chart["has_data"] = any(total > 0 for total in chart["totals"])

    # Income vs expenses per month (same months as the monthly chart; totals from SQL).
    income_sql = {row["month"]: row["total"]
                  for row in get_monthly_income(user_id, m_start.isoformat(), m_end.isoformat())}
    income_totals = [income_sql.get(f"{m:%Y-%m}", 0) for m in months]
    charts["income_vs_expenses"] = {
        "labels": monthly["labels"],
        "income": income_totals,
        "expenses": monthly["totals"],
        "title": monthly["title"],
        "has_data": any(v > 0 for v in income_totals + monthly["totals"]),
        "has_income": any(v > 0 for v in income_totals),
    }

    forecast = get_forecast(user_id, today)
    forecast_change = None
    if forecast["status"] == "ok":
        charts["forecast"] = forecast_chart(forecast, today)
        forecast_change = forecast_vs_last_month(forecast, comparison["last_total"])

    return render_template(
        "analytics.html",
        forecast=forecast,
        forecast_change=forecast_change,
        comparison=comparison,
        same_days=same_days,
        this_month_label=f"{today:%B %Y}",
        last_month_label=f"{last_start:%B %Y}",
        charts=charts,
        has_any_expenses=get_expense_summary(user_id)["count"] > 0,
        filter_endpoint="analytics",
        **date_filter,
    )


def validate_expense_form(form_data):
    """Check the expense form. Used by both add and edit so the rules are identical.

    Returns (form, errors, amount):
      form   - exactly what the user typed, to show again if there's an error
      errors - {field_name: message}; empty if everything is valid
      amount - the amount as a float rounded to 2 decimals (only meaningful if no errors)
    """
    return validate_money_form(form_data, "category", CATEGORIES, "Please choose a category from the list.")


def validate_income_form(form_data):
    """Check the income form: exactly the same rules as expenses, with a source instead of a category."""
    return validate_money_form(form_data, "source", INCOME_SOURCES, "Please choose a source from the list.")


def validate_money_form(form_data, kind_field, choices, choice_message):
    """Shared rules for expenses and income: amount > 0, kind_field one of `choices`,
    a real date, and an optional description of at most 200 characters."""
    form = {
        "amount": form_data.get("amount", "").strip(),
        kind_field: form_data.get(kind_field, "").strip(),
        "date": form_data.get("date", "").strip(),
        "description": form_data.get("description", "").strip(),
    }
    errors = {}

    # Amount: a real number greater than 0
    amount = None
    try:
        amount = float(form["amount"])
    except ValueError:
        errors["amount"] = "Amount must be a number."
    else:
        if not math.isfinite(amount) or amount <= 0:
            errors["amount"] = "Amount must be greater than 0."
        else:
            amount = round(amount, 2)
            if amount == 0:
                errors["amount"] = "Amount must be at least ₹0.01."

    # Category (expenses) / source (income): must be one of ours
    if form[kind_field] not in choices:
        errors[kind_field] = choice_message

    # Date: must be a real calendar date in YYYY-MM-DD form
    try:
        datetime.strptime(form["date"], "%Y-%m-%d")
    except ValueError:
        errors["date"] = "Please enter a valid date."

    # Description: optional, but not too long
    if len(form["description"]) > MAX_DESCRIPTION_LENGTH:
        errors["description"] = f"Description must be {MAX_DESCRIPTION_LENGTH} characters or fewer."

    return form, errors, amount


def render_expense_form(form, errors, expense=None, status=200):
    """Show the shared add/edit form. Pass `expense` when editing."""
    return render_template(
        "expense_form.html",
        form=form,
        errors=errors,
        categories=CATEGORIES,
        expense=expense,
    ), status


@app.route("/expenses/add", methods=["GET", "POST"])
@login_required
def add_expense_page():
    if request.method == "GET":
        form = {"amount": "", "category": "", "date": date.today().isoformat(), "description": ""}
        return render_expense_form(form, {})

    form, errors, amount = validate_expense_form(request.form)
    if errors:
        return render_expense_form(form, errors, status=400)

    # The user id comes from the session, never from the form.
    warnings = add_expenses_with_budget_check(
        session["user_id"], [(amount, form["category"], form["date"], form["description"])])
    flash("Expense added.", "success")
    for warning in warnings:
        flash(warning, "warning")
    return redirect(url_for("profile"))


@app.route("/expenses/scan", methods=["GET", "POST"])
@login_required
def scan_receipt_page():
    """Upload a receipt photo -> preview of the extracted fields. Saves NOTHING: the preview
    form submits to add_expense_page, so saving needs the user's Confirm click and goes
    through the normal validation (user id from the session)."""
    if not vision_available():   # e.g. Groq without a vision model on this server
        return render_template("scan_receipt.html", result=None, error=RECEIPTS_UNAVAILABLE,
                               unavailable=True, scan_disabled=True, categories=CATEGORIES),             (200 if request.method == "GET" else 503)
    if request.method == "GET":
        return render_template("scan_receipt.html", result=None, error=None, categories=CATEGORIES)

    wait = ai_rate_limit_wait(session["user_id"])
    if wait is not None:
        return render_template("scan_receipt.html", result=None, error=ai_rate_message(wait),
                               categories=CATEGORIES), 429
    upload = request.files.get("receipt")
    data = upload.read(MAX_UPLOAD_BYTES + 1) if upload else b""   # never read more than needed
    try:
        result = scan_receipt(upload.filename if upload else "", data)
    except ReceiptError as e:
        return render_template("scan_receipt.html", result=None, error=str(e), categories=CATEGORIES), 400
    except AIBusy as e:
        return render_template("scan_receipt.html", result=None, error=str(e), categories=CATEGORIES), 429
    except AIError as e:
        return render_template("scan_receipt.html", result=None, error="Receipt scanning unavailable",
                               error_detail=str(e), unavailable=True, categories=CATEGORIES), 503
    finally:
        data = None   # the upload only ever lived in memory; drop it
    return render_template("scan_receipt.html", result=result, error=None, categories=CATEGORIES)


@app.errorhandler(413)
def upload_too_large(error):
    if request.path == url_for("ai_voice"):   # fetch() call: answer with JSON, not a redirect
        return jsonify(error="That recording is too large. Please keep it under 5 MB (about 30 seconds)."), 413
    flash("That image is too large. Please upload a photo under 5 MB.", "error")
    return redirect(url_for("scan_receipt_page"))


@app.route("/expenses/<int:expense_id>/edit", methods=["GET", "POST"])
@login_required
def edit_expense_page(expense_id):
    # Only finds the expense if it belongs to the logged-in user.
    expense = get_expense(expense_id, session["user_id"])
    if expense is None:
        abort(404)  # same response for "someone else's" and "doesn't exist"

    if request.method == "GET":
        form = {
            "amount": f"{expense['amount']:.2f}",
            "category": expense["category"],
            "date": expense["date"],
            "description": expense["description"] or "",
        }
        return render_expense_form(form, {}, expense=expense)

    form, errors, amount = validate_expense_form(request.form)
    if errors:
        return render_expense_form(form, errors, expense=expense, status=400)

    update_expense(
        expense_id, session["user_id"], amount, form["category"], form["date"], form["description"]
    )
    flash("Expense updated.", "success")
    return redirect(url_for("profile"))


@app.route("/expenses/<int:expense_id>/delete", methods=["POST"])
@login_required
def delete_expense_page(expense_id):
    # delete_expense only removes the row if it belongs to the logged-in user.
    if not delete_expense(expense_id, session["user_id"]):
        abort(404)
    flash("Expense deleted.", "success")
    return redirect(url_for("profile"))


# ---------- AI assistant (JSON routes used by static/js/ai_assistant.js) ----------

MAX_AI_MESSAGE_LENGTH = 500

# Rate limit for routes that run a model on the GPU (chat, insights, receipt scan): at most
# AI_RATE_LIMIT requests per user in any AI_RATE_WINDOW seconds. Kept in memory (reset on restart).
AI_RATE_LIMIT = 30
AI_RATE_WINDOW = 10 * 60
AI_RATE_MESSAGE = ("You've sent a lot of AI requests in the last 10 minutes. "
                   "Please take a short break and try again in {minutes}.")
_ai_requests = defaultdict(deque)   # user_id -> times of their recent AI requests
_ai_requests_lock = threading.Lock()


def ai_rate_limit_wait(user_id):
    """Count one AI request for this user. Returns None if allowed, else seconds to wait.
    A refused request is not counted."""
    now = time.monotonic()
    with _ai_requests_lock:
        recent = _ai_requests[user_id]
        while recent and now - recent[0] >= AI_RATE_WINDOW:
            recent.popleft()
        if len(recent) >= AI_RATE_LIMIT:
            return AI_RATE_WINDOW - (now - recent[0])
        recent.append(now)
        return None


def ai_rate_message(wait_seconds):
    minutes = max(1, math.ceil(wait_seconds / 60))
    return AI_RATE_MESSAGE.format(minutes=f"{minutes} minute{'s' if minutes != 1 else ''}")


def ai_json_login_required(view):
    """Like login_required, but answers with JSON instead of redirecting (for fetch calls)."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return jsonify(error="Your session has expired. Please log in again."), 401
        return view(*args, **kwargs)
    return wrapped


def expense_line(e):
    """'₹250 · Transport · 05 Nov — auto' (description part left out if empty)."""
    line = f"{short_inr(e['amount'])} · {e['category']} · {short_date(e['date'])}"
    return line + (f" — {e['description']}" if e["description"] else "")


def expense_row(e):
    """One expense formatted for a preview table row."""
    return {
        "amount": short_inr(e["amount"]),
        "category": e["category"],
        "date": format_date(e["date"]),
        "description": e["description"] or "—",
    }


AI_FIELD_LABELS = {"amount": "Amount", "category": "Category", "date": "Date", "description": "Description"}


def format_field(field, value):
    if field == "amount":
        return short_inr(value)
    if field == "date":
        return short_date(value)
    return value or "—"


def changed_fields(before, after):
    """[{'field': 'Amount', 'before': '₹250', 'after': '₹280'}, ...] for fields that differ."""
    return [
        {"field": label, "before": format_field(key, before[key]), "after": format_field(key, after[key])}
        for key, label in AI_FIELD_LABELS.items()
        if before[key] != after[key]
    ]


def build_ai_card(action_id, action, user_id):
    """Turn a pending action into display data for ai_assistant.js. Returns None if it can't be shown."""
    kind = action["type"]
    if kind == "add":
        return {"id": action_id, "type": "add", "rows": [expense_row(e) for e in action["expenses"]]}
    if kind == "edit":
        return {
            "id": action_id, "type": "edit",
            "label": action["before"]["description"] or action["before"]["category"],
            "row": expense_row(action["before"]),
            "changes": changed_fields(action["before"], action["after"]),
        }
    if kind == "delete":
        return {"id": action_id, "type": "delete", "row": expense_row(action["before"])}
    if kind == "choose":
        options = []
        for expense_id in action["candidate_ids"]:
            expense = get_expense(expense_id, user_id)  # only finds this user's expenses
            if expense is not None:
                options.append({"expense_id": expense_id, "row": expense_row(expense_snapshot(expense))})
        if not options:
            return None
        wanted = ", ".join(
            f"{AI_FIELD_LABELS[k]} → {format_field(k, v)}" for k, v in action["changes"].items()
        )
        return {"id": action_id, "type": "choose", "op": action["op"], "wanted": wanted, "options": options}
    if kind == "add_income":
        return {"id": action_id, "type": "add_income", "rows": [{
            "amount": short_inr(i["amount"]), "source": i["source"],
            "date": format_date(i["date"]), "description": i["description"] or "—",
        } for i in action["incomes"]]}
    if kind == "budget":
        return {
            "id": action_id, "type": "budget",
            "name": "Overall monthly budget" if action["category"] is None else f"{action['category']} budget",
            "before": short_inr(action["before"]) if action["before"] is not None else "not set",
            "after": short_inr(action["after"]),
        }
    return None


def get_ai_pending():
    """This user's pending AI actions: {action_id: action}. Stored in the signed session cookie."""
    return dict(session.get("ai_pending", {}))


def pop_ai_action(data):
    """Remove and return the pending action named in the request, or None if it's unknown/stale."""
    pending = get_ai_pending()
    action = pending.pop(str(data.get("action_id", "")), None)
    session["ai_pending"] = pending
    return action


@app.route("/ai/assistant", methods=["POST"])
@ai_json_login_required
def ai_assistant():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify(error="Please type a message."), 400
    if len(message) > MAX_AI_MESSAGE_LENGTH:
        return jsonify(error=f"Please keep messages under {MAX_AI_MESSAGE_LENGTH} characters."), 400

    user_id = session["user_id"]  # the AI never chooses the user
    wait = ai_rate_limit_wait(user_id)
    if wait is not None:
        AI_LOG.warning("user %s: RATE LIMIT reached (%d AI requests in 10 minutes)", user_id, AI_RATE_LIMIT)
        return jsonify(error=ai_rate_message(wait)), 429

    # A new message replaces any previews that weren't confirmed.
    session.pop("ai_pending", None)

    # Guardrails in code, first: prompt injection, greetings, clearly off-topic requests are
    # answered here and never saved to chat memory (run_expense_assistant checks again, after
    # translating Hindi, and asks the model a yes/no topic question only when the rules can't tell).
    verdict = screen_by_rules(message)
    if verdict:
        kind, reply = verdict
        log = AI_LOG.warning if kind in ("injection", "data_access") else AI_LOG.info
        log("user %s: GUARD %s -> %r (not saved to memory)", user_id, kind.upper(), message[:200])
        return jsonify(reply=reply, cards=[], guard=kind)

    # Reject negative amounts in code, before the AI sees them (the AI might "fix" -20 into 20).
    error = amount_problem(message)
    if error:
        record_chat(user_id, error, user_text=message, kind="error")  # keep it in the chat history
        return jsonify(error=error), 400

    # "Reply in Hindi" setting (localStorage in the browser). Only changes how the reply reads.
    reply_in_hindi = data.get("reply_lang") == "hinglish"
    details = {}   # filled with "understood_as" when a Hindi/Hinglish message was translated
    try:
        reply, actions = run_expense_assistant(
            message, user_id, date.today(), validate_expense_form, CATEGORIES,
            validate_income=validate_income_form, income_sources=INCOME_SOURCES,
            reply_in_hindi=reply_in_hindi, details=details,
        )
    except MessageRejected as e:
        # e.g. a translation code couldn't verify: ask the user instead of guessing.
        record_chat(user_id, str(e), user_text=message, kind="error")
        return jsonify(error=str(e)), 400
    except AIBusy as e:
        return jsonify(error=str(e)), 429          # "AI is busy, try again in a moment."
    except AIError as e:
        return jsonify(error="AI assistant unavailable", detail=str(e)), 503

    # Group all new expenses into one "add" card; every edit/delete/choice gets its own card.
    pending = {}
    adds = [a["expense"] for a in actions if a["type"] == "add"]
    if adds:
        pending[secrets.token_hex(8)] = {"type": "add", "expenses": adds}
    incomes = [a["income"] for a in actions if a["type"] == "add_income"]
    if incomes:
        pending[secrets.token_hex(8)] = {"type": "add_income", "incomes": incomes}
    for action in actions:
        if action["type"] not in ("add", "add_income"):
            pending[secrets.token_hex(8)] = action

    cards = [c for c in (build_ai_card(i, a, user_id) for i, a in pending.items()) if c]
    # Keep the actions on the server until the user clicks Confirm. The browser only gets
    # ids, so what's changed is exactly what was previewed.
    session["ai_pending"] = {c["id"]: pending[c["id"]] for c in cards}
    return jsonify(reply=reply, cards=cards, understood_as=details.get("understood_as"),
                   guard=details.get("guard"), sql=details.get("sql"))


@app.route("/ai/select", methods=["POST"])
@ai_json_login_required
def ai_select():
    """The user picked one expense from a "several match" list: show its edit/delete preview."""
    data = request.get_json(silent=True) or {}
    action = pop_ai_action(data)
    try:
        expense_id = int(data.get("expense_id"))
    except (TypeError, ValueError):
        expense_id = None
    if not action or action["type"] != "choose" or expense_id not in action["candidate_ids"]:
        return jsonify(error="This list is out of date. Please ask the assistant again."), 400

    user_id = session["user_id"]
    expense = get_expense(expense_id, user_id)  # only finds this user's expenses
    if expense is None:
        return jsonify(error="That expense no longer exists. Nothing changed."), 400

    before = expense_snapshot(expense)
    if action["op"] == "delete":
        new_action = {"type": "delete", "expense_id": expense_id, "before": before}
    else:
        after, error = apply_expense_changes(before, action["changes"], validate_expense_form)
        if error:
            return jsonify(error=f"{error} Nothing changed."), 400
        if after == before:
            return jsonify(error="That expense already has those values. Nothing changed."), 400
        new_action = {"type": "edit", "expense_id": expense_id, "before": before, "after": after}

    action_id = secrets.token_hex(8)
    pending = get_ai_pending()
    pending[action_id] = new_action
    session["ai_pending"] = pending
    return jsonify(card=build_ai_card(action_id, new_action, user_id))


@app.route("/ai/confirm", methods=["POST"])
@ai_json_login_required
def ai_confirm():
    """Carry out ONE previewed action. This is the only place AI actions change the database."""
    action = pop_ai_action(request.get_json(silent=True) or {})
    if not action or action["type"] == "choose":
        return jsonify(error="This preview is out of date. Please ask the assistant again."), 400

    user_id = session["user_id"]  # never from the AI or the browser

    if action["type"] == "add":
        # Check every expense again, then save them all (or none).
        checked = []
        for e in action["expenses"]:
            expense, error = apply_expense_changes(e, {}, validate_expense_form)
            if error:
                return jsonify(error="One of the expenses is no longer valid. Nothing changed."), 400
            checked.append(expense)
        warnings = add_expenses_with_budget_check(
            user_id, [(e["amount"], e["category"], e["date"], e["description"]) for e in checked])
        lines = [f"✅ Added {expense_line(e)}" for e in checked] + warnings
        return ai_result(user_id, "\n".join(lines), [e["date"] for e in checked],
                         kind="warning" if warnings else "success")

    if action["type"] == "add_income":
        # Check every income entry again with the income form's rules, then save all or none.
        checked = []
        for i in action["incomes"]:
            form, errors, amount = validate_income_form({**i, "amount": str(i["amount"])})
            if errors:
                return jsonify(error="One of the income entries is no longer valid. Nothing changed."), 400
            checked.append((amount, form["source"], form["date"], form["description"]))
        add_incomes(user_id, checked)
        lines = [f"💵 Added income {short_inr(a)} · {s} · {short_date(d)}" + (f" — {desc}" if desc else "")
                 for a, s, d, desc in checked]
        return ai_result(user_id, "\n".join(lines), [d for _, _, d, _ in checked])

    if action["type"] == "budget":
        # Check again: valid category, amount > 0, and not changed since the preview.
        category, amount, errors = validate_budget_form(
            {"category": action["category"] or "", "amount": str(action["after"])})
        if errors:
            return jsonify(error=f"{' '.join(errors)} Nothing changed."), 400
        if get_budget(user_id, category) != action["before"]:
            return jsonify(error="That budget was changed since the preview. Nothing changed. Please ask again."), 400
        set_budget(user_id, category, amount)
        name = "Overall monthly budget" if category is None else f"{category} budget"
        was = f" (was {short_inr(action['before'])})" if action["before"] is not None else ""
        return ai_result(user_id, f"💰 {name} set to {short_inr(amount)}{was}")

    # Edit / delete: the expense must still be this user's, and unchanged since the preview.
    current = get_expense(action["expense_id"], user_id)
    if current is None:
        return jsonify(error="That expense no longer exists. Nothing changed."), 400
    if expense_snapshot(current) != action["before"]:
        return jsonify(error="That expense was changed since the preview. Nothing changed. Please ask again."), 400

    if action["type"] == "delete":
        delete_expense(action["expense_id"], user_id)
        return ai_result(user_id, f"🗑️ Deleted {expense_line(action['before'])}",
                         [action["before"]["date"]])

    after, error = apply_expense_changes(action["after"], {}, validate_expense_form)
    if error:
        return jsonify(error=f"{error} Nothing changed."), 400
    update_expense(action["expense_id"], user_id, after["amount"], after["category"],
                   after["date"], after["description"])
    before = action["before"]
    changes = ", ".join(f"{c['before']} → {c['after']}" for c in changed_fields(before, after))
    return ai_result(user_id, f"✏️ Updated {before['description'] or before['category']}: {changes}",
                     [before["date"], after["date"]])


def ai_result(user_id, message, dates=(), kind="success"):
    """Send a Confirm/Cancel result to the browser and add it to the user's AI chat, so the
    AI knows it happened ("delete it" can refer to it) and it survives a page refresh."""
    record_chat(user_id, message, kind=kind, dates=dates)
    return jsonify(message=message, kind=kind)


@app.route("/ai/cancel", methods=["POST"])
@ai_json_login_required
def ai_cancel():
    pop_ai_action(request.get_json(silent=True) or {})
    return ai_result(session["user_id"], "Cancelled — nothing changed", kind="note")


@app.route("/ai/history")
@ai_json_login_required
def ai_history():
    """The logged-in user's remembered chat, plus any previews still waiting for Confirm."""
    user_id = session["user_id"]
    pending = get_ai_pending()
    cards = [c for c in (build_ai_card(i, a, user_id) for i, a in pending.items()) if c]
    session["ai_pending"] = {c["id"]: pending[c["id"]] for c in cards}  # drop any that can't show
    return jsonify(messages=get_chat_history(user_id), cards=cards)


@app.route("/analytics/insights", methods=["POST"])
@ai_json_login_required
def analytics_insights():
    """Run the read-only insights workflow (ai_insights.py) for the logged-in user."""
    wait = ai_rate_limit_wait(session["user_id"])
    if wait is not None:
        return jsonify(status="unavailable", error="Too many AI requests", detail=ai_rate_message(wait)), 429
    try:
        result = run_insights(session["user_id"])  # user id only from the session
    except AIBusy as e:
        return jsonify(status="unavailable", error=str(e)), 429
    except AIError as e:
        return jsonify(status="unavailable", error="Insights unavailable", detail=str(e)), 503
    return jsonify(result)


@app.route("/ai/new-chat", methods=["POST"])
@ai_json_login_required
def ai_new_chat():
    """Forget this user's conversation and any previews waiting for Confirm."""
    clear_chat(session["user_id"])
    session.pop("ai_pending", None)
    return jsonify(message="New chat started.")


# ---------- Voice input (voice.py: faster-whisper on the CPU) ----------
# Speech only becomes TEXT in the chat box. Nothing is sent to the AI or saved from here:
# the user reads/fixes the transcript and presses Send, and every change still needs Confirm.

@app.route("/ai/voice", methods=["POST"])
@ai_json_login_required
def ai_voice():
    """Recording → transcript. The audio is only held in memory and dropped right away."""
    upload = request.files.get("audio")
    mode = request.form.get("lang", "en")    # the EN / हिंदी toggle; never auto-detected
    data = upload.read(MAX_AUDIO_BYTES + 1) if upload else b""   # never read more than needed
    try:
        result = transcribe_audio(data, mode)
    except VoiceError as e:
        return jsonify(error=str(e)), 400
    except VoiceUnavailable as e:
        return jsonify(error="Voice input unavailable", detail=str(e)), 503
    finally:
        data = None   # never stored: no file, no database row
    return jsonify(text=result["text"], lang=result["mode"])


@app.route("/ai/voice/status")
@ai_json_login_required
def ai_voice_status():
    """Lets the page say "Downloading voice model (first time only)..." when needed."""
    return jsonify(model_ready=model_is_downloaded())


# ---------- Income (same rules and security as expenses) ----------

def render_income_form(form, errors, income=None, status=200):
    return render_template("income_form.html", form=form, errors=errors,
                           sources=INCOME_SOURCES, income=income), status


@app.route("/income/add", methods=["GET", "POST"])
@login_required
def add_income_page():
    if request.method == "GET":
        form = {"amount": "", "source": "", "date": date.today().isoformat(), "description": ""}
        return render_income_form(form, {})
    form, errors, amount = validate_income_form(request.form)
    if errors:
        return render_income_form(form, errors, status=400)
    # The user id comes from the session, never from the form.
    add_income(session["user_id"], amount, form["source"], form["date"], form["description"])
    flash("Income added.", "success")
    return redirect(url_for("profile"))


@app.route("/income/<int:income_id>/edit", methods=["GET", "POST"])
@login_required
def edit_income_page(income_id):
    income = get_income(income_id, session["user_id"])   # only this user's income
    if income is None:
        abort(404)                                          # same for "someone else's" and "doesn't exist"
    if request.method == "GET":
        form = {"amount": f"{income['amount']:.2f}", "source": income["source"],
                "date": income["date"], "description": income["description"] or ""}
        return render_income_form(form, {}, income=income)
    form, errors, amount = validate_income_form(request.form)
    if errors:
        return render_income_form(form, errors, income=income, status=400)
    update_income(income_id, session["user_id"], amount, form["source"], form["date"], form["description"])
    flash("Income updated.", "success")
    return redirect(url_for("profile"))


@app.route("/income/<int:income_id>/delete", methods=["POST"])
@login_required
def delete_income_page(income_id):
    if not delete_income(income_id, session["user_id"]):   # only removes this user's row
        abort(404)
    flash("Income deleted.", "success")
    return redirect(url_for("profile"))


@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


@app.route("/logout")
def logout():
    if "user_id" in session:
        clear_chat(session["user_id"])  # forget this user's AI conversation
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=config.DEBUG, port=config.PORT)
