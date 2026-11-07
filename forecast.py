"""Month-end spending forecast for Spendly (classic statistics / simulation, no LLM).

How it works (forecast_month)
-----------------------------
The rest of the month is split into two parts:

  1. FIXED MONTHLY ITEMS (find_recurring): descriptions that appear once a month in at least 2 of
     the last 3 complete months (rent on the 1st, the electricity bill around the 6th, Netflix on
     the 20th). If one hasn't happened yet this month, its typical (median) amount is expected on
     its typical day, with the chance it had in the past (3 of 3 months -> always).
  2. EVERYDAY SPENDING: daily totals of everything else over the last HISTORY_DAYS days, kept
     separately for weekdays and weekends. Days with no spending count as ₹0. Recent days weigh
     more (the weight halves every HALF_LIFE_DAYS days), so changed habits show up quickly.
     One-off unusual expenses (flagged by anomaly.py, unless marked "This is normal") are left
     out: a ₹32,000 phone is not an everyday pattern.

Then the rest of the month is played out SIMULATIONS times: every remaining weekday draws one
real past weekday total, every weekend day a real weekend total, plus the fixed items. Each run is
one possible month-end total. The MEDIAN run is the forecast; the 10th-90th percentile of the runs
is the range ("80% range"). Showing a range is honest: one number would hide that a doctor's visit
or one big weekend can move the total a lot. The backtest checks the range really contains the
actual total about 80% of the time.

Too little history -> {"status": "not_enough_data"}: never a guess.
Run `python evals/ml_eval.py` for the backtest (MAE, % error, range coverage) against two baselines:
a run-rate rule and scikit-learn linear regression (linear_regression_forecast).
"""
import calendar
import math
import random
import statistics
from collections import defaultdict
from datetime import date, timedelta

from anomaly import format_inr

SIMULATIONS = 1000
HISTORY_DAYS = 90           # everyday spending is learned from the last 90 days
HALF_LIFE_DAYS = 45         # a day 45 days ago counts half as much as yesterday
MIN_HISTORY_DAYS = 42       # ~6 weeks of history ...
MIN_EXPENSES = 30           # ... and 30 expenses, or no forecast at all
LOW_PCT, HIGH_PCT = 10, 90  # the range shown: 80% of the simulated months fall inside it
RECURRING_MONTHS = 3        # fixed items are looked for in the last 3 complete months
TOP_CATEGORIES = 3
SEED = 20261107             # fixed: the same data always gives the same forecast
NOT_ENOUGH = "Not enough data for a forecast yet"


def month_end(day):
    return date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])


def month_start(day, months_back=0):
    year, month = day.year, day.month - months_back
    while month <= 0:
        month, year = month + 12, year - 1
    return date(year, month, 1)


def same_days_last_month(today):
    """(first, last) day of the same days last month, for a fair month-to-date comparison:
    7 Nov -> (1 Oct, 7 Oct). Clamped to last month's end: 31 Mar -> (1 Feb, 28 Feb).
    (Comparing 7 days of November with ALL of October once showed '▼ 68.3%'.)"""
    last_start = month_start(today, 1)
    last_end = month_start(today) - timedelta(days=1)
    return last_start, min(last_start + timedelta(days=today.day - 1), last_end)


def same_days_label(today):
    """'1–7 Nov vs 1–7 Oct' (the label under a month-to-date % change)."""
    first, last = same_days_last_month(today)
    return f"1–{today.day} {today:%b} vs 1–{last.day} {last:%b}"


def is_weekend(day):
    return day.weekday() >= 5      # Saturday, Sunday


def round_to(value, step=100):
    """Forecasts are shown rounded (₹48,200, not ₹48,213.57): they are estimates."""
    return float(round(value / step) * step)


def _key(description):
    return " ".join((description or "").lower().split())


def _parse(expenses):
    return [{**e, "amount": float(e["amount"]), "day": date.fromisoformat(e["date"])} for e in expenses]


def history_problem(expenses, as_of):
    """None if there is enough history for a forecast, else a short reason (shown to the user)."""
    known = [e for e in expenses if e["day"] <= as_of]
    if not known:
        return "Add some expenses first."
    days = (as_of - min(e["day"] for e in known)).days + 1
    needs = []
    if days < MIN_HISTORY_DAYS:
        needs.append(f"about 6 weeks of history (you have {days} day{'s' if days != 1 else ''})")
    if len(known) < MIN_EXPENSES:
        needs.append(f"at least {MIN_EXPENSES} expenses (you have {len(known)})")
    return ("It needs " + " and ".join(needs) + ".") if needs else None


MAX_DAY_SPREAD = 7          # a fixed item comes on (about) the same day every month


def find_recurring(expenses, as_of):
    """Fixed monthly items: {key: {"description", "category", "amount", "day", "chance"}}.

    Looks at this month (so far) and the last 3 complete months the history covers. A description
    counts when it appears EXACTLY once in each month it appears in, in at least 3 of those months
    (or in all of them when there are only 2), always within MAX_DAY_SPREAD days of the month.
    Backtest lessons: with only complete months, a 7-week history found no rent at all (a ₹16k
    "rent day" in the everyday pool); with "2 of 3 months", random items (vegetables, a donation)
    became monthly bills. chance = how many of the complete months had it."""
    first = min((e["day"] for e in expenses), default=as_of)
    current = month_start(as_of)
    complete = [m for m in (month_start(as_of, k) for k in range(1, RECURRING_MONTHS + 1))
                if month_end(m) >= first]
    months = [current] + complete
    needed = max(2, min(3, len(months)))
    if len(months) < 2:
        return {}
    seen = defaultdict(lambda: defaultdict(list))   # key -> month -> [expenses]
    for e in expenses:
        m = date(e["day"].year, e["day"].month, 1)
        if m in months and e["day"] <= as_of and _key(e["description"]):
            seen[_key(e["description"])][m].append(e)
    items = {}
    for key, by_month in seen.items():
        if len(by_month) < needed or any(len(rows) != 1 for rows in by_month.values()):
            continue
        rows = [rows[0] for rows in by_month.values()]
        days = [e["day"].day for e in rows]
        if max(days) - min(days) > MAX_DAY_SPREAD:
            continue
        latest = max(rows, key=lambda e: e["day"])
        in_complete = sum(1 for m in by_month if m != current)
        items[key] = {
            "description": latest["description"], "category": latest["category"],
            "amount": statistics.median(e["amount"] for e in rows),
            "day": int(statistics.median(days)),
            "chance": in_complete / len(complete) if complete else 1.0,
        }
    return items


def _percentile(sorted_values, pct):
    """Linear-interpolated percentile of an already sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def _daily_pools(expenses, as_of, recurring, known_end, one_offs=frozenset()):
    """Everyday spending per past day (fixed items and one-off unusual expenses left out), split
    weekday / weekend, with recency weights. Days without expenses are ₹0 days."""
    first = min(e["day"] for e in expenses)
    start = max(first, as_of - timedelta(days=HISTORY_DAYS))
    totals = defaultdict(float)
    for e in expenses:
        if (start <= e["day"] <= known_end and _key(e["description"]) not in recurring
                and e.get("id") not in one_offs):
            totals[e["day"]] += e["amount"]
    pools = {False: ([], []), True: ([], [])}    # weekend? -> (values, weights)
    day = start
    while day <= known_end:
        values, weights = pools[is_weekend(day)]
        values.append(totals.get(day, 0.0))
        weights.append(0.5 ** ((as_of - day).days / HALF_LIFE_DAYS))
        day += timedelta(days=1)
    return pools


def one_off_ids(expenses, as_of, dismissed=None):
    """Ids of expenses the anomaly detector flags (judged only on history up to as_of). They are
    left out of the everyday pattern: a ₹32,000 phone is not something to expect every month.
    One flagged day in the pool once stretched the range to ₹36,800–₹73,300. Expenses the user
    marked "This is normal" stay in."""
    from anomaly import find_anomalies
    known = [e for e in expenses if e["date"] <= as_of.isoformat() and e.get("id") is not None]
    return frozenset(find_anomalies(known, dismissed))


def forecast_month(expenses, as_of, budget=None, partial_today=True, category=None,
                   simulations=SIMULATIONS, seed=SEED, check_history=True, one_offs=None):
    """Forecast the month-end total of as_of's month (for one category, or all spending).

    expenses      - dicts with amount, category, date, description (rows up to as_of; later
                    rows are ignored, so a backtest can pass the full history).
    as_of         - the forecast's "today".
    budget        - the monthly limit to compare with (or None).
    partial_today - True in the app: today is still running, so it is simulated too (minus what
                    was already spent today). False in the backtest: day `as_of` is complete.
    one_offs      - ids of one-off unusual expenses to leave out of the everyday pattern
                    (None = work them out with the anomaly detector).
    Returns a dict with status "ok" or "not_enough_data" (see build_forecast for the fields).
    """
    if one_offs is None:
        one_offs = one_off_ids(expenses, as_of)
    rows = [e for e in _parse(expenses) if e["day"] <= as_of]
    if check_history:
        problem = history_problem(rows, as_of)
        if problem:
            return {"status": "not_enough_data", "message": NOT_ENOUGH, "detail": problem}
    if category:
        rows = [e for e in rows if e["category"] == category]
    if not rows:
        return {"status": "not_enough_data", "message": NOT_ENOUGH,
                "detail": f"There are no {category or ''} expenses yet.".replace("  ", " ")}

    start, end = month_start(as_of), month_end(as_of)
    this_month = [e for e in rows if e["day"] >= start]
    spent = round(sum(e["amount"] for e in this_month), 2)
    recurring = find_recurring(rows, as_of)
    paid = {_key(e["description"]) for e in this_month}
    first_open = as_of if partial_today else as_of + timedelta(days=1)
    open_days = [first_open + timedelta(days=i) for i in range((end - first_open).days + 1)]
    known_end = as_of - timedelta(days=1) if partial_today else as_of
    pools = _daily_pools(rows, as_of, recurring, known_end, one_offs)
    spent_today = sum(e["amount"] for e in this_month
                      if e["day"] == as_of and _key(e["description"]) not in recurring)

    # Fixed items still to come this month, on their usual day (or the next open day if late).
    upcoming = []
    for key, item in recurring.items():
        if key in paid or not open_days:
            continue
        day = max(date(as_of.year, as_of.month, min(item["day"], end.day)), open_days[0])
        upcoming.append({**item, "date": day})
    upcoming.sort(key=lambda i: i["date"])

    rng = random.Random(f"{seed}-{as_of}-{category}")
    cum_weights = {}
    for weekend, (values, weights) in pools.items():
        acc, total = [], 0.0
        for w in weights:
            total += w
            acc.append(total)
        cum_weights[weekend] = acc
    index = {d: i for i, d in enumerate(open_days)}
    paths = []
    for _ in range(simulations if open_days else 1):
        daily = [0.0] * len(open_days)
        for weekend in (False, True):
            days = [d for d in open_days if is_weekend(d) == weekend]
            values = pools[weekend][0]
            if not days or not values:
                continue
            # Two-level bootstrap. 1) Resample the past days themselves: a plausible "level" of
            # spending (we only have ~60 weekdays, so the true daily rate is uncertain too).
            # 2) Draw the remaining days from that resample. With step 2 alone the range was
            # overconfident: only 44% of real month-ends fell inside the "80%" range.
            level = rng.choices(values, cum_weights=cum_weights[weekend], k=len(values))
            draws = rng.choices(level, k=len(days))
            for d, v in zip(days, draws):
                if partial_today and d == as_of:
                    v = max(0.0, v - spent_today)    # today: only what a usual day still adds
                daily[index[d]] += v
        for item in upcoming:
            if rng.random() < item["chance"]:
                daily[index[item["date"]]] += item["amount"]
        path, total = [], spent
        for v in daily:
            total += v
            path.append(total)
        paths.append(path)

    totals = sorted(p[-1] if p else spent for p in paths)
    expected = _percentile(totals, 50)
    low, high = _percentile(totals, LOW_PCT), _percentile(totals, HIGH_PCT)
    bands = {"p10": [], "p50": [], "p90": []}
    for i in range(len(open_days)):
        column = sorted(p[i] for p in paths)
        for name, pct in (("p10", LOW_PCT), ("p50", 50), ("p90", HIGH_PCT)):
            bands[name].append(round(_percentile(column, pct), 2))

    result = {
        "status": "ok", "as_of": as_of.isoformat(), "category": category,
        "month_label": f"{as_of:%B %Y}", "days_in_month": end.day,
        "days_left": (end - as_of).days, "spent": spent,
        "expected_raw": round(expected, 2), "low_raw": round(low, 2), "high_raw": round(high, 2),
        "expected": max(round_to(expected), spent), "low": max(round_to(low), spent),
        "high": max(round_to(high), spent),
        "upcoming": [{"description": i["description"], "amount": i["amount"],
                      "date": i["date"].isoformat()} for i in upcoming],
        "path": {"dates": [d.isoformat() for d in open_days], **bands},
        "budget": None,
    }
    if budget:
        result["budget"] = budget_outlook(budget, spent, totals, open_days, bands["p50"], category)
    return result


def budget_outlook(limit, spent, totals, open_days, median_path, category=None):
    """Will this month go over `limit`? Uses the simulated months."""
    name = format_inr(limit) + (f" {category}" if category else "") + " budget"
    chance = sum(t > limit for t in totals) / len(totals) if totals else 0.0
    out = {"limit": limit, "chance": round(chance, 2), "date": None}
    if spent > limit:
        out.update(status="over", message=f"You've already gone over your {name} "
                                          f"({format_inr(spent)} spent so far).")
    elif chance >= 0.5:
        crossing = next((d for d, v in zip(open_days, median_path) if v > limit), None)
        out["date"] = crossing.isoformat() if crossing else None
        when = f" around {crossing:%d %b}" if crossing else " by the end of the month"
        out.update(status="likely", message=f"You'll likely go over your {name}{when}.")
    elif chance >= 0.1:
        out.update(status="maybe", message=f"You might go over your {name} if spending runs high "
                                           f"(about {round(chance * 100):.0f}% of simulated months did).")
    else:
        out.update(status="ok", message=f"You're on track to stay within your {name}.")
    return out


def build_forecast(expenses, as_of, budgets=None, partial_today=True, simulations=SIMULATIONS,
                   dismissed=None):
    """The whole forecast: all spending plus the top 3 categories (by the last 90 days' spending).
    budgets   - {None: overall limit, "Food": limit, ...} (this month's budgets).
    dismissed - the user's "This is normal" marks (those expenses stay in the everyday pattern)."""
    budgets = budgets or {}
    one_offs = one_off_ids(expenses, as_of, dismissed)
    overall = forecast_month(expenses, as_of, budgets.get(None), partial_today, simulations=simulations,
                             one_offs=one_offs)
    if overall["status"] != "ok":
        return overall
    since = (as_of - timedelta(days=HISTORY_DAYS)).isoformat()
    by_category = defaultdict(float)
    for e in expenses:
        # One-offs don't count: a ₹32,000 phone made Shopping a "top category" for 3 months.
        if since <= e["date"] <= as_of.isoformat() and e.get("id") not in one_offs:
            by_category[e["category"]] += float(e["amount"])
    top = sorted(by_category, key=lambda c: (-by_category[c], c))[:TOP_CATEGORIES]
    overall["categories"] = [
        forecast_month(expenses, as_of, budgets.get(c), partial_today, category=c,
                       simulations=simulations, check_history=False, one_offs=one_offs)
        for c in top
    ]
    overall["categories"] = [c for c in overall["categories"] if c["status"] == "ok"]
    return overall


# ---------- Sentences (used by the dashboard and the AI assistant; numbers from the forecast) ----------

def forecast_sentence(f, category=None):
    """'By the end of November 2026 you'll likely spend about ₹48,200 (probably between ₹44,900
    and ₹52,300). So far: ₹21,450, with 23 days left.'"""
    if f["status"] != "ok":
        return f"{NOT_ENOUGH}. {f.get('detail', '')}".strip()
    on = f" on {category}" if category else ""
    if f["days_left"] == 0 and f["low"] == f["high"]:
        return f"You've spent {format_inr(f['spent'])}{on} in {f['month_label']} so far, and it's the last day."
    left = f"{f['days_left']} day{'s' if f['days_left'] != 1 else ''} left"
    spread = ("" if f["low"] == f["high"]
              else f" (probably between {format_inr(f['low'])} and {format_inr(f['high'])})")
    return (f"By the end of {f['month_label']} you'll likely spend about {format_inr(f['expected'])}{on}"
            f"{spread}. So far: {format_inr(f['spent'])}, with {left}.")


# ---------- Reading the database (all SQL is in database/db.py) ----------

def current_budgets(user_id, today):
    from database.db import get_budget_status
    rows = get_budget_status(user_id, month_start(today).isoformat(), month_end(today).isoformat())
    return {r["category"]: r["monthly_limit"] for r in rows}


def get_forecast(user_id, today=None):
    """This user's forecast for the current month (today counts as still running)."""
    from database.db import get_anomaly_dismissals, get_expenses_up_to
    today = today or date.today()
    expenses = get_expenses_up_to(user_id, today.isoformat())
    return build_forecast(expenses, today, current_budgets(user_id, today), partial_today=True,
                          dismissed=get_anomaly_dismissals(user_id))


def get_category_forecast(user_id, category, today=None):
    """One category's forecast (any category, not only the top 3)."""
    from database.db import get_anomaly_dismissals, get_expenses_up_to
    today = today or date.today()
    expenses = get_expenses_up_to(user_id, today.isoformat())
    overall_check = history_problem(_parse(expenses), today)
    if overall_check:
        return {"status": "not_enough_data", "message": NOT_ENOUGH, "detail": overall_check}
    return forecast_month(expenses, today, current_budgets(user_id, today).get(category), True,
                          category=category, check_history=False,
                          one_offs=one_off_ids(expenses, today, get_anomaly_dismissals(user_id)))


# ---------- Baselines for the backtest (evals/ml_eval.py) ----------

def run_rate_forecast(expenses, as_of):
    """The simplest rule: spent so far / days gone x days in the month."""
    start = month_start(as_of)
    spent = sum(float(e["amount"]) for e in expenses if start.isoformat() <= e["date"] <= as_of.isoformat())
    return spent / as_of.day * month_end(as_of).day


def linear_regression_forecast(expenses, as_of):
    """scikit-learn LinearRegression on the same everyday daily totals (features: day of the week
    + a time trend, recency-weighted), plus the same fixed monthly items. None if scikit-learn
    isn't installed. Used only to compare models in the backtest."""
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return None
    rows = [e for e in _parse(expenses) if e["day"] <= as_of]
    recurring = find_recurring(rows, as_of)
    one_offs = one_off_ids(expenses, as_of)
    first = min(e["day"] for e in rows)
    start = max(first, as_of - timedelta(days=HISTORY_DAYS))
    totals = defaultdict(float)
    for e in rows:
        if e["day"] >= start and _key(e["description"]) not in recurring and e.get("id") not in one_offs:
            totals[e["day"]] += e["amount"]

    def features(day):
        return [1.0 if day.weekday() == k else 0.0 for k in range(7)] + [(day - as_of).days / 30]

    days = [start + timedelta(days=i) for i in range((as_of - start).days + 1)]
    model = LinearRegression(fit_intercept=False)
    model.fit([features(d) for d in days], [totals.get(d, 0.0) for d in days],
              sample_weight=[0.5 ** ((as_of - d).days / HALF_LIFE_DAYS) for d in days])
    end = month_end(as_of)
    future = [as_of + timedelta(days=i) for i in range(1, (end - as_of).days + 1)]
    predicted = sum(max(0.0, v) for v in model.predict([features(d) for d in future])) if future else 0.0
    this_month = [e for e in rows if e["day"] >= month_start(as_of)]
    paid = {_key(e["description"]) for e in this_month}
    fixed = sum(i["amount"] * i["chance"] for k, i in recurring.items() if k not in paid and future)
    return sum(e["amount"] for e in this_month) + predicted + fixed


if __name__ == "__main__":
    # Quick look: python forecast.py <email>  (read-only)
    import sys
    from database.db import get_user_by_email
    user = get_user_by_email(sys.argv[1] if len(sys.argv) > 1 else "ml_demo@spendly.com")
    if user is None:
        sys.exit("No such user. Run: python scripts/seed_ml_demo.py")
    f = get_forecast(user["id"])
    print(forecast_sentence(f))
    if f["status"] == "ok":
        if f["budget"]:
            print(f["budget"]["message"])
        for c in f["categories"]:
            print(" -", forecast_sentence(c, c["category"]))
