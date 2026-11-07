"""Ground truth: the TRUE answer of each case, computed here with SQL on the eval database copy.

This deliberately does NOT reuse the app's own query functions or its period parser (that would
test the app against itself). Periods are worked out with plain date maths below, and every
number comes from one of the fixed, parameterized queries in this file (read-only).
"""
import calendar
import sqlite3
from datetime import date, timedelta

# ---------- Periods (independent of ai_agent.resolve_period) ----------

PERIODS = ["none", "today", "yesterday", "day_before_yesterday", "this_week", "last_week",
           "this_month", "last_month", "empty_month"]


def _month(year, month):
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _add_months(day, n):
    m = day.year * 12 + day.month - 1 + n
    return m // 12, m % 12 + 1


def period_range(token, today):
    """(date_from, date_to) as ISO strings, or (None, None) for all time."""
    if token in (None, "", "none"):
        return None, None
    if token == "today":
        a = b = today
    elif token == "yesterday":
        a = b = today - timedelta(days=1)
    elif token == "day_before_yesterday":
        a = b = today - timedelta(days=2)
    elif token == "this_week":
        a = today - timedelta(days=today.weekday())
        b = a + timedelta(days=6)
    elif token == "last_week":
        a = today - timedelta(days=today.weekday() + 7)
        b = a + timedelta(days=6)
    elif token == "this_month":
        a, b = _month(today.year, today.month)
    elif token == "last_month":
        a, b = _month(*_add_months(today, -1))
    elif token == "empty_month":
        a, b = _month(*_add_months(today, -8))
    else:
        raise ValueError(f"unknown period {token!r}")
    return a.isoformat(), b.isoformat()


def placeholders(today):
    """Values for {…} in case messages: month names that match the period tokens."""
    last = date(*_add_months(today, -1), 1)
    empty = date(*_add_months(today, -8), 1)
    return {
        "this_month_name": f"{today:%B}",
        "this_month_label": f"{today:%B %Y}",
        "last_month_label": f"{last:%B %Y}",
        "empty_month_label": f"{empty:%B %Y}",
    }


def resolve_date(token, today):
    """'today' / 'yesterday' / 'day_before_yesterday' (or an ISO date) -> ISO date."""
    offsets = {"today": 0, "yesterday": 1, "day_before_yesterday": 2}
    if token in offsets:
        return (today - timedelta(days=offsets[token])).isoformat()
    return token


# ---------- Fixed read-only queries ----------
# Every filter is optional: a NULL parameter switches it off.
_FILTER = ("FROM expenses WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
           "AND (? IS NULL OR category = ?) AND (? IS NULL OR amount > ?) AND (? IS NULL OR amount >= ?) "
           "AND (? IS NULL OR amount < ?) AND (? IS NULL OR amount <= ?) AND (? IS NULL OR amount = ?) "
           "AND (? IS NULL OR (CASE WHEN strftime('%w', date) IN ('0','6') THEN 'weekend' "
           "ELSE 'weekday' END) = ?)")
AGG_SQL = ("SELECT COALESCE(ROUND(SUM(amount), 2), 0) AS sum, ROUND(AVG(amount), 2) AS avg, "
           "COUNT(*) AS count, MIN(amount) AS min, MAX(amount) AS max, "
           "ROUND(COALESCE(SUM(amount), 0) * 1.0 / ?, 2) AS per_day " + _FILTER)
ROWS_SQL = "SELECT amount, category, date, description " + _FILTER + " ORDER BY date, id"
CATEGORY_GROUP_SQL = ("SELECT category AS name, COUNT(*) AS count, ROUND(SUM(amount), 2) AS sum "
                      + _FILTER + " GROUP BY category")
DOW_GROUP_SQL = ("SELECT CASE strftime('%w', date) WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' "
                 "WHEN '2' THEN 'Tuesday' WHEN '3' THEN 'Wednesday' WHEN '4' THEN 'Thursday' "
                 "WHEN '5' THEN 'Friday' ELSE 'Saturday' END AS name, COUNT(*) AS count, "
                 "ROUND(SUM(amount), 2) AS sum " + _FILTER + " GROUP BY name")
SHARE_SQL = ("SELECT CASE WHEN total > 0 THEN ROUND(part * 100.0 / total, 1) END FROM ("
             "SELECT COALESCE(SUM(CASE WHEN category = ? THEN amount END), 0) AS part, "
             "COALESCE(SUM(amount), 0) AS total FROM expenses WHERE user_id = ? "
             "AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?))")
INCOME_SQL = ("SELECT COALESCE(ROUND(SUM(amount), 2), 0) FROM income WHERE user_id = ? "
              "AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?)")
FIRST_DATE_SQL = "SELECT MIN(date) FROM expenses WHERE user_id = ?"
RANGE_SUM_SQL = ("SELECT COALESCE(ROUND(SUM(amount), 2), 0) FROM expenses "
                 "WHERE user_id = ? AND date >= ? AND date <= ?")


class GroundTruth:
    """Answers truth specs from cases.json for one user, read-only."""

    def __init__(self, db_path, user_id, today):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.user_id, self.today = user_id, today

    def close(self):
        self.conn.close()

    def _params(self, spec):
        a, b = period_range(spec.get("period"), self.today)
        p = [self.user_id, a, a, b, b, spec.get("category"), spec.get("category")]
        for key in ("gt", "ge", "lt", "le", "eq", "day_type"):
            p += [spec.get(key), spec.get(key)]
        return p

    def _days(self, spec):
        """Days in the period, counted up to today (all time: from the first expense)."""
        a, b = period_range(spec.get("period"), self.today)
        if a is None:
            a = self.conn.execute(FIRST_DATE_SQL, (self.user_id,)).fetchone()[0] or self.today.isoformat()
            b = self.today.isoformat()
        start, end = date.fromisoformat(a), date.fromisoformat(b)
        if start <= self.today:
            end = min(end, self.today)
        return max((end - start).days + 1, 1)

    def value(self, spec):
        """One number: sum / avg / count / min / max / per_day / share / income / saved / diff."""
        agg = spec["agg"]
        if agg == "diff":
            return round(abs(self.value(spec["a"]) - self.value(spec["b"])), 2)
        if agg == "share":
            a, b = period_range(spec.get("period"), self.today)
            return self.conn.execute(SHARE_SQL, (spec["category"], self.user_id, a, a, b, b)).fetchone()[0]
        if agg in ("income", "saved"):
            a, b = period_range(spec.get("period"), self.today)
            income = self.conn.execute(INCOME_SQL, (self.user_id, a, a, b, b)).fetchone()[0]
            if agg == "income":
                return income
            spent = self.value({**spec, "agg": "sum"})
            return round(income - spent, 2)
        row = self.conn.execute(AGG_SQL, (self._days(spec), *self._params(spec))).fetchone()
        return dict(zip(["sum", "avg", "count", "min", "max", "per_day"], row))[agg]

    def rows(self, spec):
        """The matching expenses: [(amount, category, date, description), ...]."""
        return self.conn.execute(ROWS_SQL, self._params(spec)).fetchall()

    def top_names(self, spec):
        """Names tied for the top (or bottom) value: categories or weekdays, by count or sum."""
        sql = DOW_GROUP_SQL if spec.get("group") == "dow" else CATEGORY_GROUP_SQL
        groups = self.conn.execute(sql, self._params(spec)).fetchall()
        if not groups:
            return []
        index = 1 if spec.get("by", "sum") == "count" else 2
        pick = min if spec.get("order") == "min" else max
        best = pick(g[index] for g in groups)
        return sorted(g[0] for g in groups if g[index] == best)

    # ---------- Insights (analytics card) ----------

    def _sum(self, a, b):
        return self.conn.execute(RANGE_SUM_SQL, (self.user_id, a.isoformat(), b.isoformat())).fetchone()[0]

    def insights_numbers(self):
        """The fair month-to-date change (vs the SAME days of last month), the old misleading one
        (whole month vs ALL of last month), and this month's income / savings / savings rate."""
        t = self.today
        this_start, this_end = _month(t.year, t.month)
        last_start, last_end = _month(*_add_months(t, -1))
        same_end = date(last_start.year, last_start.month, min(t.day, last_end.day))
        to_date, same_last = self._sum(this_start, t), self._sum(last_start, same_end)
        this_full, last_full = self._sum(this_start, this_end), self._sum(last_start, last_end)
        income = self.conn.execute(INCOME_SQL, (self.user_id, this_start.isoformat(), this_start.isoformat(),
                                                this_end.isoformat(), this_end.isoformat())).fetchone()[0]
        saved = round(income - this_full, 2)
        return {
            "same_days_pct": round((to_date - same_last) * 100 / same_last, 1) if same_last else None,
            "old_pct": round((this_full - last_full) * 100 / last_full, 1) if last_full else None,
            "same_days_note": f"1–{t.day} {t:%b} vs 1–{same_end.day} {same_end:%b}",
            "income": income,
            "saved": saved,
            "rate": round(saved * 100 / income, 1) if income else None,
        }
