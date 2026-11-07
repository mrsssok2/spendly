"""All database code for Spendly (SQLite, no ORM)."""
import os
import pathlib
import re
import sqlite3
import time
from datetime import datetime

from werkzeug.security import generate_password_hash

import config

# From SPENDLY_DB_PATH (config.py). The tests set it to a temporary file so the real spendly.db
# is never touched; in Docker it points into the named volume (/app/data/spendly.db).
DB_PATH = config.DB_PATH


def get_db():
    """Open a connection to the SQLite database.

    Rows behave like dicts (row["name"]), and foreign keys are enforced.
    SQLite turns foreign keys OFF by default, so we enable them on every connection.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def check_database():
    """For /health: True if the database file can be opened and answers a trivial query."""
    try:
        conn = get_db()
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
        return True
    except sqlite3.Error:
        return False


def init_db():
    """Create the tables if they don't already exist."""
    conn = get_db()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS expenses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                amount      REAL    NOT NULL CHECK (amount > 0),
                category    TEXT    NOT NULL,
                date        TEXT    NOT NULL,  -- stored as YYYY-MM-DD
                description TEXT,
                created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            -- Money coming in. Kept separate from expenses (never as negative expenses).
            CREATE TABLE IF NOT EXISTS income (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                amount      REAL    NOT NULL CHECK (amount > 0),
                source      TEXT    NOT NULL,
                date        TEXT    NOT NULL,  -- stored as YYYY-MM-DD
                description TEXT,
                created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            -- One row per budget: category NULL = the overall monthly budget.
            CREATE TABLE IF NOT EXISTS budgets (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                category      TEXT,
                monthly_limit REAL    NOT NULL CHECK (monthly_limit > 0),
                created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            -- At most one budget per user per category, and one overall budget. (A plain
            -- UNIQUE(user_id, category) would allow many NULLs, i.e. many overall budgets.)
            CREATE UNIQUE INDEX IF NOT EXISTS idx_budgets_user_category
                ON budgets (user_id, COALESCE(category, ''));

            -- "This is normal": an expense the user said is fine, so it is never flagged as
            -- unusual again. The amount is kept: if the expense is edited, it is judged again.
            CREATE TABLE IF NOT EXISTS anomaly_dismissals (
                user_id    INTEGER NOT NULL,
                expense_id INTEGER NOT NULL,
                amount     REAL    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, expense_id),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (expense_id) REFERENCES expenses (id) ON DELETE CASCADE
            );

            """
        )
        conn.commit()
    finally:
        conn.close()


def create_user(name, email, password):
    """Create a new user with a hashed password.

    Returns the new user's id, or None if the email is already registered.
    """
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, generate_password_hash(password)),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        # The UNIQUE constraint on email was violated.
        return None
    finally:
        conn.close()


def get_user_by_email(email):
    """Return the user row with this email, or None if there isn't one."""
    conn = get_db()
    try:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    finally:
        conn.close()


def add_expense(user_id, amount, category, date, description):
    """Save a new expense for a user and return its id."""
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO expenses (user_id, amount, category, date, description) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, amount, category, date, description or None),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def add_expenses(user_id, expenses):
    """Save several expenses for a user in one transaction: all are saved, or none.

    expenses - list of (amount, category, date, description) tuples.
    """
    conn = get_db()
    try:
        with conn:  # commits if every insert works, rolls back if any fails
            conn.executemany(
                "INSERT INTO expenses (user_id, amount, category, date, description) "
                "VALUES (?, ?, ?, ?, ?)",
                [(user_id, amount, category, date, description or None)
                 for amount, category, date, description in expenses],
            )
    finally:
        conn.close()


def search_expenses(user_id, description=None, category=None, date=None, amount=None,
                    date_from=None, date_to=None, sort="newest", limit=11):
    """Find one user's expenses matching every given filter (None = ignore that filter).

    description - case-insensitive "contains" match, e.g. "auto" matches "Auto rickshaw".
    date        - one exact day; date_from / date_to - an inclusive range.
    sort        - "newest" (default), "biggest" or "smallest" first.
    At most `limit` rows. Always limited to this user's expenses.
    """
    pattern = None
    if description:
        # Escape LIKE wildcards so "%" or "_" typed by the user are matched literally.
        escaped = description.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
    conn = get_db()
    try:
        return conn.execute(
            "SELECT id, amount, category, date, description FROM expenses "
            "WHERE user_id = ? "
            "AND (? IS NULL OR description LIKE ? ESCAPE '\\') "
            "AND (? IS NULL OR category = ?) "
            "AND (? IS NULL OR date = ?) "
            "AND (? IS NULL OR date >= ?) "
            "AND (? IS NULL OR date <= ?) "
            "AND (? IS NULL OR ROUND(amount, 2) = ROUND(?, 2)) "
            # The sort order is chosen with a parameter too, so no SQL is built from strings.
            "ORDER BY CASE ? WHEN 'biggest' THEN -amount WHEN 'smallest' THEN amount ELSE 0 END, "
            "date DESC, id DESC LIMIT ?",
            (user_id, pattern, pattern, category, category, date, date,
             date_from, date_from, date_to, date_to, amount, amount, sort, limit),
        ).fetchall()
    finally:
        conn.close()


def get_spending_summary(user_id, date_from=None, date_to=None, category=None):
    """Total, count and per-category breakdown of one user's spending (all filters optional).

    Returns {"total": 470.0, "count": 3,
             "categories": [{"category": "Food", "total": 470.0, "count": 3}, ...]}
    with the biggest category first. All maths is done by SQLite.
    """
    params = (user_id, date_from, date_from, date_to, date_to, category, category)
    conn = get_db()
    try:
        totals = conn.execute(
            "SELECT COALESCE(ROUND(SUM(amount), 2), 0) AS total, COUNT(*) AS count "
            "FROM expenses "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
            "AND (? IS NULL OR category = ?)",
            params,
        ).fetchone()
        categories = conn.execute(
            "SELECT category, ROUND(SUM(amount), 2) AS total, COUNT(*) AS count "
            "FROM expenses "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
            "AND (? IS NULL OR category = ?) "
            "GROUP BY category ORDER BY total DESC, category ASC",
            params,
        ).fetchall()
        return {
            "total": totals["total"],
            "count": totals["count"],
            "categories": [dict(row) for row in categories],
        }
    finally:
        conn.close()


def get_expense(expense_id, user_id):
    """Return one expense, but only if it belongs to this user. Otherwise None."""
    conn = get_db()
    try:
        return conn.execute(
            "SELECT id, user_id, amount, category, date, description FROM expenses "
            "WHERE id = ? AND user_id = ?",
            (expense_id, user_id),
        ).fetchone()
    finally:
        conn.close()


def update_expense(expense_id, user_id, amount, category, date, description):
    """Update one of this user's expenses. Returns True if a row was changed."""
    conn = get_db()
    try:
        cursor = conn.execute(
            "UPDATE expenses SET amount = ?, category = ?, date = ?, description = ? "
            "WHERE id = ? AND user_id = ?",
            (amount, category, date, description or None, expense_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_expense(expense_id, user_id):
    """Delete one of this user's expenses. Returns True if a row was deleted."""
    conn = get_db()
    try:
        cursor = conn.execute(
            "DELETE FROM expenses WHERE id = ? AND user_id = ?",
            (expense_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_user_by_id(user_id):
    """Return the user row with this id, or None if there isn't one."""
    conn = get_db()
    try:
        return conn.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    finally:
        conn.close()


def _filter_params(user_id, date_from, date_to):
    """Values for the shared date-filter WHERE clause used below:

        WHERE user_id = ?
          AND (? IS NULL OR date >= ?)   -- skipped when date_from is None
          AND (? IS NULL OR date <= ?)   -- skipped when date_to is None

    Dates are 'YYYY-MM-DD' text, which compares correctly as plain strings.
    """
    date_from = date_from or None
    date_to = date_to or None
    return (user_id, date_from, date_from, date_to, date_to)


def get_expense_summary(user_id, date_from=None, date_to=None):
    """Return {"total": ..., "count": ...} for a user's expenses (optionally in a date range)."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT COALESCE(ROUND(SUM(amount), 2), 0) AS total, COUNT(*) AS count "
            "FROM expenses "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?)",
            _filter_params(user_id, date_from, date_to),
        ).fetchone()
        return {"total": row["total"], "count": row["count"]}
    finally:
        conn.close()


def get_category_totals(user_id, date_from=None, date_to=None):
    """Return one row per category with its total, biggest spending first."""
    conn = get_db()
    try:
        return conn.execute(
            "SELECT category, ROUND(SUM(amount), 2) AS total, COUNT(*) AS count "
            "FROM expenses "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
            "GROUP BY category "
            "ORDER BY total DESC, category ASC",
            _filter_params(user_id, date_from, date_to),
        ).fetchall()
    finally:
        conn.close()


def get_monthly_totals(user_id, date_from=None, date_to=None):
    """Spending per month, oldest first: [{"month": "2026-10", "total": 6273.5, "count": 7}, ...].
    Months with no spending are simply missing (the caller fills them with 0)."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT substr(date, 1, 7) AS month, ROUND(SUM(amount), 2) AS total, COUNT(*) AS count "
            "FROM expenses "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
            "GROUP BY month ORDER BY month",
            _filter_params(user_id, date_from, date_to),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_daily_totals(user_id, date_from=None, date_to=None):
    """Spending per day, oldest first: [{"date": "2026-11-05", "total": 370.0}, ...].
    Days with no spending are simply missing (the caller fills them with 0)."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT date, ROUND(SUM(amount), 2) AS total "
            "FROM expenses "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
            "GROUP BY date ORDER BY date",
            _filter_params(user_id, date_from, date_to),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_month_comparison(user_id, this_start, this_end, last_start, last_end):
    """This month's total, last month's total and the % change, all calculated by SQLite.

    Returns {"this_total": 1399.0, "last_total": 6273.5, "change_percent": -77.7}.
    change_percent is None when last month had no spending (a % change can't be computed).
    """
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT this_total, last_total, "
            "       CASE WHEN last_total > 0 "
            "            THEN ROUND((this_total - last_total) * 100.0 / last_total, 1) END AS change_percent "
            "FROM (SELECT "
            "  (SELECT COALESCE(ROUND(SUM(amount), 2), 0) FROM expenses "
            "   WHERE user_id = ? AND date >= ? AND date <= ?) AS this_total, "
            "  (SELECT COALESCE(ROUND(SUM(amount), 2), 0) FROM expenses "
            "   WHERE user_id = ? AND date >= ? AND date <= ?) AS last_total)",
            (user_id, this_start, this_end, user_id, last_start, last_end),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def compare_spending(user_id, first_from, first_to, second_from, second_to, category=None):
    """Totals of two date ranges and their difference, all calculated by SQLite.
    Returns {"first": 1900.0, "second": 1574.0, "difference": 326.0} (difference = first - second)."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT first, second, ROUND(first - second, 2) AS difference FROM (SELECT "
            "  (SELECT COALESCE(ROUND(SUM(amount), 2), 0) FROM expenses "
            "   WHERE user_id = ? AND date >= ? AND date <= ? AND (? IS NULL OR category = ?)) AS first, "
            "  (SELECT COALESCE(ROUND(SUM(amount), 2), 0) FROM expenses "
            "   WHERE user_id = ? AND date >= ? AND date <= ? AND (? IS NULL OR category = ?)) AS second)",
            (user_id, first_from, first_to, category, category,
             user_id, second_from, second_to, category, category),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


CATEGORY_SHARE_SQL = (
    "SELECT part, total, CASE WHEN total > 0 THEN ROUND(part * 100.0 / total, 1) END AS percent "
    "FROM (SELECT COALESCE(ROUND(SUM(CASE WHEN category = ? THEN amount END), 2), 0) AS part, "
    "             COALESCE(ROUND(SUM(amount), 2), 0) AS total "
    "      FROM expenses "
    "      WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?))"
)


def get_category_share(user_id, category, date_from=None, date_to=None):
    """One category's share of all spending in a date range, all calculated by SQLite.
    Returns {"part": 680.0, "total": 3474.0, "percent": 19.6}; percent is None when total is 0."""
    conn = get_db()
    try:
        row = conn.execute(CATEGORY_SHARE_SQL,
                           (category, *_filter_params(user_id, date_from, date_to))).fetchone()
        return dict(row)
    finally:
        conn.close()


WEEKDAY_SPENDING_SQL = (
    "SELECT CAST(strftime('%w', date) AS INTEGER) AS weekday, COUNT(*) AS count, "
    "       ROUND(SUM(amount), 2) AS total "
    "FROM expenses WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
    "AND (? IS NULL OR category = ?) "
    "GROUP BY weekday ORDER BY total DESC, weekday"
)


def get_spending_by_weekday(user_id, date_from=None, date_to=None, category=None):
    """Total spending per day of the week (SUM over ALL matching rows), calculated by SQLite.
    Returns [{"weekday": 6, "count": 3, "total": 2400.0}, ...] (0 = Sunday ... 6 = Saturday),
    biggest total first; only weekdays that have expenses."""
    conn = get_db()
    try:
        params = (*_filter_params(user_id, date_from, date_to), category or None, category or None)
        return [dict(r) for r in conn.execute(WEEKDAY_SPENDING_SQL, params).fetchall()]
    finally:
        conn.close()


def get_average_expense(user_id, date_from=None, date_to=None):
    """Average size and number of one user's expenses in a date range (SQLite does the maths).
    Returns {"average": 450.25, "count": 12}; average is 0 when there are no expenses."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT COALESCE(ROUND(AVG(amount), 2), 0) AS average, COUNT(*) AS count "
            "FROM expenses "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?)",
            _filter_params(user_id, date_from, date_to),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


# ---------- Fixed measurements (text-to-SQL templates) ----------
# The AI only PICKS one of these by name; it never writes their maths. Real model: "avg spend per
# expense" got SUM / COUNT(DISTINCT date) (per spending DAY, ₹579) instead of AVG(amount) (₹289.50).
# Each query: value, count, total. Params: [divisor,] user_id, from x2, to x2, category x2.
SPENDING_MEASURES = {
    # measure: (SQL, divisor: None / "days" / "months")
    "average_per_expense": (
        "SELECT ROUND(AVG(amount), 2) AS value, COUNT(*) AS count, COALESCE(ROUND(SUM(amount), 2), 0) AS total "
        "FROM expenses WHERE user_id = ? AND (? IS NULL OR date >= ?) "
        "AND (? IS NULL OR date <= ?) AND (? IS NULL OR category = ?)", None),
    "average_per_day": (
        "SELECT ROUND(COALESCE(SUM(amount), 0) * 1.0 / ?, 2) AS value, COUNT(*) AS count, "
        "COALESCE(ROUND(SUM(amount), 2), 0) AS total "
        "FROM expenses WHERE user_id = ? AND (? IS NULL OR date >= ?) "
        "AND (? IS NULL OR date <= ?) AND (? IS NULL OR category = ?)", "days"),
    "average_per_week": (
        "SELECT ROUND(COALESCE(SUM(amount), 0) * 7.0 / ?, 2) AS value, COUNT(*) AS count, "
        "COALESCE(ROUND(SUM(amount), 2), 0) AS total "
        "FROM expenses WHERE user_id = ? AND (? IS NULL OR date >= ?) "
        "AND (? IS NULL OR date <= ?) AND (? IS NULL OR category = ?)", "days"),
    "average_per_month": (
        "SELECT ROUND(COALESCE(SUM(amount), 0) * 1.0 / ?, 2) AS value, COUNT(*) AS count, "
        "COALESCE(ROUND(SUM(amount), 2), 0) AS total "
        "FROM expenses WHERE user_id = ? AND (? IS NULL OR date >= ?) "
        "AND (? IS NULL OR date <= ?) AND (? IS NULL OR category = ?)", "months"),
    "total": (
        "SELECT COALESCE(ROUND(SUM(amount), 2), 0) AS value, COUNT(*) AS count, "
        "COALESCE(ROUND(SUM(amount), 2), 0) AS total "
        "FROM expenses WHERE user_id = ? AND (? IS NULL OR date >= ?) "
        "AND (? IS NULL OR date <= ?) AND (? IS NULL OR category = ?)", None),
    "count": (
        "SELECT COUNT(*) AS value, COUNT(*) AS count, COALESCE(ROUND(SUM(amount), 2), 0) AS total "
        "FROM expenses WHERE user_id = ? AND (? IS NULL OR date >= ?) "
        "AND (? IS NULL OR date <= ?) AND (? IS NULL OR category = ?)", None),
    "min": (
        "SELECT MIN(amount) AS value, COUNT(*) AS count, COALESCE(ROUND(SUM(amount), 2), 0) AS total "
        "FROM expenses WHERE user_id = ? AND (? IS NULL OR date >= ?) "
        "AND (? IS NULL OR date <= ?) AND (? IS NULL OR category = ?)", None),
    "max": (
        "SELECT MAX(amount) AS value, COUNT(*) AS count, COALESCE(ROUND(SUM(amount), 2), 0) AS total "
        "FROM expenses WHERE user_id = ? AND (? IS NULL OR date >= ?) "
        "AND (? IS NULL OR date <= ?) AND (? IS NULL OR category = ?)", None),
}


def get_spending_measure(user_id, measure, today, date_from=None, date_to=None, category=None):
    """One fixed measurement of a user's expenses, calculated by SQLite (see SPENDING_MEASURES).

    Per day / week / month divide the total by the number of days / calendar months in the
    period, counted up to today for a period still running ("November" on 07 Nov = 7 days).
    All time starts at the user's first expense. Returns {"value", "count", "total", "days",
    "months", "first", "last", "sql"}; value is None when there are no expenses (or no days)."""
    sql, divisor = SPENDING_MEASURES[measure]
    today = today.isoformat() if hasattr(today, "isoformat") else str(today)
    conn = get_db()
    try:
        first = date_from or conn.execute("SELECT MIN(date) FROM expenses WHERE user_id = ?",
                                          (user_id,)).fetchone()[0]
        last = date_to or today
        if first and first <= today:
            last = min(last, today)
        days = months = 0
        if first and last >= first:
            a, b = datetime.strptime(first, "%Y-%m-%d"), datetime.strptime(last, "%Y-%m-%d")
            days = (b - a).days + 1
            months = (b.year - a.year) * 12 + b.month - a.month + 1
        params = (*_filter_params(user_id, date_from, date_to), category or None, category or None)
        if divisor:
            n = days if divisor == "days" else months
            if not n:
                row = {"value": None, "count": 0, "total": 0}
            else:
                row = dict(conn.execute(sql, (n, *params)).fetchone())
        else:
            row = dict(conn.execute(sql, params).fetchone())
        if not row["count"]:
            row["value"] = None
        return {**row, "days": days, "months": months, "first": first, "last": last, "sql": sql}
    finally:
        conn.close()


# ---------- Income ----------

def add_income(user_id, amount, source, date, description):
    """Save one income entry for a user and return its id."""
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO income (user_id, amount, source, date, description) VALUES (?, ?, ?, ?, ?)",
            (user_id, amount, source, date, description or None),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def add_incomes(user_id, incomes):
    """Save several income entries in one transaction (all or none).
    incomes - list of (amount, source, date, description) tuples."""
    conn = get_db()
    try:
        with conn:
            conn.executemany(
                "INSERT INTO income (user_id, amount, source, date, description) VALUES (?, ?, ?, ?, ?)",
                [(user_id, amount, source, date, description or None)
                 for amount, source, date, description in incomes],
            )
    finally:
        conn.close()


def get_income(income_id, user_id):
    """One income entry, but only if it belongs to this user. Otherwise None."""
    conn = get_db()
    try:
        return conn.execute(
            "SELECT id, user_id, amount, source, date, description FROM income WHERE id = ? AND user_id = ?",
            (income_id, user_id),
        ).fetchone()
    finally:
        conn.close()


def update_income(income_id, user_id, amount, source, date, description):
    """Update one of this user's income entries. Returns True if a row was changed."""
    conn = get_db()
    try:
        cursor = conn.execute(
            "UPDATE income SET amount = ?, source = ?, date = ?, description = ? WHERE id = ? AND user_id = ?",
            (amount, source, date, description or None, income_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_income(income_id, user_id):
    """Delete one of this user's income entries. Returns True if a row was deleted."""
    conn = get_db()
    try:
        cursor = conn.execute("DELETE FROM income WHERE id = ? AND user_id = ?", (income_id, user_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_income_for_user(user_id, date_from=None, date_to=None):
    """A user's income entries (optionally in a date range), newest first."""
    conn = get_db()
    try:
        return conn.execute(
            "SELECT id, amount, source, date, description FROM income "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
            "ORDER BY date DESC, id DESC",
            _filter_params(user_id, date_from, date_to),
        ).fetchall()
    finally:
        conn.close()


def get_money_summary(user_id, date_from=None, date_to=None):
    """Income, expenses, balance and savings rate for one user, all calculated by SQLite.

    {"income": 30000.0, "income_count": 1, "expenses": 12000.0, "expense_count": 9,
     "balance": 18000.0, "savings_rate": 60.0}
    savings_rate = balance / income x 100, or None when there's no income (can't divide by 0).
    """
    params = _filter_params(user_id, date_from, date_to) * 4    # the same filter for all 4 sub-queries
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT income, income_count, expenses, expense_count, "
            "       ROUND(income - expenses, 2) AS balance, "
            "       CASE WHEN income > 0 THEN ROUND((income - expenses) * 100.0 / income, 1) END AS savings_rate "
            "FROM (SELECT "
            "  (SELECT COALESCE(ROUND(SUM(amount), 2), 0) FROM income "
            "   WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?)) AS income, "
            "  (SELECT COUNT(*) FROM income "
            "   WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?)) AS income_count, "
            "  (SELECT COALESCE(ROUND(SUM(amount), 2), 0) FROM expenses "
            "   WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?)) AS expenses, "
            "  (SELECT COUNT(*) FROM expenses "
            "   WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?)) AS expense_count)",
            params,
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def get_monthly_income(user_id, date_from=None, date_to=None):
    """Income per month, oldest first: [{"month": "2026-10", "total": 30000.0}, ...]."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT substr(date, 1, 7) AS month, ROUND(SUM(amount), 2) AS total FROM income "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
            "GROUP BY month ORDER BY month",
            _filter_params(user_id, date_from, date_to),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# ---------- Budgets ----------

def set_budget(user_id, category, monthly_limit):
    """Create or change one of this user's budgets. category None = overall budget."""
    conn = get_db()
    try:
        with conn:  # one transaction: update if it exists, otherwise insert
            cursor = conn.execute(
                "UPDATE budgets SET monthly_limit = ? "
                "WHERE user_id = ? AND COALESCE(category, '') = COALESCE(?, '')",
                (monthly_limit, user_id, category),
            )
            if cursor.rowcount == 0:
                conn.execute(
                    "INSERT INTO budgets (user_id, category, monthly_limit) VALUES (?, ?, ?)",
                    (user_id, category, monthly_limit),
                )
    finally:
        conn.close()


def delete_budget(user_id, category):
    """Remove one of this user's budgets. Returns True if one was removed."""
    conn = get_db()
    try:
        cursor = conn.execute(
            "DELETE FROM budgets WHERE user_id = ? AND COALESCE(category, '') = COALESCE(?, '')",
            (user_id, category),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_budget(user_id, category):
    """This user's monthly limit for a category (None = overall), or None if not set."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT monthly_limit FROM budgets "
            "WHERE user_id = ? AND COALESCE(category, '') = COALESCE(?, '')",
            (user_id, category),
        ).fetchone()
        return row["monthly_limit"] if row else None
    finally:
        conn.close()


def get_budget_status(user_id, month_start, month_end):
    """Every budget of this user with what was spent against it between month_start and
    month_end. All maths is done by SQLite. Overall budget first, then categories A-Z.

    [{"category": None, "monthly_limit": 2000.0, "spent": 1500.0, "percent": 75.0,
      "remaining": 500.0}, ...]   (remaining is negative when the budget is exceeded)
    """
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT category, monthly_limit, spent, "
            "       ROUND(spent * 100.0 / monthly_limit, 1) AS percent, "
            "       ROUND(monthly_limit - spent, 2) AS remaining "
            "FROM ("
            "  SELECT b.category, b.monthly_limit, "
            "         (SELECT COALESCE(ROUND(SUM(e.amount), 2), 0) FROM expenses e "
            "          WHERE e.user_id = b.user_id AND e.date >= ? AND e.date <= ? "
            "            AND (b.category IS NULL OR e.category = b.category)) AS spent "
            "  FROM budgets b WHERE b.user_id = ?"
            ") "
            "ORDER BY category IS NOT NULL, category",
            (month_start, month_end, user_id),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_expenses_for_user(user_id, date_from=None, date_to=None):
    """Return a user's expenses (optionally in a date range), newest first."""
    conn = get_db()
    try:
        return conn.execute(
            "SELECT id, amount, category, date, description "
            "FROM expenses "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
            "ORDER BY date DESC, id DESC",
            _filter_params(user_id, date_from, date_to),
        ).fetchall()
    finally:
        conn.close()


# ---------- Forecast & anomaly detection (forecast.py / anomaly.py) ----------

def get_expenses_up_to(user_id, date_to=None, date_from=None):
    """One user's expenses, OLDEST first (date, then id), up to and including date_to.
    The ML code reads its history from here; backtests pass a past date_to so the model
    never sees what happened after its 'today'."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, amount, category, date, description FROM expenses "
            "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) "
            "ORDER BY date, id",
            _filter_params(user_id, date_from, date_to),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def dismiss_anomaly(user_id, expense_id):
    """'This is normal': stop flagging one of THIS user's expenses. Returns False if the
    expense doesn't exist or belongs to someone else (nothing is saved then)."""
    conn = get_db()
    try:
        with conn:
            cursor = conn.execute(
                "INSERT OR REPLACE INTO anomaly_dismissals (user_id, expense_id, amount) "
                "SELECT user_id, id, amount FROM expenses WHERE id = ? AND user_id = ?",
                (expense_id, user_id),
            )
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_anomaly_dismissals(user_id):
    """{expense_id: amount when dismissed} for this user."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT expense_id, amount FROM anomaly_dismissals WHERE user_id = ?", (user_id,)
        ).fetchall()
        return {r["expense_id"]: r["amount"] for r in rows}
    finally:
        conn.close()


def delete_user(user_id):
    """Delete a user and (ON DELETE CASCADE) all their expenses, income, budgets and
    dismissals. Only used by scripts/seed_ml_demo.py to re-create the demo user."""
    conn = get_db()
    try:
        with conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return cursor.rowcount > 0
    finally:
        conn.close()


# ---------- Read-only SQL written by the AI (Text-to-SQL) ----------
# The AI's SQL is never trusted. Safety comes from what the connection CAN do, not from reading
# the SQL text:
#   1. a separate READ-ONLY connection (mode=ro) with PRAGMA query_only=ON: no writes are possible;
#   2. per-user TEMP TABLES my_expenses / my_income, filled (parameterized, user id from the
#      session) with only the logged-in user's rows, without id or user_id columns;
#   3. an AUTHORIZER that SQLite asks about every table, column, function and statement while
#      planning the query: only SELECT, reads of those two temp tables (and of the query's own
#      CTEs/subqueries) and built-in functions are allowed. Everything else is denied: the real
#      tables, users, budgets, sqlite_master, PRAGMA, ATTACH, writes.
#   4. one statement only, at most SQL_MAX_ROWS rows, SQL_TIMEOUT_SECONDS seconds.
#
# Why temp TABLES and not temp VIEWS: reads made by a view reach the authorizer as "table
# expenses, via view my_expenses", but a CTE is reported the same way, and the AI picks CTE
# names. "WITH my_expenses AS (SELECT * FROM main.expenses) ..." would look like the real view
# and read every user's rows. With copies in temp tables, the real tables are never readable.

SQL_MAX_ROWS = 200
SQL_TIMEOUT_SECONDS = 2.0
SQL_MAX_LENGTH = 2000

SQL_TABLES = ("my_expenses", "my_income")
_DENIED_FUNCTIONS = {"load_extension"}

_USER_TABLE_SQL = [
    "CREATE TEMP TABLE my_expenses AS "
    "SELECT amount, category, date, description FROM main.expenses "
    "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?)",
    "CREATE TEMP TABLE my_income AS "
    "SELECT amount, source, date, description FROM main.income "
    "WHERE user_id = ? AND (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?)",
]


class SQLRejected(Exception):
    """The query was blocked or failed. The message is safe to show the AI (no data in it)."""


_SCHEMA_TABLES = {"sqlite_master", "sqlite_schema", "sqlite_temp_master", "sqlite_temp_schema",
                  "sqlite_sequence", "sqlite_stat1", "sqlite_stat4"}


def _make_sql_authorizer(real_tables):
    """The authorizer SQLite asks about every step of planning the AI's query. Allow-list only.

    real_tables - lower-case names of every real table/view/index in the database. When a query
    reads no column (COUNT(*), SELECT 1), SQLite reports the table WITHOUT its schema name, just
    like a CTE, so a read of one of these names is denied however it is reported.
    """
    def authorizer(action, arg1, arg2, db_name, source):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table = (arg1 or "").lower()
            if table in SQL_TABLES and db_name in ("temp", None):
                return sqlite3.SQLITE_OK      # this user's own rows
            if db_name is None and table not in real_tables:
                return sqlite3.SQLITE_OK      # the query's own CTE or subquery
            return sqlite3.SQLITE_DENY        # expenses, users, budgets, sqlite_master, ...
        if action == sqlite3.SQLITE_FUNCTION:
            if (arg2 or "").lower() in _DENIED_FUNCTIONS:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK          # built-in functions (strftime, ROUND, AVG, ...)
        if action == getattr(sqlite3, "SQLITE_RECURSIVE", 33):
            return sqlite3.SQLITE_OK          # WITH RECURSIVE (e.g. a list of dates)
        return sqlite3.SQLITE_DENY            # writes, PRAGMA, ATTACH, CREATE, transactions, ...
    return authorizer


def _statement_count_problem(sql):
    """None if `sql` is ONE statement (a trailing ";" is fine), else why not.
    Skips text inside quotes and comments, so a ';' there doesn't count."""
    i, n, end_seen = 0, len(sql), False
    while i < n:
        ch = sql[i]
        if ch in "'\"`[":
            close = "]" if ch == "[" else ch
            j = sql.find(close, i + 1)
            while j != -1 and close != "]" and sql[j + 1:j + 2] == close:   # '' escape
                j = sql.find(close, j + 2)
            if j == -1:
                return "unclosed quote"
            if end_seen:
                return "more than one statement"
            i = j + 1
            continue
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j == -1 else j + 1
            continue
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if ch == ";":
            end_seen = True
        elif not ch.isspace() and end_seen:
            return "more than one statement"
        i += 1
    return None


def run_user_sql(user_id, sql, date_from=None, date_to=None,
                 max_rows=SQL_MAX_ROWS, timeout_seconds=SQL_TIMEOUT_SECONDS):
    """Run ONE read-only SELECT written by the AI against this user's own data only.

    The query can only see two temp tables holding this user's rows (`user_id` from the session),
    limited to date_from..date_to (YYYY-MM-DD, resolved in code) when given:
        my_expenses(amount, category, date, description)
        my_income(amount, source, date, description)
    Returns {"columns": [...], "rows": [tuple, ...], "truncated": bool}.
    Raises SQLRejected with a short reason if the query is blocked, fails or takes too long.
    """
    sql = (sql or "").strip()
    if not sql:
        raise SQLRejected("empty query")
    if len(sql) > SQL_MAX_LENGTH:
        raise SQLRejected(f"query longer than {SQL_MAX_LENGTH} characters")
    problem = _statement_count_problem(sql)
    if problem:
        raise SQLRejected(problem)
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        raise SQLRejected("only SELECT queries are allowed")

    uri = pathlib.Path(DB_PATH).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    try:
        params = (int(user_id), date_from or None, date_from or None, date_to or None, date_to or None)
        for table_sql in _USER_TABLE_SQL:
            conn.execute(table_sql, params)   # TEMP tables work on a read-only connection
        conn.execute("PRAGMA query_only = ON")
        real_tables = {row[0].lower() for row in conn.execute(
            "SELECT name FROM main.sqlite_master UNION SELECT name FROM temp.sqlite_master")}
        real_tables = (real_tables | _SCHEMA_TABLES) - set(SQL_TABLES)
        conn.set_authorizer(_make_sql_authorizer(real_tables))   # from here on: allow-list only

        deadline = time.monotonic() + timeout_seconds
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as e:
            text = str(e)
            if "interrupted" in text:
                raise SQLRejected(f"took longer than {timeout_seconds:g} seconds") from None
            if "not authorized" in text or "prohibited" in text:
                raise SQLRejected("not allowed: only SELECT on my_expenses and my_income") from None
            raise SQLRejected(text) from None
        except (sqlite3.Warning, sqlite3.ProgrammingError) as e:   # e.g. "one statement at a time"
            raise SQLRejected(str(e)) from None
        if cursor.description is None:
            raise SQLRejected("the query returned no table")
        return {
            "columns": [d[0] for d in cursor.description],
            "rows": [tuple(r) for r in rows[:max_rows]],
            "truncated": len(rows) > max_rows,
        }
    finally:
        conn.close()


def seed_db():
    """Add a demo user with sample expenses, but only if no users exist yet."""
    conn = get_db()
    try:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if user_count > 0:
            return

        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            ("Demo User", "demo@spendly.com", generate_password_hash("demo123")),
        )
        demo_user_id = cursor.lastrowid

        sample_expenses = [
            (450.00, "Food", "2026-10-03", "Groceries for the week"),
            (120.00, "Transport", "2026-10-05", "Auto rickshaw to office"),
            (1899.00, "Bills", "2026-10-08", "Electricity bill"),
            (2499.00, "Shopping", "2026-10-12", "New running shoes"),
            (350.00, "Entertainment", "2026-10-18", "Movie tickets"),
            (680.00, "Health", "2026-10-22", "Pharmacy - vitamins"),
            (275.50, "Food", "2026-10-29", "Dinner with friends"),
            (999.00, "Bills", "2026-11-01", "Mobile and internet recharge"),
        ]
        conn.executemany(
            "INSERT INTO expenses (user_id, amount, category, date, description) "
            "VALUES (?, ?, ?, ?, ?)",
            [(demo_user_id, *expense) for expense in sample_expenses],
        )
        conn.commit()
    finally:
        conn.close()
