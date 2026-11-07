"""Spendly as an MCP server, so Claude (desktop app, Claude Code) can use it.

Runs over stdio: the MCP client starts this file and talks JSON-RPC on stdin/stdout.
So NOTHING else may ever be printed to stdout. All logs go to stderr.

The server acts as ONE user: SPENDLY_USER_EMAIL from the .env file next to this file
(or the environment). Tools never take a user id.

It reuses the website's rules: validate_expense_form / validate_income_form and
add_expenses_with_budget_check from app.py, and the SQL in database/db.py (all totals
are calculated by SQLite). There are deliberately NO edit or delete tools.

    python mcp_server.py   # normally started by the MCP client, not by hand
"""
import contextlib
import logging
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(PROJECT_DIR, ".env")


def load_env_file(path=ENV_FILE):
    """Read KEY=value lines from .env into os.environ (real environment variables win).
    A tiny reader so no extra package is needed. Must run before database.db is imported,
    because db.py reads SPENDLY_DB_PATH at import time."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


load_env_file()

# Logs (including every write) go to stderr: stdout belongs to the MCP protocol.
LOG = logging.getLogger("spendly.mcp")
if not LOG.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[MCP] %(asctime)s %(message)s", "%H:%M:%S"))
    LOG.addHandler(_handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False

# Importing app.py sets up the database. Any stray print during import goes to stderr, not stdout.
with contextlib.redirect_stdout(sys.stderr):
    sys.path.insert(0, PROJECT_DIR)
    import app as spendly
    import database.db as db

from mcp.server.mcpserver import MCPServer          # noqa: E402  (FastMCP was renamed in mcp 2.x)
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp.types import ToolAnnotations                 # noqa: E402

CATEGORIES = spendly.CATEGORIES
INCOME_SOURCES = spendly.INCOME_SOURCES
DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 100

INSTRUCTIONS = f"""Spendly is a personal expense tracker (Indian rupees, ₹). You act as ONE fixed user.
- Dates are always YYYY-MM-DD. Leave date_from / date_to empty for all time.
- Expense categories: {", ".join(CATEGORIES)}.
- Income sources: {", ".join(INCOME_SOURCES)}.
- Amounts are plain numbers in rupees, greater than 0 (e.g. 250 or 99.50), no ₹ sign.
- Every number you report must come from a tool result: never add up or estimate yourself.
- add_expense and add_income SAVE immediately. Only call them when the user clearly asked to
  record that exact amount. There are no edit or delete tools: mistakes must be fixed on the website."""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False,
                        open_world_hint=False)

mcp = MCPServer("Spendly", instructions=INSTRUCTIONS)


class UserNotConfigured(Exception):
    """SPENDLY_USER_EMAIL is missing or doesn't match a Spendly account."""


def current_user():
    """The ONE user this server acts as, from SPENDLY_USER_EMAIL. Never from the AI."""
    email = os.environ.get("SPENDLY_USER_EMAIL", "").strip()
    if not email:
        raise UserNotConfigured(
            f"SPENDLY_USER_EMAIL is not set. Add a line like SPENDLY_USER_EMAIL=you@example.com "
            f"to {ENV_FILE}")
    user = db.get_user_by_email(email)
    if user is None:
        raise UserNotConfigured(
            f"No Spendly account has the email {email!r} (SPENDLY_USER_EMAIL in {ENV_FILE}). "
            "Register it on the website first, or fix the email.")
    return user


def _user_id():
    try:
        return current_user()["id"]
    except UserNotConfigured as e:
        raise ToolError(str(e)) from None


def _check_range(date_from, date_to):
    """Optional YYYY-MM-DD dates ('' = no limit). Returns (date_from, date_to) as str or None."""
    date_from, date_to = (date_from or "").strip() or None, (date_to or "").strip() or None
    for name, value in (("date_from", date_from), ("date_to", date_to)):
        if value is not None and not spendly.is_valid_date(value):
            raise ToolError(f"{name} must be a real date in YYYY-MM-DD form, e.g. 2026-11-06 (got {value!r}).")
    if date_from and date_to and date_from > date_to:
        raise ToolError("date_from must be on or before date_to.")
    return date_from, date_to


def _check_category(category):
    category = (category or "").strip()
    if not category:
        return None
    match = next((c for c in CATEGORIES if c.lower() == category.lower()), None)
    if match is None:
        raise ToolError(f"Unknown category {category!r}. Use one of: {', '.join(CATEGORIES)}.")
    return match


def _choice(value, choices):
    """'food' -> 'Food' (the form itself needs the exact spelling)."""
    value = str(value or "").strip()
    return next((c for c in choices if c.lower() == value.lower()), value)


def _form_errors(errors):
    return "Nothing was saved: " + " ".join(errors.values())


def _range_text(date_from, date_to):
    if date_from and date_to:
        return f"{date_from} to {date_to}"
    if date_from:
        return f"from {date_from}"
    if date_to:
        return f"up to {date_to}"
    return "all time"


# ---------- Read-only tools ----------

@mcp.tool(annotations=READ_ONLY)
def get_spending_summary(date_from: str = "", date_to: str = "", category: str = "") -> dict:
    """Total spending (₹), number of expenses and a per-category breakdown, biggest first.
    Use this for "how much did I spend" questions.

    Args:
        date_from: First day to include, YYYY-MM-DD. Empty = no start limit.
        date_to: Last day to include, YYYY-MM-DD. Empty = no end limit (both empty = all time).
        category: Only this category; empty = all. One of: Food, Transport, Bills, Health,
            Entertainment, Shopping, Other.
    """
    user_id = _user_id()
    date_from, date_to = _check_range(date_from, date_to)
    category = _check_category(category)
    summary = db.get_spending_summary(user_id, date_from, date_to, category)
    return {"currency": "INR", "period": _range_text(date_from, date_to), "category": category,
            **summary}


@mcp.tool(annotations=READ_ONLY)
def list_expenses(date_from: str = "", date_to: str = "", category: str = "",
                  limit: int = DEFAULT_LIST_LIMIT) -> dict:
    """List expenses, newest first: amount (₹), category, date (YYYY-MM-DD) and description.

    Args:
        date_from: First day to include, YYYY-MM-DD. Empty = no start limit.
        date_to: Last day to include, YYYY-MM-DD. Empty = no end limit.
        category: Only this category; empty = all. One of: Food, Transport, Bills, Health,
            Entertainment, Shopping, Other.
        limit: How many to return, 1-100 (default 20). For totals use get_spending_summary instead.
    """
    user_id = _user_id()
    date_from, date_to = _check_range(date_from, date_to)
    category = _check_category(category)
    if not 1 <= int(limit) <= MAX_LIST_LIMIT:
        raise ToolError(f"limit must be between 1 and {MAX_LIST_LIMIT}.")
    rows = db.search_expenses(user_id, category=category, date_from=date_from, date_to=date_to,
                              limit=int(limit))
    return {"currency": "INR", "period": _range_text(date_from, date_to), "category": category,
            "count": len(rows),
            "expenses": [{"amount": r["amount"], "category": r["category"], "date": r["date"],
                          "description": r["description"] or ""} for r in rows]}


@mcp.tool(annotations=READ_ONLY)
def get_budget_status() -> dict:
    """This month's budgets: monthly limit, spent, remaining (negative = over budget) and percent
    used, all in ₹. category null = the overall monthly budget. Budgets are always for the current
    calendar month. Status: "ok" (< 70%), "warning" (70-99%), "over" (100% or more)."""
    user_id = _user_id()
    month_from, month_to = spendly.this_month_range()
    rows = spendly.budget_rows(user_id)
    return {"currency": "INR", "month": f"{month_from} to {month_to}",
            "budgets": [{"category": r["category"], "monthly_limit": r["monthly_limit"],
                         "spent": r["spent"], "remaining": r["remaining"], "percent": r["percent"],
                         "status": r["level"]} for r in rows],
            "note": None if rows else "No budgets set. Budgets can be set on the website."}


@mcp.tool(annotations=READ_ONLY)
def get_savings(date_from: str = "", date_to: str = "") -> dict:
    """Income, expenses, balance (income - expenses) and savings rate, all in ₹.
    savings_rate = balance / income x 100 (%), or null when there is no income in the period.

    Args:
        date_from: First day to include, YYYY-MM-DD. Empty = no start limit.
        date_to: Last day to include, YYYY-MM-DD. Empty = no end limit (both empty = all time).
    """
    user_id = _user_id()
    date_from, date_to = _check_range(date_from, date_to)
    return {"currency": "INR", "period": _range_text(date_from, date_to),
            **db.get_money_summary(user_id, date_from, date_to)}


# ---------- Write tools (save immediately; logged to stderr) ----------

@mcp.tool(annotations=WRITE)
def add_expense(amount: float, category: str, date: str, description: str = "") -> dict:
    """SAVE one expense (money spent) right away. Only call this when the user clearly asked to
    record this exact amount. It can't be edited or deleted here (only on the website).

    Args:
        amount: Rupees spent, a number greater than 0, e.g. 250 or 99.50 (no ₹ sign).
        category: One of: Food, Transport, Bills, Health, Entertainment, Shopping, Other.
        date: The day it was spent, YYYY-MM-DD, e.g. 2026-11-06.
        description: Optional short note, max 200 characters, e.g. "auto to office".
    """
    user_id = _user_id()
    form, errors, clean = spendly.validate_expense_form({
        "amount": str(amount), "category": _choice(category, CATEGORIES),
        "date": str(date), "description": str(description or "")})
    if errors:
        raise ToolError(_form_errors(errors))
    expense = (clean, form["category"], form["date"], form["description"])
    warnings = spendly.add_expenses_with_budget_check(user_id, [expense])
    LOG.info("WRITE add_expense user=%s amount=%.2f category=%s date=%s description=%r",
             user_id, clean, form["category"], form["date"], form["description"])
    return {"saved": True, "type": "expense", "amount": clean, "category": form["category"],
            "date": form["date"], "description": form["description"],
            "budget_warnings": [w for w in warnings if w]}


@mcp.tool(annotations=WRITE)
def add_income(amount: float, source: str, date: str, description: str = "") -> dict:
    """SAVE one income entry (money coming IN) right away. Never use this for spending. Only
    call this when the user clearly asked to record this exact amount.

    Args:
        amount: Rupees received, a number greater than 0, e.g. 30000 (no ₹ sign).
        source: One of: Salary, Freelance, Business, Gift, Refund, Other.
            Money from family/friends -> Gift; money returned -> Refund.
        date: The day it came in, YYYY-MM-DD, e.g. 2026-11-06.
        description: Optional short note, max 200 characters, e.g. "from mom".
    """
    user_id = _user_id()
    form, errors, clean = spendly.validate_income_form({
        "amount": str(amount), "source": _choice(source, INCOME_SOURCES),
        "date": str(date), "description": str(description or "")})
    if errors:
        raise ToolError(_form_errors(errors))
    db.add_income(user_id, clean, form["source"], form["date"], form["description"])
    LOG.info("WRITE add_income user=%s amount=%.2f source=%s date=%s description=%r",
             user_id, clean, form["source"], form["date"], form["description"])
    return {"saved": True, "type": "income", "amount": clean, "source": form["source"],
            "date": form["date"], "description": form["description"]}


def main():
    try:
        user = current_user()
    except UserNotConfigured as e:
        LOG.error("Can't start: %s", e)
        sys.exit(1)
    LOG.info("Spendly MCP server (stdio) acting as user %s (%s), database %s",
             user["id"], user["email"], db.DB_PATH)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
