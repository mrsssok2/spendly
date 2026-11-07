"""The Spendly MCP server (mcp_server.py). Tools are called through a real in-process MCP
client, on the fresh test database from conftest.py (never the real spendly.db).
The server acts as SPENDLY_USER_EMAIL, which each test sets with monkeypatch."""
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import date, timedelta

import pytest
from mcp import Client

import app as spendly
import database.db as db
import mcp_server

TODAY = date.today()
YESTERDAY = (TODAY - timedelta(days=1)).isoformat()


def call(name, **arguments):
    """Call one tool through the MCP protocol. Returns (is_error, structured result or error text)."""
    async def run():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool(name, arguments)
    result = asyncio.run(run())
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    if result.is_error:
        return True, text
    return False, json.loads(text)          # what Claude actually reads: the result as JSON text


def list_tool_names():
    async def run():
        async with Client(mcp_server.mcp) as client:
            return await client.list_tools()
    return {t.name: t for t in asyncio.run(run()).tools}


@pytest.fixture
def asha(make_user, monkeypatch):
    """The server's user. Ravi (other_user) must never be visible through it."""
    user_id = make_user("asha@example.com", name="Asha")
    monkeypatch.setenv("SPENDLY_USER_EMAIL", "asha@example.com")
    return user_id


@pytest.fixture
def write_log():
    """Capture the [MCP] logger (it doesn't propagate, so attach a handler directly)."""
    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Keep()
    mcp_server.LOG.addHandler(handler)
    yield records
    mcp_server.LOG.removeHandler(handler)


# ---------- The tool list ----------

def test_exactly_the_six_tools_and_no_edit_or_delete():
    tools = list_tool_names()
    assert set(tools) == {"get_spending_summary", "list_expenses", "add_expense", "add_income",
                          "get_budget_status", "get_savings"}
    assert not any("edit" in n or "update" in n or "delete" in n for n in tools)


def test_tool_descriptions_name_dates_categories_and_rupees():
    tools = list_tool_names()
    for name, tool in tools.items():
        assert tool.description, f"{name} has no description"
        assert "₹" in tool.description, name
    assert "YYYY-MM-DD" in tools["add_expense"].description
    assert "Entertainment" in tools["add_expense"].description
    assert "Freelance" in tools["add_income"].description
    assert tools["get_savings"].annotations.read_only_hint is True
    assert tools["add_expense"].annotations.read_only_hint is False
    assert "user_id" not in str(tools["add_expense"].input_schema)        # never a user id


# ---------- get_spending_summary ----------

def test_get_spending_summary(asha, other_user):
    db.add_expense(asha, 300, "Food", "2026-10-05", "lunch")
    db.add_expense(asha, 200, "Transport", "2026-10-20", "auto")
    db.add_expense(asha, 50, "Food", "2026-11-02", "chai")
    db.add_expense(other_user, 9999, "Food", "2026-10-05", "Ravi's")       # not Asha's
    error, result = call("get_spending_summary")
    assert not error and result["total"] == 550 and result["count"] == 3
    assert result["categories"][0] == {"category": "Food", "total": 350, "count": 2}

    _, october = call("get_spending_summary", date_from="2026-10-01", date_to="2026-10-31")
    assert october["total"] == 500 and october["period"] == "2026-10-01 to 2026-10-31"
    _, food = call("get_spending_summary", category="food")              # any case
    assert food["total"] == 350 and food["category"] == "Food"


@pytest.mark.parametrize("arguments, message", [
    ({"date_from": "05-11-2026"}, "YYYY-MM-DD"),
    ({"date_to": "2026-02-30"}, "YYYY-MM-DD"),
    ({"date_from": "2026-11-05", "date_to": "2026-11-01"}, "on or before"),
    ({"category": "Groceries"}, "Unknown category"),
])
def test_get_spending_summary_rejects_bad_filters(asha, arguments, message):
    error, text = call("get_spending_summary", **arguments)
    assert error and message in text


# ---------- list_expenses ----------

def test_list_expenses(asha, other_user):
    db.add_expense(asha, 300, "Food", "2026-10-05", "lunch")
    db.add_expense(asha, 200, "Transport", "2026-10-20", "auto")
    db.add_expense(asha, 50, "Food", "2026-11-02", None)
    db.add_expense(other_user, 9999, "Food", "2026-10-06", "Ravi's")
    _, result = call("list_expenses")
    assert [e["amount"] for e in result["expenses"]] == [50, 200, 300]        # newest first
    assert result["expenses"][0] == {"amount": 50, "category": "Food", "date": "2026-11-02",
                                     "description": ""}
    _, food = call("list_expenses", category="Food", date_to="2026-10-31")
    assert [e["description"] for e in food["expenses"]] == ["lunch"]
    _, one = call("list_expenses", limit=1)
    assert one["count"] == 1


@pytest.mark.parametrize("limit", [0, 101])
def test_list_expenses_limit_is_checked(asha, limit):
    error, text = call("list_expenses", limit=limit)
    assert error and "between 1 and 100" in text


# ---------- add_expense ----------

def test_add_expense_saves_with_website_rules_and_logs_to_stderr(asha, write_log):
    error, result = call("add_expense", amount=250, category="transport", date=YESTERDAY,
                         description="auto")
    assert not error and result["saved"] is True and result["category"] == "Transport"
    [row] = db.get_expenses_for_user(asha)
    assert (row["amount"], row["category"], row["date"], row["description"]) == (
        250, "Transport", YESTERDAY, "auto")
    assert any(m.startswith("WRITE add_expense") and "amount=250.00" in m for m in write_log)


@pytest.mark.parametrize("arguments, message", [
    ({"amount": 0}, "greater than 0"),
    ({"amount": -20}, "greater than 0"),
    ({"category": "Groceries"}, "category"),
    ({"date": "06/11/2026"}, "valid date"),
    ({"date": "2026-02-30"}, "valid date"),
    ({"description": "x" * 201}, "200 characters"),
])
def test_add_expense_invalid_saves_nothing(asha, write_log, arguments, message):
    good = {"amount": 250, "category": "Food", "date": YESTERDAY, "description": "lunch"}
    error, text = call("add_expense", **{**good, **arguments})
    assert error and "Nothing was saved" in text and message in text
    assert db.get_expenses_for_user(asha) == [] and write_log == []


def test_add_expense_returns_budget_warning(asha):
    db.set_budget(asha, "Food", 1000)
    db.add_expense(asha, 700, "Food", TODAY.isoformat(), "groceries")
    _, result = call("add_expense", amount=200, category="Food", date=TODAY.isoformat())
    assert result["budget_warnings"] == ["⚠️ You've now used 90% of your ₹1,000 Food budget"]


# ---------- add_income ----------

def test_add_income_saves_and_is_not_an_expense(asha, write_log):
    error, result = call("add_income", amount=2000, source="gift", date=TODAY.isoformat(),
                         description="from mom")
    assert not error and result["source"] == "Gift"
    [row] = db.get_income_for_user(asha)
    assert (row["amount"], row["source"], row["description"]) == (2000, "Gift", "from mom")
    assert db.get_expenses_for_user(asha) == []
    assert any(m.startswith("WRITE add_income") for m in write_log)


@pytest.mark.parametrize("arguments, message", [
    ({"amount": 0}, "greater than 0"),
    ({"source": "Lottery"}, "source"),
    ({"date": "yesterday"}, "valid date"),
])
def test_add_income_invalid_saves_nothing(asha, arguments, message):
    good = {"amount": 5000, "source": "Freelance", "date": TODAY.isoformat()}
    error, text = call("add_income", **{**good, **arguments})
    assert error and message in text
    assert db.get_income_for_user(asha) == []


# ---------- get_budget_status ----------

def test_get_budget_status(asha):
    _, empty = call("get_budget_status")
    assert empty["budgets"] == [] and "No budgets" in empty["note"]

    db.set_budget(asha, None, 2000)
    db.set_budget(asha, "Food", 500)
    db.add_expense(asha, 600, "Food", TODAY.isoformat(), "dinner")
    db.add_expense(asha, 900, "Food", (spendly.month_start(TODAY, 0) - timedelta(days=1)).isoformat(),
                   "last month: not counted")
    _, result = call("get_budget_status")
    overall, food = result["budgets"]
    assert overall == {"category": None, "monthly_limit": 2000, "spent": 600, "remaining": 1400,
                       "percent": 30, "status": "ok"}
    assert food["category"] == "Food" and food["remaining"] == -100 and food["status"] == "over"


# ---------- get_savings ----------

def test_get_savings(asha):
    _, nothing = call("get_savings")
    assert nothing["savings_rate"] is None                                  # no income: no rate
    db.add_income(asha, 30000, "Salary", "2026-11-01", "salary")
    db.add_expense(asha, 12000, "Bills", "2026-11-03", "rent")
    db.add_expense(asha, 5000, "Bills", "2026-10-03", "rent (October)")
    _, result = call("get_savings", date_from="2026-11-01", date_to="2026-11-30")
    assert (result["income"], result["expenses"], result["balance"], result["savings_rate"]) == (
        30000, 12000, 18000, 60.0)


# ---------- The one user, from SPENDLY_USER_EMAIL ----------

def test_missing_email_is_a_clear_error(make_user, monkeypatch):
    monkeypatch.delenv("SPENDLY_USER_EMAIL", raising=False)
    error, text = call("get_savings")
    assert error and "SPENDLY_USER_EMAIL is not set" in text


def test_unknown_email_is_a_clear_error_and_nothing_is_saved(make_user, monkeypatch):
    other = make_user("ravi@example.com", name="Ravi")
    monkeypatch.setenv("SPENDLY_USER_EMAIL", "nobody@example.com")
    error, text = call("add_expense", amount=100, category="Food", date=YESTERDAY)
    assert error and "No Spendly account has the email 'nobody@example.com'" in text
    assert db.get_expenses_for_user(other) == []


def test_env_file_is_read_but_real_environment_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nSPENDLY_TEST_A="from file"\nSPENDLY_TEST_B=from file\n', encoding="utf-8")
    monkeypatch.delenv("SPENDLY_TEST_A", raising=False)
    monkeypatch.setenv("SPENDLY_TEST_B", "from environment")
    mcp_server.load_env_file(str(env))
    assert os.environ["SPENDLY_TEST_A"] == "from file"
    assert os.environ["SPENDLY_TEST_B"] == "from environment"
    monkeypatch.delenv("SPENDLY_TEST_A")


def test_server_refuses_to_start_with_unknown_email_and_keeps_stdout_clean(fresh_database, tmp_path):
    """Run the real script: a clear error on stderr, exit code 1, and NOTHING on stdout."""
    env = {**os.environ, "SPENDLY_DB_PATH": fresh_database,
           "SPENDLY_USER_EMAIL": "nobody@example.com"}
    result = subprocess.run([sys.executable, mcp_server.__file__], env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=120, stdin=subprocess.DEVNULL)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "No Spendly account has the email 'nobody@example.com'" in result.stderr
