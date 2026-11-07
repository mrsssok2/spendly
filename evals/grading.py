"""Grading: compare what the app did with the ground truth (all in code), plus the optional
LLM-as-judge for free text (reported separately, never part of the main score)."""
import json
import re

MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
DATES = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                                  # 2026-11-05
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\.?(?:,?\s+\d{{4}})?", re.I),   # 05 Nov 2026
    re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b", re.I),   # November 5
    re.compile(rf"\b{MONTHS}\s+\d{{4}}\b", re.I),                           # November 2026
]
NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\w.,])\d+(?:\.\d+)?")
MONEY = re.compile(r"(?:₹|\brs\.?\s*|\binr\s*)\s*(\d[\d,]*(?:\.\d+)?)", re.I)
NO_DATA = re.compile(
    r"\bno\b[^.\n]*\b(expenses?|spending|records?|transactions?|data|rows?|matching)|"
    r"haven.?t (spent|recorded)|didn.?t spend|nothing|not (spent|found)|\b0 expenses\b|"
    r"₹0(?:\.00)?\b|kuch nahi|koi kharcha nahi", re.I)
REFUSAL = re.compile(
    r"can.?t|cannot|only help|not able|unable|not allowed|couldn.?t answer|won.?t|"
    r"don.?t have access|only (?:answer|work)", re.I)


def _strip_dates(text):
    for pattern in DATES:
        text = pattern.sub(" ", text)
    return text


def numbers_in(text):
    """Every number in a reply (dates removed): '₹1,23,456.50' -> 123456.5."""
    return [float(n.replace(",", "")) for n in NUMBER.findall(_strip_dates(text or ""))]


def money_in(text):
    return [float(n.replace(",", "")) for n in MONEY.findall(_strip_dates(text or ""))]


def close(found, expected):
    """Tiny rounding differences are fine: 289.5 vs 289.50, or 289.5 written as ₹290 / ₹289."""
    if abs(found - expected) <= 0.011:
        return True
    return found == int(found) and abs(found - expected) < 1


def has_number(text, expected):
    if expected == 0 and NO_DATA.search(text or ""):
        return True                      # "You have no expenses today" answers a ₹0 total
    return any(close(f, expected) for f in numbers_in(text))


def grade_answer(reply, expect, truth, question):
    """Checks for a question's reply. Returns (passed, expected_text, problems)."""
    problems, expected = [], []
    reply = reply or ""
    for spec in expect.get("numbers", []):
        value = truth.value(spec)
        if value is None:
            expected.append(f"{spec['agg']}: none (no data)")
            if not NO_DATA.search(reply):
                problems.append(f"expected 'no data' for {spec}")
            continue
        expected.append(f"{spec['agg']}={value:g}")
        if not has_number(reply, value):
            problems.append(f"missing {spec['agg']} {value:g}")
    if "names" in expect:
        names = truth.top_names(expect["names"])
        expected.append("names=" + "/".join(names))
        missing = [n for n in names if not re.search(rf"\b{re.escape(n)}\b", reply, re.I)]
        if missing:
            problems.append(f"missing name(s) {missing}")
    if "rows" in expect:
        rows = truth.rows(expect["rows"])
        amounts = sorted({float(r[0]) for r in rows})
        expected.append("rows=" + ", ".join(f"{r[3] or r[1]} {r[0]:g}" for r in rows))
        if not rows:
            if not NO_DATA.search(reply):
                problems.append("expected an empty list")
        else:
            missing = [a for a in amounts if not has_number(reply, a)]
            allowed = amounts + numbers_in(question)
            extra = [m for m in money_in(reply) if not any(close(m, a) for a in allowed)]
            if missing:
                problems.append(f"missing row amount(s) {missing}")
            if extra:
                problems.append(f"amount(s) not in the list {extra}")
    if "empty" in expect:
        expected.append("no matching expenses")
        if truth.value({**expect["empty"], "agg": "count"}) != 0:
            problems.append("eval bug: the 'empty' filter has rows")
        if not NO_DATA.search(reply):
            problems.append("does not say there is nothing")
    if "text" in expect:
        expected.append(f"text: {expect['text']!r}")
        if not re.search(expect["text"], reply, re.I):
            problems.append(f"missing text {expect['text']!r}")
        if expect.get("no_money") and money_in(reply):
            problems.append(f"a guessed amount {money_in(reply)}")
    if expect.get("no_preview") is not None:
        pass   # checked by grade_previews
    return not problems, "; ".join(expected), problems


def _same(a, b):
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        try:
            return abs(float(a) - float(b)) < 0.005
        except (TypeError, ValueError):
            return False
    return str(a or "").strip().lower() == str(b or "").strip().lower()


def flatten_pending(pending):
    """The session's pending actions as a flat list of comparable previews."""
    out = []
    for action in pending.values():
        kind = action["type"]
        if kind == "add":
            out += [{"type": "add", **e} for e in action["expenses"]]
        elif kind == "add_income":
            out += [{"type": "income", **i} for i in action["incomes"]]
        elif kind == "edit":
            out.append({"type": "edit", "before": action["before"], "after": action["after"]})
        elif kind == "delete":
            out.append({"type": "delete", "before": action["before"]})
        elif kind == "budget":
            out.append({"type": "budget", "category": action["category"], "amount": action["after"]})
        else:
            out.append({"type": kind})
    return out


def _preview_matches(got, want):
    if got["type"] != want["type"]:
        return False
    for key, value in want.items():
        if key == "type":
            continue
        if key in ("before", "after"):
            if not all(_same(got[key].get(k), v) for k, v in value.items()):
                return False
        elif not _same(got.get(key), value):
            return False
    return True


def grade_previews(previews, wanted):
    """Every wanted preview must be there with exactly these fields, and nothing else."""
    problems, left = [], list(previews)
    for want in wanted:
        match = next((p for p in left if _preview_matches(p, want)), None)
        if match is None:
            problems.append(f"no preview matching {want}")
        else:
            left.remove(match)
    if left:
        problems.append(f"unexpected preview(s) {left}")
    return not problems, problems


def grade_receipt(result, want):
    problems = []
    if not _same(result.get("amount"), want["amount"]):
        problems.append(f"amount {result.get('amount')!r} != {want['amount']}")
    if not _same(result.get("category"), want["category"]):
        problems.append(f"category {result.get('category')!r} != {want['category']}")
    return not problems, problems


# ---------- Optional LLM-as-judge (free text only; reported separately) ----------

JUDGE_PROMPT = """You grade one reply from a personal-expense app. Check each item of the checklist
against the reply. Be strict: answer true only if the reply clearly meets the item.
Reply with JSON only: {"answers": [true or false for each item, in order], "note": "one short reason"}"""


def judge(llm, text, checklist):
    """Returns (passed, answers, note). The judge never changes the main score."""
    items = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(checklist))
    raw = llm.invoke([("system", JUDGE_PROMPT),
                      ("human", f"CHECKLIST:\n{items}\n\nREPLY:\n<<<\n{text}\n>>>")]).content
    try:
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        answers = [bool(a) for a in data.get("answers", [])][:len(checklist)]
        note = str(data.get("note", ""))[:200]
    except (ValueError, AttributeError):
        return False, [], f"judge reply not JSON: {raw[:120]!r}"
    if len(answers) != len(checklist):
        return False, answers, f"judge answered {len(answers)} of {len(checklist)} items"
    return all(answers), answers, note


# ---------- Insights card (ai_insights.py) ----------

INCOME_WORDS = re.compile(r"\b(?:savings?|saved|income|earn\w*|salary)\b|%\s+of\s+income", re.I)
CLAUSES = re.compile(r"(?<=[.!?])\s+|\s+[—–]\s+|;\s*|,\s+|\n")
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
FORECAST_WORDS = re.compile(r"on track to (?:exceed|go over|cross)|likely (?:to )?(?:go|be) over|forecast", re.I)


def grade_insights(result, expect, t, budget_forecasts=()):
    """Checks for the insights card. t = GroundTruth.insights_numbers(); budget_forecasts =
    [(category or None, limit)] the forecast says will likely go over. Returns a list of problems."""
    problems = []
    if result.get("status") != "ok":
        return [f"status {result.get('status')!r}: {result.get('message')}"]
    text = " ".join([result.get("summary") or ""] + list(result.get("tips") or []))
    numbers = {n["label"]: n for n in result.get("key_numbers") or []}

    if expect.get("no_income_words"):
        # No income this month: the card must not call anything savings or income (the bug:
        # "your savings so far are ₹22,385 (60.5% of income)" = spending + budget used).
        m = INCOME_WORDS.search(text)
        if m:
            problems.append(f"mentions {m.group(0)!r} but the user has no income")

    if expect.get("same_days_change"):
        if "Change" in numbers:
            problems.append("still shows the old 'Change' (month so far vs ALL of last month)")
        got = numbers.get("vs same days last month")
        want = t["same_days_pct"]
        if want is None:
            if got:
                problems.append(f"shows a same-days change {got['value']!r} but last month's same days are empty")
        elif not got:
            problems.append(f"no 'vs same days last month' number (expected {want}%)")
        else:
            value = [float(x) for x in PERCENT.findall(got["value"])]
            sign_ok = (want > 0) == ("▲" in got["value"]) if want else True
            if not value or not close(value[0], abs(want)) or not sign_ok:
                problems.append(f"same-days change {got['value']!r}, expected {want}%")
            if got.get("note") != t["same_days_note"]:
                problems.append(f"note {got.get('note')!r}, expected {t['same_days_note']!r}")
        old = t["old_pct"]
        if old is not None and want is not None and abs(abs(old) - abs(want)) >= 1:
            shown = " ".join([text] + [n["value"] for n in numbers.values()])
            if f"{abs(old)}%" in shown:
                problems.append(f"shows the misleading {abs(old)}% (month so far vs ALL of last month)")

    if expect.get("savings_labels") and t["income"]:
        # Every ₹ amount / % in a clause about savings or income must be the TRUE savings,
        # income or savings rate (not spending or a budget %).
        ok_money = [abs(t["saved"]), t["income"]]
        for clause in CLAUSES.split(text):
            if not INCOME_WORDS.search(clause):
                continue
            for amount in money_in(clause):
                if not any(close(amount, v) for v in ok_money):
                    problems.append(f"{clause.strip()!r}: ₹{amount:,.2f} is not the savings/income "
                                    f"(saved ₹{t['saved']:,.2f}, income ₹{t['income']:,.2f})")
            for pct in (float(p) for p in PERCENT.findall(clause)):
                if t["rate"] is not None and not close(pct, abs(t["rate"])):
                    problems.append(f"{clause.strip()!r}: {pct}% is not the savings rate ({t['rate']}%)")

    if expect.get("forecast_tip") and budget_forecasts:
        tips = " ".join(result.get("tips") or [])
        named = [c or "monthly budget" for c, _ in budget_forecasts]
        if not (FORECAST_WORDS.search(tips)
                and any(re.search(rf"\b{re.escape(n)}\b", tips, re.I) for n in named)):
            problems.append(f"no tip uses the forecast for the budget(s) likely to go over: {named}")
    return problems
