"""Unusual-expense detection for Spendly (classic statistics, no LLM).

How an expense is judged
------------------------
Each expense is compared with the user's EARLIER expenses ("peers"), never with later ones, so a
flag means "this was unusual when it happened":

  1. Peers = the last WINDOW expenses in the same category. If at least MIN_WORD_PEERS (2) of them
     share a description word with this one ("groceries", "dinner", "rent"), only those are
     used. Categories are mixed bags (Food = ₹30 chai AND ₹1,000 weekly groceries), and without
     this every grocery run would look unusual next to the chai.
  2. typical = MEDIAN of the peers' amounts; spread = MAD (median absolute deviation).
  3. Flagged only when ALL of these hold:
       - the category has at least MIN_PAST earlier expenses (otherwise: never flagged),
       - the user has at least MIN_HISTORY_DAYS of history before it,
       - robust z = (amount - typical) / (1.4826 x MAD) >= Z_LIMIT   ("far outside the usual spread"),
       - amount >= MIN_RATIO x typical and amount - typical >= MIN_EXCESS   ("big enough to matter").
     Only HIGH amounts are flagged: a cheap lunch is not a problem worth a warning.
  4. Expenses the user marked "This is normal" are never flagged again (unless edited later).

Why median + MAD, not mean + standard deviation
-----------------------------------------------
The mean and the standard deviation are pulled towards the very outliers we want to catch: one
₹32,000 phone makes the "average Shopping expense" and its "normal range" so big that the next
spike hides inside it (masking). The standard deviation even SQUARES distances, so one spike
dominates it. The median and the MAD hardly move however extreme a few values are (up to half the
data can be spikes), and they don't assume a symmetric bell curve, which money never is.
1.4826 x MAD estimates the standard deviation of normal data, so Z_LIMIT = 3.5 is the usual
"modified z-score" cut-off (Iglewicz & Hoaglin).

`python evals/ml_eval.py` compares this with the IQR rule, mean/std and scikit-learn's
IsolationForest on the demo user's deliberate spikes (precision / recall).
"""
import re
import statistics
from collections import defaultdict
from datetime import date

MIN_PAST = 8             # a category needs this many earlier expenses before anything is judged
MIN_HISTORY_DAYS = 28    # ... and the user this many days of history
WINDOW = 60              # peers: the last 60 expenses in the category (habits change)
MIN_WORD_PEERS = 2       # peers sharing a description word are used when there are this many
                         # (2: rent's 3rd month must be judged against rent, not all Bills)
Z_LIMIT = 3.5            # robust z-score cut-off
MIN_RATIO = 3.0          # at least 3x the typical amount ...
MIN_EXCESS = 500.0       # ... and at least ₹500 more than it
MAD_TO_SD = 1.4826       # MAD x 1.4826 = standard deviation for normally distributed data

METHODS = ("mad", "iqr", "meanstd")
IQR_FENCE = 3.0          # Tukey's "far out" fence: Q3 + 3 x IQR

# Words that don't say WHAT was bought (so they don't make two expenses peers).
STOP_WORDS = {"with", "from", "for", "the", "and", "bill", "bills", "payment", "paid", "order",
              "expense", "purchase", "bought", "online", "monthly", "new", "shop", "store"}


def format_inr(amount):
    """3280 -> '₹3,280', 2499.5 -> '₹2,499.50' (the same style as the rest of the app)."""
    text = f"{float(amount):,.2f}"
    return "₹" + (text[:-3] if text.endswith(".00") else text)


def words(description):
    """Meaningful words of a description: 'Dinner with friends' -> {'dinner', 'friends'}."""
    return {w for w in re.findall(r"[a-z]+", (description or "").lower())
            if len(w) >= 3 and w not in STOP_WORDS}


def median_mad(values):
    m = statistics.median(values)
    return m, statistics.median(abs(v - m) for v in values)


def _quartiles(values):
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[0], q[2]


def _peers(expense, past):
    """(peer amounts, label): same-word peers when there are enough, else the whole category."""
    mine = words(expense["description"])
    if mine:
        shared = [p for p in past if mine & p["_words"]]
        if len(shared) >= MIN_WORD_PEERS:
            counts = defaultdict(int)
            for p in shared:
                for w in mine & p["_words"]:
                    counts[w] += 1
            word = max(sorted(counts), key=lambda w: counts[w])
            return [p["amount"] for p in shared], word
    return [p["amount"] for p in past], None


def _score(amount, peers, method):
    """(typical, score, is_far): how far `amount` is above its peers by the chosen statistic."""
    if method == "mad":
        typical, mad = median_mad(peers)
        # MAD can be 0 (rent: the same ₹15,000 every month); 10% of typical is the smallest spread.
        scale = max(MAD_TO_SD * mad, 0.1 * typical, 1.0)
        z = (amount - typical) / scale
        return typical, z, z >= Z_LIMIT
    if method == "iqr":
        typical = statistics.median(peers)
        q1, q3 = _quartiles(peers)
        iqr = max(q3 - q1, 0.1 * typical, 1.0)
        score = (amount - q3) / iqr
        return typical, score, score >= IQR_FENCE
    if method == "meanstd":
        typical = statistics.fmean(peers)
        sd = max(statistics.pstdev(peers), 1.0)
        z = (amount - typical) / sd
        return typical, z, z >= Z_LIMIT
    raise ValueError(f"unknown method {method!r}")


def ratio_text(ratio):
    return f"{ratio:.0f}" if ratio >= 3 else f"{ratio:.1f}"


def explanation(amount, typical, category, word):
    """'₹3,280 is about 6× your typical Food expense (₹520)'."""
    ratio = amount / typical if typical else 0
    what = f"“{word}” expense in {category}" if word else f"{category} expense"
    return f"{format_inr(amount)} is about {ratio_text(ratio)}× your typical {what} ({format_inr(typical)})"


def find_anomalies(expenses, dismissed=None, method="mad"):
    """Judge every expense against its earlier peers.

    expenses  - dicts with id, amount, category, date ('YYYY-MM-DD'), description, OLDEST FIRST.
    dismissed - {expense_id: amount}: "This is normal" marks (ignored if the amount changed since).
    method    - "mad" (the app's method), or "iqr" / "meanstd" for the evaluation's comparison.
    Returns {expense_id: {"amount", "typical", "ratio", "score", "category", "word", "peers",
                          "explanation"}} for the flagged expenses only.
    """
    dismissed = dismissed or {}
    if not expenses:
        return {}
    first_day = date.fromisoformat(min(e["date"] for e in expenses))
    history = defaultdict(list)
    flagged = {}
    for e in sorted(expenses, key=lambda x: (x["date"], x["id"])):
        e = {**e, "amount": float(e["amount"]), "_words": words(e.get("description"))}
        past = history[e["category"]]
        old_enough = (date.fromisoformat(e["date"]) - first_day).days >= MIN_HISTORY_DAYS
        if len(past) >= MIN_PAST and old_enough:
            peers, word = _peers(e, past[-WINDOW:])
            typical, score, far = _score(e["amount"], peers, method)
            big = (typical > 0 and e["amount"] >= MIN_RATIO * typical
                   and e["amount"] - typical >= MIN_EXCESS)
            normal = e["id"] in dismissed and round(dismissed[e["id"]], 2) == round(e["amount"], 2)
            if far and big and not normal:
                flagged[e["id"]] = {
                    "amount": e["amount"], "typical": round(typical, 2),
                    "ratio": round(e["amount"] / typical, 1), "score": round(score, 1),
                    "category": e["category"], "word": word, "peers": len(peers),
                    "date": e["date"], "description": e.get("description") or "",
                    "explanation": explanation(e["amount"], typical, e["category"], word),
                }
        past.append(e)
    return flagged


# ---------- Reading the database (all SQL is in database/db.py) ----------

def get_anomalies(user_id, date_from=None, date_to=None, category=None):
    """This user's flagged expenses, newest first, optionally only within a date range/category.
    The baselines always use the user's WHOLE history, whatever range is shown."""
    from database.db import get_anomaly_dismissals, get_expenses_up_to
    flagged = find_anomalies(get_expenses_up_to(user_id), get_anomaly_dismissals(user_id))
    items = [{"id": expense_id, **info} for expense_id, info in flagged.items()
             if (not date_from or info["date"] >= date_from) and (not date_to or info["date"] <= date_to)
             and (not category or info["category"] == category)]
    return sorted(items, key=lambda a: (a["date"], a["id"]), reverse=True)


def anomalies_by_id(user_id):
    """{expense_id: flag} for the Transactions table's ⚠️ badges."""
    return {a["id"]: a for a in get_anomalies(user_id)}
