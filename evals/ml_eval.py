"""Evaluation of the classic-ML features (forecast.py, anomaly.py) on the ML demo user.

    python evals/ml_eval.py          # prints the report, saves evals/results/<date>_ml.json

No Ollama needed and the real database is never opened: the demo user is generated
(scripts/seed_ml_demo.py, fixed seed, relative to today) into a TEMPORARY database.

FORECAST BACKTEST: for every complete past month, pretend it is day 10 (and day 20) of that month,
hide everything after it, forecast the month-end total, then compare with what the month really
ended at. Reported: MAE (average miss in ₹), MAPE (average miss as % of the real total) and range
coverage (how often the real total fell inside the shown low-high range; the target is ~80%).
Compared with two baselines: run-rate (spent so far / days x days in month) and scikit-learn
linear regression on the same daily totals. Cut-offs with too little history are skipped (the
app shows "Not enough data" there), never guessed.

ANOMALY DETECTION: the demo data has deliberate spikes (seed_ml_demo.SPIKES). Every flagged
expense is a true positive (a spike) or a false positive (a normal expense):
precision = TP / flagged, recall = TP / spikes. Compared: median+MAD (the app), IQR, mean+std
(same peers and rules, only the statistic differs) and scikit-learn IsolationForest.
"""
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EVALS_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))

CUTOFF_DAYS = (10, 20)
# Pass marks used by the "ml" eval group in run_evals.py.
MAX_MAPE = 10.0            # average % error of the forecast
MIN_COVERAGE = 60.0        # % of real totals inside the range (the target is 80)
MIN_RECALL = 1.0           # every deliberate spike caught
MIN_PRECISION = 0.8


def _month_starts(first, today):
    m, out = date(first.year, first.month, 1), []
    while m < date(today.year, today.month, 1):
        out.append(m)
        m = (m + timedelta(days=32)).replace(day=1)
    return out


def _stats(rows, key):
    done = [r for r in rows if r.get(key) is not None]
    if not done:
        return {"n": 0, "mae": None, "mape": None}
    errors = [abs(r[key] - r["actual"]) for r in done]
    return {"n": len(done), "mae": round(sum(errors) / len(done), 2),
            "mape": round(100 * sum(e / r["actual"] for e, r in zip(errors, done)) / len(done), 2)}


def backtest(expenses, today, spike_dates=(), simulations=None):
    """One row per (past month, cut-off day). spike_after: a deliberate one-off spike happened
    AFTER the cut-off in that month (no forecast can know about it)."""
    import forecast
    first = date.fromisoformat(expenses[0]["date"])
    rows = []
    for m in _month_starts(first, today):
        end = forecast.month_end(m)
        actual = round(sum(e["amount"] for e in expenses if m.isoformat() <= e["date"] <= end.isoformat()), 2)
        for cut in CUTOFF_DAYS:
            as_of = m.replace(day=cut)
            known = [e for e in expenses if e["date"] <= as_of.isoformat()]   # never the future
            f = forecast.forecast_month(known, as_of, partial_today=False,
                                        simulations=simulations or forecast.SIMULATIONS)
            row = {"month": f"{m:%Y-%m}", "cutoff_day": cut, "actual": actual, "status": f["status"],
                   "spike_after": any(as_of.isoformat() < d <= end.isoformat() for d in spike_dates)}
            if f["status"] == "ok":
                row.update(forecast=f["expected_raw"], low=f["low_raw"], high=f["high_raw"],
                           inside=f["low_raw"] <= actual <= f["high_raw"],
                           run_rate=round(forecast.run_rate_forecast(known, as_of), 2))
                lr = forecast.linear_regression_forecast(known, as_of)
                row["linear_regression"] = round(lr, 2) if lr is not None else None
            else:
                row["detail"] = f.get("detail")
            rows.append(row)
    return rows


def forecast_report(rows):
    done = [r for r in rows if r["status"] == "ok"]
    report = {"forecasts": len(done), "skipped_not_enough_data": len(rows) - len(done), "by_model": {}}
    for key in ("forecast", "run_rate", "linear_regression"):
        report["by_model"][key] = {"all": _stats(done, key)}
        for cut in CUTOFF_DAYS:
            report["by_model"][key][f"day_{cut}"] = _stats([r for r in done if r["cutoff_day"] == cut], key)
    report["range_coverage_percent"] = round(100 * sum(r["inside"] for r in done) / len(done), 1) if done else None
    calm = [r for r in done if not r["spike_after"]]
    report["range_coverage_without_later_spike_percent"] = (
        round(100 * sum(r["inside"] for r in calm) / len(calm), 1) if calm else None)
    return report


def isolation_forest_flags(expenses):
    """IsolationForest per category on log(amount). Unlike the app's method it sees the WHOLE
    history at once (also the future), which flatters it. None if scikit-learn is missing."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return None
    by_cat = defaultdict(list)
    for e in expenses:
        by_cat[e["category"]].append(e)
    flagged = set()
    for rows in by_cat.values():
        if len(rows) < 8:
            continue
        model = IsolationForest(random_state=0, contamination="auto")
        X = [[math.log(e["amount"])] for e in rows]
        for e, label in zip(rows, model.fit_predict(X)):
            # IsolationForest also flags unusually SMALL amounts; only high ones count as spikes.
            if label == -1 and e["amount"] > sorted(r["amount"] for r in rows)[len(rows) // 2]:
                flagged.add(e["id"])
    return flagged


def anomaly_report(expenses, spike_ids):
    from anomaly import METHODS, find_anomalies
    by_id = {e["id"]: e for e in expenses}

    def score(flagged):
        tp = len(flagged & spike_ids)
        return {"flagged": len(flagged), "true_positives": tp, "false_positives": len(flagged - spike_ids),
                "precision": round(tp / len(flagged), 3) if flagged else None,
                "recall": round(tp / len(spike_ids), 3) if spike_ids else None,
                "missed": [f"{by_id[i]['date']} {by_id[i]['description']}" for i in sorted(spike_ids - flagged)],
                "false_alarms": [f"{by_id[i]['date']} {by_id[i]['description']} ₹{by_id[i]['amount']:g}"
                                 for i in sorted(flagged - spike_ids)][:10]}
    out = {m: score(set(find_anomalies(expenses, method=m))) for m in METHODS}
    iso = isolation_forest_flags(expenses)
    out["isolation_forest"] = score(iso) if iso is not None else {"status": "scikit-learn not installed"}
    return out


def evaluate(db, today=None, simulations=None):
    """Seed the demo user into the database `db` points at (must be a TEMPORARY one) and run both
    evaluations. Returns the report dict."""
    import seed_ml_demo
    today = today or date.today()
    user_id, spikes = seed_ml_demo.seed(db, today)
    expenses = db.get_expenses_up_to(user_id)
    spike_keys = {(s[2], s[3]) for s in spikes}
    spike_ids = {e["id"] for e in expenses if (e["date"], e["description"]) in spike_keys}
    rows = backtest(expenses, today, [s[2] for s in spikes], simulations)
    report = {
        "today": today.isoformat(), "expenses": len(expenses), "spikes": len(spike_ids),
        "forecast": forecast_report(rows), "backtest": rows,
        "anomalies": anomaly_report(expenses, spike_ids),
    }
    report["passed"], report["problems"] = check(report)
    return report


def check(report):
    """The pass marks for the 'ml' eval group."""
    problems = forecast_problems(report) + anomaly_problems(report)
    return not problems, problems


def forecast_problems(report):
    problems = []
    f = report["forecast"]
    mape = f["by_model"]["forecast"]["all"]["mape"]
    if not f["forecasts"]:
        problems.append("no backtest forecasts ran")
    else:
        if mape > MAX_MAPE:
            problems.append(f"forecast MAPE {mape}% > {MAX_MAPE}%")
        if f["range_coverage_percent"] < MIN_COVERAGE:
            problems.append(f"range coverage {f['range_coverage_percent']}% < {MIN_COVERAGE}%")
    return problems


def anomaly_problems(report):
    problems = []
    a = report["anomalies"]["mad"]
    if (a["recall"] or 0) < MIN_RECALL:
        problems.append(f"anomaly recall {a['recall']} < {MIN_RECALL} (missed {a['missed']})")
    if (a["precision"] or 0) < MIN_PRECISION:
        problems.append(f"anomaly precision {a['precision']} < {MIN_PRECISION} ({a['false_alarms']})")
    return problems


def _inr(v):
    return "—" if v is None else f"₹{v:,.0f}"


def print_report(r):
    print(f"ML evaluation · demo user · today {r['today']} · {r['expenses']} expenses, {r['spikes']} deliberate spikes\n")
    print("FORECAST BACKTEST (forecast from day 10 and day 20 of each past month)")
    print(f"  {'month':<8} {'day':>3} {'actual':>9} {'forecast':>9} {'range':>21} {'in?':>4} "
          f"{'run-rate':>9} {'lin.reg':>9}")
    for row in r["backtest"]:
        if row["status"] != "ok":
            print(f"  {row['month']:<8} {row['cutoff_day']:>3} {_inr(row['actual']):>9}  skipped: not enough data")
            continue
        rng = f"{_inr(row['low'])}–{_inr(row['high'])}"
        print(f"  {row['month']:<8} {row['cutoff_day']:>3} {_inr(row['actual']):>9} {_inr(row['forecast']):>9} "
              f"{rng:>21} {'yes' if row['inside'] else 'NO':>4} {_inr(row['run_rate']):>9} "
              f"{_inr(row['linear_regression']):>9}" + ("  (spike after the cut-off)" if row["spike_after"] else ""))
    f = r["forecast"]
    print(f"\n  {'model':<20} {'MAE all':>9} {'MAPE all':>9} {'MAPE d10':>9} {'MAPE d20':>9}")
    for name, label in (("forecast", "Spendly (simulation)"), ("run_rate", "run-rate baseline"),
                        ("linear_regression", "linear regression")):
        s = f["by_model"][name]
        if not s["all"]["n"]:
            print(f"  {label:<20} (not run)")
            continue
        print(f"  {label:<20} {_inr(s['all']['mae']):>9} {s['all']['mape']:>8.1f}% "
              f"{s['day_10']['mape'] or 0:>8.1f}% {s['day_20']['mape'] or 0:>8.1f}%")
    print(f"  range coverage: {f['range_coverage_percent']}% of real totals inside the 80% range "
          f"({f['forecasts']} forecasts, {f['skipped_not_enough_data']} skipped: not enough data)")
    print(f"  range coverage without a deliberate spike after the cut-off: "
          f"{f['range_coverage_without_later_spike_percent']}%")

    print("\nANOMALY DETECTION (deliberate spikes)")
    print(f"  {'method':<18} {'flagged':>7} {'TP':>3} {'FP':>3} {'precision':>9} {'recall':>7}")
    for name, s in r["anomalies"].items():
        if "status" in s:
            print(f"  {name:<18} {s['status']}")
            continue
        p = "—" if s["precision"] is None else f"{s['precision']:.2f}"
        print(f"  {name:<18} {s['flagged']:>7} {s['true_positives']:>3} {s['false_positives']:>3} "
              f"{p:>9} {s['recall']:>7.2f}")
        for miss in s["missed"]:
            print(f"      missed: {miss}")
        for fa in s["false_alarms"][:3]:
            print(f"      false alarm: {fa}")
    print(f"\nRESULT: {'PASS' if r['passed'] else 'FAIL'}" + ("" if r["passed"] else f"  {r['problems']}"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    folder = tempfile.mkdtemp(prefix="spendly-ml-eval-")
    os.environ["SPENDLY_DB_PATH"] = os.path.join(folder, "ml_eval.db")   # before importing db
    import database.db as db
    real = os.path.join(PROJECT_DIR, "database", "spendly.db")
    assert os.path.abspath(db.DB_PATH) != os.path.abspath(real), "refusing to run on the real database"
    report = evaluate(db)
    print_report(report)
    os.makedirs(os.path.join(EVALS_DIR, "results"), exist_ok=True)
    path = os.path.join(EVALS_DIR, "results", f"{date.today().isoformat()}_ml.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"run": {"date": datetime.now().isoformat(timespec="seconds")}, **report}, fh,
                  ensure_ascii=False, indent=1)
    print(f"Saved: {os.path.relpath(path, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
