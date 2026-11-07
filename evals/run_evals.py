"""Spendly AI evaluation harness (separate from pytest: uses the REAL models: local Ollama, or
Groq when LLM_PROVIDER=groq).

Safety: the app runs on a TEMPORARY COPY of database/spendly.db (made with SQLite's backup API
from a read-only connection), with a seeded eval user. Your real database is never opened for
writing, and nothing is ever confirmed, so even the copy only changes when the eval user is seeded.

    python evals/run_evals.py                     # all cases with qwen2.5:7b
    python evals/run_evals.py --group hinglish    # one group (or tag, e.g. regression)
    python evals/run_evals.py --only-failures     # re-run the failures of the last results file
    python evals/run_evals.py --model llama3.1:8b # compare another model (same cases, same grading)
    python evals/run_evals.py --judge             # also run the LLM-as-judge on free-text cases
    $env:LLM_PROVIDER="groq"; python evals/run_evals.py   # same cases on Groq (paced for the free tier)

Groq: a pause between cases (--delay), longer waits on 429 inside the run, and if Groq still
refuses (e.g. the daily token limit) the run STOPS: the cases not run are saved as "not-run"
and --only-failures runs them later. Receipt cases are skipped on Groq unless you pass
--include-receipts: evals/receipts holds real bills, and Groq is a cloud service.

Results: evals/results/<date>_<model>.json (score per group, failures, timings).
"""
import argparse
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import date, datetime

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EVALS_DIR)
REAL_DB = os.path.join(PROJECT_DIR, "database", "spendly.db")
RESULTS_DIR = os.path.join(EVALS_DIR, "results")
CASES_FILE = os.path.join(EVALS_DIR, "cases.json")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EVALS_DIR)


def parse_args():
    p = argparse.ArgumentParser(description="Spendly AI evals (real local models, temporary DB copy)")
    p.add_argument("--group", help="only cases in this group or with this tag (e.g. hinglish, regression)")
    p.add_argument("--case", action="append", help="only this case id (repeatable)")
    p.add_argument("--only-failures", action="store_true", help="re-run the failures of the last results file")
    p.add_argument("--model", default=None, help="chat/insights model (default: the provider's configured model)")
    p.add_argument("--vision-model", default=None, help="receipt model (default: the provider's configured model)")
    p.add_argument("--delay", type=float, default=None,
                   help="seconds to wait between cases (default: 0 on Ollama, 4 on Groq)")
    p.add_argument("--include-receipts", action="store_true",
                   help="on Groq: also send the real receipt images in evals/receipts to Groq")
    p.add_argument("--judge", action="store_true", help="run the optional LLM-as-judge on free-text cases")
    p.add_argument("--judge-model", default=None, help="judge model (default: the chat model)")
    p.add_argument("--receipts", default=os.path.join(EVALS_DIR, "receipts"), help="folder with receipt images")
    p.add_argument("--source-db", default=REAL_DB, help="database to COPY (it is only read)")
    p.add_argument("--verbose", action="store_true", help="also print the app's [AI] log lines")
    return p.parse_args()


# ---------- The temporary database copy (set up BEFORE the app is imported) ----------

def make_db_copy(source):
    """Copy the database with SQLite's backup API from a READ-ONLY connection."""
    folder = tempfile.mkdtemp(prefix="spendly-evals-")
    copy = os.path.join(folder, "eval.db")
    if os.path.exists(source):
        src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        dst = sqlite3.connect(copy)
        try:
            src.backup(dst)
        finally:
            src.close()
            dst.close()
    return folder, copy


def db_fingerprint(path):
    """A hash of every row in the database: any write anywhere changes it."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return hashlib.sha256("\n".join(conn.iterdump()).encode("utf-8")).hexdigest()
    finally:
        conn.close()


def file_fingerprint(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------- Capturing the app's [AI] log per case (tools and SQL used) ----------

class Capture(logging.Handler):
    KEEP = re.compile(r"tool call|sql |TEMPLATE|code (fallback|answer|compare)|GUARD|rewrite|LOOP STOP|"
                      r"translate|reply hindi|\[receipt\]|\[insights\]", re.I)

    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        line = record.getMessage().strip()
        if self.KEEP.search(line):
            self.lines.append(line[:300])


def fill(text, values):
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), text)


def fill_spec(obj, values):
    """Placeholders in nested case data (messages and turns)."""
    if isinstance(obj, str):
        return fill(obj, values)
    if isinstance(obj, list):
        return [fill_spec(x, values) for x in obj]
    if isinstance(obj, dict):
        return {k: fill_spec(v, values) for k, v in obj.items()}
    return obj


def last_results_file(model):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")), key=os.path.getmtime, reverse=True)
    same = [f for f in files if json.load(open(f, encoding="utf-8")).get("run", {}).get("model") == model]
    return (same or files or [None])[0]


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    started = time.perf_counter()
    today = date.today()

    folder, copy = make_db_copy(args.source_db)
    os.environ["SPENDLY_DB_PATH"] = copy          # must be set before `import app`
    real_before = file_fingerprint(REAL_DB)

    import ai_agent
    import ai_insights
    import app as spendly
    import database.db as db
    import httpx
    import vision
    from langchain_ollama import ChatOllama

    import grading
    from ground_truth import GroundTruth, placeholders, resolve_date
    from seed import EVAL_EMAIL, EVAL_PASSWORD, LEAK_MARKERS, seed

    assert os.path.abspath(db.DB_PATH) == os.path.abspath(copy) != os.path.abspath(REAL_DB), \
        "refusing to run: the app is not using the temporary copy"

    provider = ai_agent.PROVIDER
    model = args.model or ai_agent.chat_model_name()
    vision_model = args.vision_model or ai_agent.vision_model_name()
    if provider == "groq":
        ai_agent.GROQ_CHAT_MODEL = model
        ai_agent.GROQ_VISION_MODEL = vision_model or ""
        # Inside a run, wait out per-minute limits instead of failing the case.
        ai_agent.GROQ_RETRY_MAX_WAIT = 65
        ai_agent.GROQ_RATE_LIMIT_RETRIES = 3
    else:
        ai_agent.OLLAMA_MODEL = model
        ai_agent.OLLAMA_VISION_MODEL = vision_model
    delay = args.delay if args.delay is not None else (4.0 if provider == "groq" else 0.0)
    spendly.AI_RATE_LIMIT = 10 ** 9       # this process only: evals send many requests

    if provider == "groq":
        available = ai_agent.groq_models()          # never prints the key
        if isinstance(available, str):
            sys.exit(f"Groq isn't usable: {available}. Set GROQ_API_KEY in .env.")
        for name in {model, vision_model, args.judge_model or model} - {None}:
            if name not in available:
                print(f"WARNING: model {name!r} is not available on this Groq account")
    else:
        try:
            installed = {m["name"] for m in httpx.get(f"{ai_agent.OLLAMA_URL}/api/tags", timeout=5).json()["models"]}
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Ollama isn't reachable at {ai_agent.OLLAMA_URL} ({e}). Start Ollama first.")
        for name in {model, vision_model, args.judge_model or model}:
            if name not in installed and f"{name}:latest" not in installed:
                print(f"WARNING: model {name!r} is not installed in Ollama (ollama pull {name})")

    # Count the tokens each case uses on Groq (the free tier limits tokens per minute and day).
    tokens = {"case": 0, "total": 0}
    if provider == "groq":
        original_generate = ai_agent._GroqChat._generate

        def counting_generate(self, *a, **k):
            result = original_generate(self, *a, **k)
            used = ((result.llm_output or {}).get("token_usage") or {}).get("total_tokens") or 0
            tokens["case"] += used
            tokens["total"] += used
            return result
        ai_agent._GroqChat._generate = counting_generate

    capture = Capture()
    for handler in list(ai_agent.AI_LOG.handlers):
        if not args.verbose:
            ai_agent.AI_LOG.removeHandler(handler)
    ai_agent.AI_LOG.addHandler(capture)

    db.init_db()
    user_id, _ = seed(db, today)
    truth = GroundTruth(copy, user_id, today)
    values = placeholders(today)
    judge_llm = None
    if args.judge:
        judge_llm = (ai_agent._groq_model(args.judge_model or model, json_mode=True) if provider == "groq"
                     else ChatOllama(model=args.judge_model or model, base_url=ai_agent.OLLAMA_URL,
                                     temperature=0, format="json"))

    client = spendly.app.test_client()
    response = client.post("/login", data={"email": EVAL_EMAIL, "password": EVAL_PASSWORD})
    assert response.status_code == 302, "eval user could not log in"

    # ----- choose the cases -----
    with open(CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)
    suffix = ""
    if args.only_failures:
        last = last_results_file(model)
        if not last:
            sys.exit("No earlier results in evals/results to re-run.")
        failed = {c["id"] for c in json.load(open(last, encoding="utf-8"))["cases"]
                  if c["status"] in ("fail", "error", "not-run")}
        print(f"Re-running {len(failed)} failure(s) from {os.path.basename(last)}")
        cases = [c for c in cases if c["id"] in failed]
        suffix = "_rerun"
    if args.group:
        cases = [c for c in cases if c["group"] == args.group or args.group in c.get("tags", [])]
        suffix += f"_{args.group}"
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
        suffix += "_cases"
    if not cases:
        sys.exit("No cases selected.")

    # ----- the classic-ML group: the demo user is seeded into the COPY before any case runs, so the
    # per-case "database changed" check still holds. ml cases grade the backtest / spike detection
    # (evals/ml_eval.py); ml_chat cases ask the real model as the demo user. -----
    ml_report, ml_client, ml_user = None, None, None
    if any(c.get("kind") in ("ml", "ml_chat") or c.get("as_ml_demo") for c in cases):
        import ml_eval
        import seed_ml_demo
        import anomaly
        import forecast
        ml_report = ml_eval.evaluate(db, today)      # seeds ml_demo@spendly.com in the temp copy
        ml_user = db.get_user_by_email(seed_ml_demo.DEMO_EMAIL)["id"]
        ml_client = spendly.app.test_client()
        r = ml_client.post("/login", data={"email": seed_ml_demo.DEMO_EMAIL, "password": seed_ml_demo.DEMO_PASSWORD})
        assert r.status_code == 302, "ML demo user could not log in"

    def ml_expected(expect):
        """The numbers a forecast / anomaly reply must contain, from forecast.py / anomaly.py."""
        if "forecast" in expect:
            category = expect.get("category")
            f = (forecast.get_category_forecast(ml_user, category, today) if category
                 else forecast.get_forecast(ml_user, today))
            if f["status"] != "ok":
                return [], [f["message"]]
            texts = [f["budget"]["message"]] if expect.get("budget_message") and f.get("budget") else []
            return [f[k] for k in expect["forecast"]], texts
        if "anomalies" in expect:
            items = anomaly.get_anomalies(ml_user)[:5]
            return [a["amount"] for a in items], []
        return [], []

    def pending_previews():
        with client.session_transaction() as s:
            return grading.flatten_pending(dict(s.get("ai_pending", {})))

    def leaked(text):
        return [m for m in LEAK_MARKERS if m.lower() in (text or "").lower()]

    results = []
    print(f"Spendly evals: {len(cases)} case(s), provider {provider}, model {model}, vision {vision_model}, "
          f"today {today}\nDatabase: temporary copy {copy}\n"
          + (f"Pause between cases: {delay:g} s\n" if delay else ""))

    stopped = None                      # set when Groq keeps refusing (rate/daily limit)
    for n, raw_case in enumerate(cases, 1):
        c = fill_spec(raw_case, values)
        if stopped:
            results.append({"id": c["id"], "group": c["group"], "tags": c.get("tags", []),
                            "question": c["message"], "turns": c["turns"], "status": "not-run",
                            "expected": "", "got": stopped, "problems": [], "sql": None})
            continue
        if delay and n > 1:
            time.sleep(delay)
        tokens["case"] = 0
        kind = c.get("kind", "chat")
        expect = c["expect"]
        capture.lines = []
        t0 = time.perf_counter()
        before = db_fingerprint(copy)
        out = {"id": c["id"], "group": c["group"], "tags": c.get("tags", []), "question": c["message"],
               "turns": c["turns"], "status": "pass", "expected": "", "got": "", "problems": [], "sql": None}
        try:
            if kind == "sql":
                # A SQL attack straight at the text-to-SQL sandbox: it must be refused.
                try:
                    result = db.run_user_sql(user_id, c["message"])
                    out["got"] = f"ran: {result['rows'][:5]}"
                    out["problems"].append("the query was NOT blocked")
                except (db.SQLRejected, sqlite3.Error) as e:
                    out["got"] = f"blocked: {e}"
                out["expected"] = "blocked"
            elif kind == "ml":
                # No model: the backtest / precision-recall report, graded by ml_eval's pass marks.
                if c["message"] == "forecast_backtest":
                    f = ml_report["forecast"]
                    s = f["by_model"]["forecast"]["all"]
                    out["expected"] = (f"MAPE <= {ml_eval.MAX_MAPE}%, range coverage >= {ml_eval.MIN_COVERAGE}%")
                    out["got"] = (f"MAE ₹{s['mae']}, MAPE {s['mape']}% (run-rate "
                                  f"{f['by_model']['run_rate']['all']['mape']}%, linear regression "
                                  f"{f['by_model']['linear_regression']['all']['mape']}%), coverage "
                                  f"{f['range_coverage_percent']}% ({f['forecasts']} forecasts)")
                    out["problems"] += ml_eval.forecast_problems(ml_report)
                else:
                    a = ml_report["anomalies"]
                    out["expected"] = f"recall >= {ml_eval.MIN_RECALL}, precision >= {ml_eval.MIN_PRECISION}"
                    out["got"] = "; ".join(f"{m}: P={s.get('precision')} R={s.get('recall')}"
                                           for m, s in a.items())
                    out["problems"] += ml_eval.anomaly_problems(ml_report)
                out["ml"] = {k: ml_report[k] for k in ("forecast", "anomalies", "backtest")}
            elif kind == "ml_chat":
                # The real model, as the demo user: the reply must carry the forecast / anomaly
                # module's own numbers (the ML itself is graded by the "ml" cases).
                ml_client.post("/ai/new-chat")
                r = ml_client.post("/ai/assistant", json={"message": c["message"]})
                body = r.get_json() or {}
                reply = body.get("reply") or body.get("error") or ""
                numbers, texts = ml_expected(expect)
                out.update(got=reply, http=r.status_code)
                out["expected"] = ", ".join([f"₹{n:,.0f}" for n in numbers] + texts)
                missing = [n for n in numbers if not grading.has_number(reply, n)]
                if missing:
                    out["problems"].append(f"missing number(s) {missing}")
                out["problems"] += [f"missing text {t!r}" for t in texts if t not in reply]
                with ml_client.session_transaction() as s:
                    if s.get("ai_pending"):
                        out["problems"].append("a question created a preview")
                if leaked(reply):
                    out["problems"].append(f"leaked another user's data {leaked(reply)}")
            elif kind == "receipt":
                path = os.path.join(args.receipts, c["message"])
                if provider == "groq" and not args.include_receipts:
                    out["status"], out["got"] = "skipped", "real receipt not sent to Groq (use --include-receipts)"
                elif not os.path.exists(path):
                    out["status"], out["got"] = "skipped", f"missing image {path}"
                else:
                    with open(path, "rb") as f:
                        result = vision.scan_receipt(c["message"], f.read(), today)
                    out["expected"] = f"₹{expect['receipt']['amount']} · {expect['receipt']['category']}"
                    out["got"] = f"₹{result.get('amount')} · {result.get('category')} ({result.get('shop')})"
                    out["problems"] += grading.grade_receipt(result, expect["receipt"])[1]
            elif kind == "insights_check":
                # Graded in code: labels of numbers (savings/income), the same-days % change and
                # forecast tips. As the ML demo user (no income, budgets, a forecast) or the eval user.
                import forecast as forecast_module
                uid = ml_user if c.get("as_ml_demo") else user_id
                result = ai_insights.run_insights(uid, today)
                gt = GroundTruth(copy, uid, today)
                try:
                    t = gt.insights_numbers()
                finally:
                    gt.close()
                likely = []
                if expect.get("forecast_tip"):
                    f = forecast_module.get_forecast(uid, today)
                    for item in ([f, *f.get("categories", [])] if f["status"] == "ok" else []):
                        b = item.get("budget")
                        if b and b["status"] == "likely":
                            likely.append((item["category"], b["limit"]))
                text = " ".join([result.get("summary") or ""] + list(result.get("tips") or []))
                out["got"] = (f"[{result.get('status')}/{result.get('source')}] "
                              + " | ".join(f"{n['label']}: {n['value']}" for n in result.get("key_numbers") or [])
                              + f" || {text}")[:900]
                out["expected"] = json.dumps(t, ensure_ascii=False)
                out["problems"] += grading.grade_insights(result, expect, t, likely)
                if leaked(text):
                    out["problems"].append(f"leaked another user's data {leaked(text)}")
            elif kind == "insights":
                result = ai_insights.run_insights(user_id, today)
                text = " ".join([result.get("summary") or ""] + list(result.get("tips") or []))
                out["got"] = f"[{result.get('status')}/{result.get('source')}] {text}"[:600]
                out["judge_text"] = text
                out["status"] = "judge-only"
            else:
                client.post("/ai/new-chat")
                for turn in c["turns"]:
                    client.post("/ai/assistant", json={"message": turn})
                payload = {"message": c["message"]}
                if c.get("reply_lang"):
                    payload["reply_lang"] = c["reply_lang"]
                r = client.post("/ai/assistant", json=payload)
                body = r.get_json() or {}
                reply = body.get("reply") or body.get("error") or ""
                previews = pending_previews()
                out.update(got=reply, sql=body.get("sql"), http=r.status_code, guard=body.get("guard"))
                out["judge_text"] = reply
                if r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {body}")
                if "previews" in expect:
                    wanted = [{**w, **({"date": resolve_date(w["date"], today)} if "date" in w else {})}
                              for w in expect["previews"]]
                    out["expected"] = json.dumps(wanted, ensure_ascii=False)
                    out["got_previews"] = previews
                    out["problems"] += grading.grade_previews(previews, wanted)[1]
                elif expect.get("refused") or expect.get("safe"):
                    refused = (r.status_code == 400 or bool(body.get("guard"))
                               or bool(grading.REFUSAL.search(reply)) or reply == ai_agent.SQL_FAILED_REPLY)
                    out["expected"] = "refused, nothing changed" if expect.get("refused") else "nothing changed, no leak"
                    if expect.get("refused") and not refused:
                        out["problems"].append("not refused")
                    if previews:
                        out["problems"].append(f"created a preview {previews}")
                else:
                    ok, out["expected"], problems = grading.grade_answer(reply, expect, truth, c["message"])
                    out["problems"] += problems
                    if previews:
                        out["problems"].append(f"a question created a preview {previews}")
                if leaked(reply):
                    out["problems"].append(f"leaked another user's data {leaked(reply)}")
            if db_fingerprint(copy) != before:
                out["problems"].append("THE DATABASE CHANGED")
        except Exception as e:  # noqa: BLE001  (one broken case must not stop the run)
            out["status"], out["problems"] = "error", [f"{type(e).__name__}: {e}"]
        if out["status"] == "pass" and out["problems"]:
            out["status"] = "fail"
        if out.get("http") == 429 or any(p.startswith("AIBusy") for p in out["problems"]):
            # Not the model's fault: Groq refused even after waiting. Stop, save, re-run later.
            stopped = "not run: Groq rate limit (try again later with --only-failures)"
            out.update(status="not-run", got=stopped)

        if c.get("judge"):
            if judge_llm is None:
                out["judge"] = {"status": "skipped (run with --judge)"}
            else:
                try:
                    passed, answers, note = grading.judge(judge_llm, out.get("judge_text", ""), c["judge"])
                    out["judge"] = {"status": "pass" if passed else "fail", "answers": answers,
                                    "checklist": c["judge"], "note": note}
                except Exception as e:  # noqa: BLE001
                    out["judge"] = {"status": "error", "note": f"{type(e).__name__}: {e}"}
        out.pop("judge_text", None)
        out["tools"] = capture.lines[:40]
        out["seconds"] = round(time.perf_counter() - t0, 1)
        if provider == "groq":
            out["tokens"] = tokens["case"]
        results.append(out)
        mark = {"pass": "PASS", "fail": "FAIL", "error": "ERR ", "skipped": "SKIP", "judge-only": "JDG ",
                "not-run": "STOP"}[out["status"]]
        judge_mark = f"  judge:{out['judge']['status']}" if "judge" in out else ""
        token_mark = f"  {tokens['case']:6} tok" if provider == "groq" else ""
        print(f"[{n:3}/{len(cases)}] {mark} {c['group']:<11} {c['id']:<30} {out['seconds']:6.1f}s{token_mark}{judge_mark}")
        if stopped:
            print(f"        STOPPED: {stopped}")
        if out["status"] in ("fail", "error"):
            print(f"        expected: {out['expected'][:150]}\n        got:      {str(out['got'])[:150]!r}\n"
                  f"        problems: {'; '.join(out['problems'])[:200]}")

    truth.close()

    # ----- scores -----
    def score(items):
        graded = [r for r in items if r["status"] in ("pass", "fail", "error")]
        passed = sum(r["status"] == "pass" for r in graded)
        return {"passed": passed, "total": len(graded),
                "percent": round(100 * passed / len(graded), 1) if graded else None}

    groups = sorted({r["group"] for r in results})
    tags = sorted({t for r in results for t in r["tags"]})
    judged = [r for r in results if r.get("judge", {}).get("status") in ("pass", "fail", "error")]
    summary = {
        "overall": score(results),
        "by_group": {g: score([r for r in results if r["group"] == g]) for g in groups},
        "by_tag": {t: score([r for r in results if t in r["tags"]]) for t in tags},
    }
    judge_summary = {"passed": sum(r["judge"]["status"] == "pass" for r in judged), "total": len(judged)} \
        if judged else {"status": "not run (use --judge)"}
    failures = [{k: r.get(k) for k in ("id", "group", "question", "turns", "expected", "got", "problems",
                                       "sql", "tools", "got_previews")}
                for r in results if r["status"] in ("fail", "error")]
    seconds = round(time.perf_counter() - started, 1)
    real_changed = file_fingerprint(REAL_DB) != real_before
    report = {
        "run": {"date": datetime.now().isoformat(timespec="seconds"), "today": today.isoformat(),
                "provider": provider, "model": model, "vision_model": vision_model,
                "tokens": tokens["total"] if provider == "groq" else None,
                "not_run": sum(r["status"] == "not-run" for r in results), "judge_model": (args.judge_model or model)
                if args.judge else None, "cases": len(results), "seconds": seconds,
                "selection": suffix.strip("_") or "all",
                "database": "temporary copy of " + os.path.basename(args.source_db)},
        "score": summary, "judge": judge_summary, "failures": failures, "cases": results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_model = re.sub(r"[^\w.-]+", "_", model)
    if stopped:
        suffix += "_partial"
    path = os.path.join(RESULTS_DIR, f"{today.isoformat()}_{safe_model}{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)

    # ----- report -----
    print("\n" + "=" * 64)
    print(f"Score by group ({model}):")
    for g, s in summary["by_group"].items():
        if s["total"]:
            print(f"  {g:<12} {s['passed']:3}/{s['total']:<3} {s['percent']:6.1f}%")
    for t, s in summary["by_tag"].items():
        if s["total"]:
            print(f"  tag:{t:<8} {s['passed']:3}/{s['total']:<3} {s['percent']:6.1f}%")
    o = summary["overall"]
    print(f"OVERALL      {o['passed']:3}/{o['total']:<3} {o['percent'] or 0:6.1f}%")
    if judged:
        print(f"LLM judge (separate, free text): {judge_summary['passed']}/{judge_summary['total']}")
    skipped = [r["id"] for r in results if r["status"] in ("skipped", "judge-only") and "judge" not in r]
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    print(f"Failures: {len(failures)}   Total time: {seconds:.0f}s   Results: {os.path.relpath(path, PROJECT_DIR)}")
    if provider == "groq":
        print(f"Groq tokens used: {tokens['total']:,}")
    if stopped:
        not_run = sum(r["status"] == "not-run" for r in results)
        print(f"STOPPED EARLY: {not_run} case(s) not run ({stopped}). Re-run them with --only-failures.")
    if real_changed:
        print("NOTE: database/spendly.db changed during the run. The evals never write to it, so "
              "this was something else (e.g. the running Flask app).")
    shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    main()
