# Spendly

Personal expense tracker built with Flask and SQLite. Developed on Windows with PowerShell.

## Rules (follow these every time)

### Tech stack
- **Flask only** for the backend. Don't use other web frameworks or Flask extensions.
- **SQLite only**, using Python's built-in `sqlite3` module. **No ORM** (no SQLAlchemy, Peewee, etc.).
- **Vanilla JS only** on the frontend. No frameworks, libraries, or build tools (no React, jQuery, npm, etc.).
  - **One approved exception: Chart.js 4.4.1** from cdn.jsdelivr.net (pinned version + `integrity` hash), used only on `/analytics`. Don't add other libraries without asking.
  - Charts only draw numbers: **all totals come from SQL in `db.py`**, passed to the page as JSON (`<script type="application/json">` + `tojson`). The browser never calculates totals.

### Code organisation
- **All routes go in `app.py`.** Don't use blueprints or put routes in other files.
- **All database code goes in `database/db.py`.** This covers connections, table creation and queries. `app.py` calls functions from `db.py` and never runs SQL itself.

- **All AI code goes in `ai_agent.py`** (chat assistant: LangChain `create_agent`) **and `ai_insights.py`** (analytics insights: a custom LangGraph `StateGraph`). Both get their model ONLY from `get_chat_model()` / `get_vision_model()` in `ai_agent.py` (provider switch below).
- Insights workflow: `fetch_data` (SQL + the code forecast) → `analyse` (all numbers computed in code) → `write_insights` (AI, JSON, facts only) → `check_output` (every ₹ amount and % must be in the facts; one retry, then a plain code-written summary). Read-only. `python ai_insights.py` prints the graph.
  - **Every fact has a fixed label** (`Facts.add`: `[spent_so_far]`, `[budget]`, `[savings]` …) and every number in it is labelled (`spent_so_far`, `budget_used_pct`, `savings`, `savings_rate_pct`, …). `label_problem()` rejects a number next to "savings"/"income" (or a % next to "budget") that isn't a fact with that label. **No income → no income/savings facts, no savings tip pattern**, and any savings/income word next to a number is rejected (it once wrote "your savings so far are ₹22,385 (60.5% of income)": spending + budget used, for a user with no income).
  - **% change = month-to-date vs the SAME days of last month** (`same_days_last_month()` in `forecast.py`: 1–7 Nov vs 1–7 Oct), labelled "vs same days last month", on the insights card AND the analytics summary card (it once showed "▼ 68.3%" = 7 days vs all of October). Category % changes too. "Forecast vs last month" compares the month-end forecast with the full last month.
  - Budgets the forecast says will likely go over become `[budget_forecast]` facts / tips ("Food is on track to exceed your ₹10,000 budget around 27 Nov"). Tests: `tests/test_insights.py`; evals: `insights_*` cases (kind `insights_check`, `grade_insights`).

### AI
- **Two providers, one switch: `LLM_PROVIDER` = `ollama` (default, local `qwen2.5:7b` / `gemma3:4b` at `http://localhost:11434`) or `groq` (cloud, for servers without a GPU, e.g. AWS: `GROQ_CHAT_MODEL` `openai/gpt-oss-120b`, `GROQ_VISION_MODEL` `qwen/qwen3.8-27b`) via `langchain-groq`.** No other cloud provider. Models are created ONLY in `get_chat_model(json_mode=False)` / `get_vision_model()` in `ai_agent.py`; tests patch exactly those.
  - **`GROQ_API_KEY` comes only from the environment / `.env` (`config.py`). Never print, log, return or hardcode it** (ChatGroq keeps it as a `SecretStr`; `/health` and errors never include it).
  - Groq 429: `_GroqChat` waits for `retry-after` and retries ONCE at the model call, only if the wait is ≤ `GROQ_RETRY_MAX_WAIT` (5 s); otherwise `AIBusy` → routes answer **429 "AI is busy, try again in a moment."** (never retry the whole agent run: the chat memory would get the message twice).
  - gpt-oss models get `reasoning_effort="low"`. ChatGroq sends `temperature=0` as `1e-8`.
  - Voice never uses the provider (local faster-whisper).
- If the AI is offline, the key is missing/wrong or the model is missing, AI calls raise `AIError` (`_call_ai()` maps Ollama, httpx and every `groq` error) with a clear, user-safe message. Catch it; never let it crash the app.
- **Agent tools never take a `user_id`.** The user id always comes from `session["user_id"]` in `app.py`.
- **Tools validate with the same rules as the forms** (`validate_expense_form` is passed in from `app.py`). Never trust AI output.
- **The AI never writes to the database directly.** Tools (add / update / delete / find) only prepare pending actions. Changes happen only in `/ai/confirm` after the user clicks Confirm, and that route re-checks ownership, validation and that the expense is unchanged since the preview.
- **The AI never sees or chooses expense ids.** It describes an expense (`match_*`) and code searches only the logged-in user's expenses. 0 matches → "No matching expense found"; 2+ → the user picks from a list (`/ai/select`). Never guess.
- **Only values the user typed.** Code drops a match date or amount the user didn't mention, and edit amounts/dates the user didn't type (see `ai_agent.py`).
- The AI's reply text is not trusted. Code replaces raw tool-call text and false "has been added/updated/deleted" claims.
- **Numbers always come from the database, never from the AI.** Spending questions use `get_spending_summary` / `find_expenses` (read-only, SQL does the maths), and the reply shown is the tool's own answer sentence. Any ₹ amount in other AI replies must appear in a tool result.
- **Time phrases are resolved in code** (`resolve_period()`: today, this week, last month, "october", …), never by the AI.
  - **The user's period always wins**: when `detect_period()` finds one, `find_expenses` / `get_spending_summary` use it whatever the AI passes (it once sent an unknown `date=` argument, then no period → all time). An empty period says "You have no … expenses today", never all-time data.
  - **Typos are fixed in code first** (`fix_typos()` / `TYPO_FIXES`: ths/mnth/wk/todya/yesterdy/avrage/amout/spnd…). A word that looks like a time word but isn't known ("this mnoth", "tooday": `unclear_time_word()`) → `UnclearPeriod` → the user is asked, never all time.
- **"How much / total / spending" = the SUM** (`is_total_question()`): if the AI lists rows instead, code runs `get_spending_summary` and drops the `find_expenses` list. Only a description word the USER typed ("how much on chai?") keeps a search (the AI once made up `description='chai'`).
- Questions are read-only: write tools refuse when the message is a question (`is_question()`).
- **Corrections are EDITS, never adds** (`correction_from_message()`, code only): "X was actually Y", "X should be Y", "correct X to Y", "X was not A, it was B", Hinglish "X 300 tha nahi, 350 tha" → `update_expense_tool` args (Y = amount, category or date). `add_expense_tool` refuses during a correction; if the AI prepares no edit, code calls `update_expense_tool` itself (same checks, Confirm still needed). Questions and "it/that" are not corrections. A Hinglish correction gets its English from code (`correction_sentence()`), not the translator (it once turned one into "How much did the movie cost?"). Tests: `test_correction_*` in `tests/test_ai_safety.py`.

### Text-to-SQL (`run_sql_query` in `ai_agent.py`, `run_user_sql()` in `db.py`)
- Only for questions no fixed tool answers (weekday vs weekend averages, which day of the week, …). The AI writes ONE SELECT; it's never trusted.
- `run_user_sql()`: separate **read-only** connection (`mode=ro` + `PRAGMA query_only=ON`) → per-user **TEMP TABLES** `my_expenses(amount, category, date, description)` / `my_income(amount, source, date, description)` filled with a parameterized `WHERE user_id = ?` (no id/user_id columns) → **authorizer** allow-list: SELECT, reads of those two tables and the query's own CTEs, built-in functions; everything else denied (real tables even via `COUNT(*)`, users, budgets, sqlite_master, PRAGMA, ATTACH, writes) → one statement, ≤ 200 rows, 2 s (progress handler).
  - Temp TABLES, not views: the authorizer reports a CTE like a view, so `WITH my_expenses AS (SELECT * FROM main.expenses)` could pass as the view. Don't switch back to views.
- Time phrases: the tool's `period` (resolved in code, tables limited to it). Date literals / `'now'` in the SQL are rejected.
- Only filters the user asked for (`sql_filter_problem()`, counts as a failed try): a category/source in the SQL must be named in THIS message (the agent once copied `category = 'Food'` + `AVG` from the previous turn's query). "Most/least/more/less" = SUM; `AVG` only when the user says average.
- Average / per day / day of week / weekday-weekend / compare / highest-lowest questions (`SQL_HINT`): the nudge points to `run_sql_query`; still no tool → "I couldn't answer that safely.", never "say a bit more". A plain-total fallback never answers `SQL_ONLY` questions.
- Comparisons state the winner and difference: two periods ("October or November?") → `period_comparison_answer()` (SQL `compare_spending`), code-written; a 2-row SQL result → code adds "X is higher by ₹D" if the AI didn't (replaces an unverified comparison).
- Day-of-week numbers in results are shown to the AI as "Thursday (4)".
- Counts and thresholds ("how many", "count", "above/below/more than 250", `SQL_COUNT_OR_THRESHOLD`) are `SQL_ONLY` too: never the summary total. For `SQL_ONLY` questions a successful query's answer wins over any fixed-tool sentence; a fixed-tool-only answer gets one nudge to SQL, then "I couldn't answer that safely."
- **No tool loops** (`loop_guard` middleware): max `MAX_SQL_CALLS` = 2 queries per request; the same query again (`normalise_sql`) isn't re-run (earlier result + "RESULT READY … do not run this query again"); a read-only tool repeated with the same arguments stops the agent. Code then answers from what the tools returned (the model once looped until the recursion limit: "The AI got stuck").
  - Write tools: a repeat with the same arguments also stops the agent (the preview is already prepared; code writes "Here's what I prepared…"). A second write call aimed at the same target in one request (same expense for edit/delete, same add / income / budget, `duplicate_of_prepared()`) is not added: the first preview wins. The SQL cap applies ONLY to `run_sql_query`. `recursion_limit` leaves room for the guard's extra step per round. (qwen2.5 once called `add_expense_tool` 10× for "spent 150 on chai today" → every action request was 503 "The AI got stuck".) Tests: `test_repeated_write_call_*` etc. in `tests/test_ai_safety.py`.
- **Fixed measurement templates** (`SPENDING_MEASURES` / `get_spending_measure()` in `db.py`; `pick_template()` in `ai_agent.py`): `average_per_expense` (= AVG(amount), the default for average/avg/mean spend or expense), `average_per_day` (total ÷ days in the period, counted up to today; only for "per day"/"daily"), `average_per_week`, `average_per_month`, `total`, `count`, `min`, `max`. The model passes `measure` to `run_sql_query`; an average in the user's words (`template_measure()`) decides in code, overriding the AI's pick or SQL ("avg spend per expense" once got SUM ÷ COUNT(DISTINCT date) = ₹579, called "per expense"). If the AI never calls `run_sql_query`, code runs the template. The answer and its label come from `template_answer()`, never from the model. Free-form SQL only when no template fits (weekday/weekend, grouped, thresholds, items like "chai", income). Percentage share is `share_answer()`.
  - "how many (food) expenses …" (`COUNT_QUESTION`, no threshold/grouping) → the `count` template, chosen in code.
  - "which day of the week do I spend the most/least (on food)?" → `weekday_answer()`, written in code before the AI from `get_spending_by_weekday()` (SUM over ALL rows, one category at most, ties shown, every weekday as a bullet; "most expenses" counts). Averages, weekday-vs-weekend and items ("…on chai") stay free-form SQL. Tests: `tests/test_question_routing.py`.
- Multi-row results are shown as bullets (`sql_result_bullets`: "- Chai · Food: ₹80.00 on 06 Nov 2026"), never "a | b | c"; dates in SQL answers are "05 Nov 2026" (`readable_dates`). An AI reply that copies a raw `|` table gets the bullets instead.
- Code repairs `AVG(CASE … ELSE 0 END)` → `ELSE NULL` (`repair_avg_else_zero`); "average" questions need `AVG` or a division.
- Answer wording (`sql_reply`): the reply may not claim "per day/week/month/expense" unless the SQL computes that (`unsupported_unit`), else `sql_code_answer` ("Average amount per expense: ₹289.50."). Money in SQL answers = `format_rupees_exact` (₹, Indian grouping, 2 decimals, never "rupees"). `tidy_reply` removes `**` markdown and "Based on the data provided" filler from every reply.
- Failed/blocked → the AI may fix it once; a 2nd failure → "I couldn't answer that safely."
- The reply is shown only if every number in it is in the result (or the user's message); otherwise a code-written result table. Results go to the AI inside DATA markers (≤ 30 rows).
- "Show SQL" toggle under the answer (`sql` in the JSON, `spendly_sql` in chat memory). Terminal: `[AI]   sql ALLOWED/BLOCKED` with reason, rows, ms.
- Tests: `tests/test_text_to_sql.py`.

### Guardrails (`ai_agent.py` + `/ai/assistant`)
- **Checked in code before the AI**, first in `/ai/assistant` (`screen_by_rules()`), then again inside `run_expense_assistant()` on the checked English (so translated Hindi is covered too). A blocked message is answered by code, logged as `[AI] … GUARD <KIND>`, and **never saved to chat memory**, so it can't leak into later turns (once, "hi" got rewritten into an earlier refused "write me a poem").
- **Prompt injection** (`PROMPT_INJECTION`): "ignore/forget your rules/instructions", "you are now…", "pretend", "act as", "reveal/show your system prompt", "SYSTEM:" lines → `INJECTION_REPLY` (logged as a warning, no pending action). The system prompt is never revealed.
- **Greetings** ("hi", "thanks", "bye": the WHOLE message) → a short code reply (`GREETING_REPLIES`). It never looks at the chat, so it never continues an earlier request.
- **Off-topic** → `OFF_TOPIC_REPLY` ("I can only help with your Spendly expenses, income and budgets."). Order: `CREATIVE_REQUEST` (write/tell… a poem, joke, story, code: refused even if it mentions money) → `OFF_TOPIC_WORDS` without `FINANCE_STRONG` (amount, ₹, money words) → `FINANCE_WORDS` / date words, or a short follow-up to an earlier chat → allowed. Only if still unsure: `is_finance_topic()`, one tool-less YES/NO model call; only a clear NO refuses.
- **Indirect injection**: saved text (expense descriptions, receipt text, search results, insights facts, the chat in the rewrite prompt) goes to the model inside `<<<DATA … END DATA>>>` (`wrap_data()`), with `clean_data_text()` so a description can't fake the markers. `SYSTEM_PROMPT`, `WRITE_PROMPT` (insights) and `READ_PROMPT` (receipts) say data is never instructions. `strip_data_markers()` removes markers the AI copies into its reply. Code still guarantees safety: questions can't write, and changes need Confirm.
- **Rate limit** (`app.py`): `AI_RATE_LIMIT` = 30 GPU requests (chat, `/analytics/insights`, receipt scan) per user per `AI_RATE_WINDOW` = 10 minutes, in memory (reset on restart) → 429 with a friendly "take a short break" message. Voice (CPU) isn't counted.
- Tests: `tests/test_guardrails.py`. Test filler messages must look like finance or app messages, or they hit the topic check and use up a scripted `fake_ai` reply.

### AI memory (LangGraph)
- Conversation memory uses a LangGraph checkpointer (`InMemorySaver`, `CHECKPOINTER` in `ai_agent.py`): lost on server restart.
- **One thread per user: `thread_id_for(user_id)` → `"user-<id>"`, with `user_id` only from the session.** "New chat" (`/ai/new-chat`) and logout call `clear_chat()`.
- Only the last `MAX_REMEMBERED_TURNS` turns are kept. Keep `num_ctx` at 4096: 8192 pushed the model partly off the GPU and broke tool calling.
- Follow-up checks may use values the chat SHOWED earlier (tool results, saved results), never the AI's own wording.
- Confirm/Cancel results are stored as "(App notice: …)" user-side messages (two AI messages in a row confused the model).
- Debug: the Flask terminal prints `[AI]` lines with every tool call, its arguments and its result.
- Follow-ups are first rewritten into standalone questions (`rewrite_followup()`, a separate tool-less model call). The agent reads the rewrite; **safety checks always use the user's original words**. The chat on screen shows the original.
  - Only **vague** follow-ups are rewritten (`is_vague_follow_up()`: "and…", "what about…", it/that/one, ≤2 words). Complete questions are never sent to the rewriter.
  - A rewrite may only add a date that's in the user's message or the **previous** question; otherwise it's discarded.
  - No date in this message (or a checked carried-over date) = all time. Dates from older messages never count.
- Today's date goes into the system prompt (rebuilt per request) and in front of every user message.
- Show AI text with `textContent` in JS, never `innerHTML`.
- **Enforce rules in code, not only in the prompt.** Example: messages with negative amounts are rejected by `find_negative_number()` before the AI sees them. Prompts guide the AI; code guarantees the rule.

### UI design
- Colours, shadows and radii are CSS variables in `static/css/style.css` (`--primary`, `--gradient`, `--surface`, `--muted`, …). Use them; don't hard-code new colours.
- Category colours/icons: `cat_class` / `cat_icon` template filters + `.cat-<name>` classes in `profile.css`. Charts use the same colours.
- Animations are CSS (plus `main.js` count-up). Everything must respect `prefers-reduced-motion` (see the rule at the end of `style.css`).
- Count-up (`data-count`) only animates: it always ends on the exact text the server rendered.
- **Dark mode**: follows the system setting, or the navbar 🌙/☀️ toggle (saved in localStorage, applied by `theme.js` in `<head>`). Every colour must be a token with a dark value in `style.css` — a hard-coded colour will stay light in dark mode. Charts read the tokens and redraw on the `spendly:themechange` event.

- **AI wording depends on the provider**: "private / local / on your computer / no API keys" is only true with Ollama. Page text about the AI comes from `AI_TEXT[provider]` in `app.py` (context processor `provider_text` → `ai_text.*`, reads `ai_agent.PROVIDER` per request); never hardcode such claims in a template. Tests: `tests/test_provider_text.py`.

### Templates & links
- **Every page template extends `templates/base.html`** with `{% extends "base.html" %}`.
- **Use `url_for()` for all links** in templates and redirects: pages, forms, static files. Never hardcode URL paths.

### Security
- **Always use parameterized SQL queries** with `?` placeholders. Never build SQL with f-strings, `+`, `%` or `.format()`.
  - ✅ `conn.execute("SELECT * FROM expenses WHERE id = ?", (expense_id,))`
  - ❌ `conn.execute(f"SELECT * FROM expenses WHERE id = {expense_id}")`

### Dependencies
- **Don't add new pip packages without asking first.** Current packages (`requirements.txt`): flask, werkzeug, python-dotenv (loads `.env` in `config.py`), gunicorn (Docker server only; doesn't run on Windows), pytest, pytest-flask, langchain, langchain-core, langchain-ollama, langchain-groq (only when `LLM_PROVIDER=groq`), opencv-python-headless (no GUI libs: never use `cv2.imshow`), Pillow, pymupdf, faster-whisper, mcp (2.x, MCP server only), scikit-learn (only to COMPARE models in the ML backtest; imported lazily, the app runs without it).

### Running
- The app **runs on port 5001**: `app.run(debug=config.DEBUG, port=config.PORT)` locally, gunicorn in Docker.

### Configuration (`config.py`)
- **Every setting comes from `config.py`**, which reads environment variables (real env > `.env` via python-dotenv > defaults): `LLM_PROVIDER`, `GROQ_API_KEY`, `GROQ_CHAT_MODEL`, `GROQ_VISION_MODEL`, `GROQ_BASE_URL`, `OLLAMA_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_VISION_MODEL`, `WHISPER_MODEL`, `SPENDLY_DB_PATH`, `SECRET_KEY`, `SPENDLY_DEBUG`, `SPENDLY_PORT`. Never hardcode a URL, model name, path or secret in another module; add it to `config.py` and `.env.example`.
- `/health` (no login): JSON `app` / `database` (`check_database()` in `db.py`) / `ai` (`ai_health()` in `ai_agent.py`: `provider`, `reachable`, `url`, `chat_model` / `vision_model` `{name, available}`, or `error`; Ollama: GET `/api/tags`; Groq: GET `/openai/v1/models`, cached 5 min so the 30 s healthcheck doesn't call Groq each time; 2 s timeout; **never the key**). 200 when the database works, even if the AI is down (`status: degraded`); 503 only when the database fails. Tests: `tests/test_health.py`, `tests/test_llm_provider.py`.

### Receipt scanning (`vision.py`)
- Models: the chat model for chat/insights, **`get_vision_model()`** for reading receipt images (`gemma3:4b` on Ollama, `GROQ_VISION_MODEL` on Groq).
- **No vision model on this server** (Groq with `GROQ_VISION_MODEL` empty, no key, or not in Groq's model list: `vision_available()`) → the scan page shows "Receipt scanning isn't available on this server" with no upload form; a POST is 503 and never reads the image.
- Pipeline: `load_upload` (real JPG/PNG via Pillow, or PDF page 1 via PyMuPDF; ≤ 5 MB; contents must match the extension) → `preprocess` (OpenCV: resize ≤ 1600 px, grayscale, CLAHE contrast, sharpen) → `read_receipt` (vision model, JSON incl. `business_type`) → `interpret` (CODE: amount > 0, day-first Indian dates, category, notes).
- Category is decided in code: `business_type` → `BUSINESS_TYPE_CATEGORY`; if missing/"other"/invalid, whole-word keywords in shop name AND items; health insurance → Health. The user can still change it.
- The amount must be the FINAL amount payable/paid (Net Amount, Grand Total, Amount Paid, Total Due, Premium Amount), never Gross/Subtotal/Arrears. For utility/insurance bills use the bill/payment date, not the due date or policy period.
- Uploads are processed **in memory only** (no files written). The preview form posts to the normal `/expenses/add` route, so nothing is saved without Confirm.

### Voice input (`voice.py`)
- faster-whisper **"small" on the CPU (`int8`)**, loaded once and reused: the GPU stays free for Ollama. Downloaded once on first use (~480 MB), then offline.
- **The user picks the language** (EN / हिंदी toggle next to the mic, saved in localStorage); auto-detection is only logged, never used (it heard Indian-accented English as Hindi). EN → `language="en"`; हिंदी → `language="hi", task="translate"`, so the assistant always gets English. Both use `EXPENSE_PROMPT` (expense vocabulary).
- Safety net: a transcript with any non-Roman letters → "That didn't sound right — try again or type it" (the box is not filled).
- `/ai/voice` (logged-in, JSON): max 5 MB / 30 s, processed **in memory only**, never stored. Silence/no speech → "I didn't catch that, try again." Model can't load → 503 `VoiceUnavailable`; typing still works.
- The transcript only fills the input box; the user reads/fixes it and presses Send (auto-send is an opt-in setting, off by default). All normal checks + Confirm still apply.
- 🔊 read-aloud uses the browser's `speechSynthesis` (off by default). Settings live in localStorage.
- Tests mock the model (`fake_whisper` in `tests/test_voice.py`); they never load Whisper.

### Hindi / Hinglish typed messages (`ai_agent.py`)
- Step 0 of the chat, before the follow-up rewrite. **Detection is code only** (`detect_language()`): Devanagari → hindi; 2+ words from `HINGLISH_WORDS` → hinglish; otherwise English, which skips translation (no AI call).
- **A code dictionary runs before the model** (`apply_hindi_dictionary()`): time phrases (`HINDI_TIME_PHRASES`: aaj → today, kal → yesterday, parso → day before yesterday, is/pichle hafte/mahine/saal → this/last week/month/year) and number words (`hindi_numbers_to_digits()`: do sau → 200, dedh sau → 150, saade teen sau → 350, दो सौ → 200, २०० → 200). Ambiguous words ("do", "ek", "saath") only count before sau/hazaar/rupaye. (The model once turned "parso" into "petrol".)
- `translate_message()` (qwen2.5:7b, tool-less) is then checked in code: same numbers as the dictionary text, every time phrase the dictionary put in still there and none added, no new date words (`HINDI_DATE_WORDS`), no category name or item that implies one ("petrol" → Transport, `CATEGORY_HINTS`) unless the user's words back it, no Devanagari. The terminal logs original → dictionary → AI. A clear expense statement (amount + date + item after the dictionary, not a question) gets **one** retry with `SIMPLE_TRANSLATE_PROMPT`, held to the same checks; every rejected reply is logged raw. Any failure → `MessageRejected("Sorry, I couldn't understand that — try in English?")` → 400, nothing runs.
- After translation, the **checked English** is the `message` all safety checks read (the English regexes can't read Hindi); the screen shows the original plus `🌐 Understood as: "…"`. Negative/zero amounts are re-checked on the English. A Hindi question word in the original (`is_hindi_question()`) makes the turn read-only.
- "Reply in Hindi" (localStorage, off by default, sent as `reply_lang: "hinglish"`): `translate_reply_to_hinglish()` keeps the English reply unless every ₹ amount, number and date is unchanged. Memory keeps English; the Hinglish is stored as `spendly_display`.
- Tests: `tests/test_hindi.py`.

### Forecast & unusual expenses (classic ML, not LLM: `forecast.py`, `anomaly.py`)
- Pure functions on expense rows from `get_expenses_up_to()` in `db.py`, plus thin `get_*` wrappers. No model call, no AI numbers.
- **Forecast** (`forecast_month` / `build_forecast`): fixed monthly items (`find_recurring`: same description once a month in 3 of the last 4 months incl. this one, day spread ≤ 7) + everyday daily totals of the last 90 days split weekday/weekend (recency-weighted, ₹0 days count, one-offs flagged by `anomaly.py` left out unless "This is normal") → 1,000 two-level bootstrap simulations (fixed seed). Median = forecast, P10–P90 = the range shown. Rounded to ₹100.
  - **Never guess**: < 42 days of history or < 30 expenses → `{"status": "not_enough_data"}` → "Not enough data for a forecast yet".
  - Budget: chance over = share of simulated months over the limit; ≥ 50% → "You'll likely go over your ₹X budget around <date>" (median path's crossing day). Top 3 categories by the last 90 days (one-offs excluded), each with its own range.
  - Always the CURRENT month (like budgets), whatever the date filter. Today counts as still running (`partial_today`).
- **Anomalies** (`find_anomalies`): each expense vs its EARLIER peers (last 60 in the category; only those sharing a description word when ≥ 2 do). Flag = robust z (median/MAD) ≥ 3.5 AND ≥ 3× typical AND ≥ ₹500 more. Never judged: < 8 earlier expenses in the category or < 28 days of history. High amounts only. Text: "₹3,280 is about 6× your typical Food expense (₹520)".
  - "This is normal": `POST /expenses/<id>/normal` → `anomaly_dismissals` (amount kept: an edited expense is judged again; deleted with the expense). 404 for other users' expenses.
- UI: dashboard (profile) forecast card + ⚠️ `<details>` badge in Transactions; analytics "actual vs forecast" line (`charts.forecast`, only when there is a forecast). The browser only draws.
- AI: `forecast_answer()` / `anomaly_answer()` answer in CODE before the agent (like `weekday_answer`); the agent also has read-only `get_forecast(category)` / `get_anomalies(period, category)` (`ML_TOOLS`: fallbacks never replace their answer). "next month" → only the current month. A message that records spending (`RECORDS_SPENDING`) goes to the agent.
- Demo data: `python scripts/seed_ml_demo.py` re-creates ONLY `ml_demo@spendly.com` / demo123 (6 months, fixed seed, 5 deliberate spikes in `SPIKES`).
- Measured: `python evals/ml_eval.py` (backtest day 10/20 of every past month: MAE, MAPE, range coverage vs run-rate and linear regression; spikes: precision/recall for MAD, IQR, mean/std, IsolationForest). Eval group `ml` in `run_evals.py`. Tests: `tests/test_forecast.py`, `tests/test_anomaly.py`.

### Income & savings
- Table `income` (amount > 0, `source` ∈ Salary, Freelance, Business, Gift, Refund, Other). **Never store income as negative expenses.**
- Income uses the same validation as expenses (`validate_money_form`) and the same security (user id from session, 404 for others').
- Balance and savings rate come from SQL (`get_money_summary`): savings rate = (income − expenses) ÷ income × 100, `None` when there's no income. Profile cards and analytics follow the date filter.
- AI: `add_income_tool` (needs Confirm; amount must be typed by the user) and `get_savings_summary` (read-only). Saving/income questions never fall back to the spending summary.
  - The description is optional: the AI never asks for it. If the AI uses no tool on a CLEAR income message (`income_from_message()`: one amount, money coming in, one known source: family/friends → Gift "from mom", freelance → Freelance, salary → Salary, refund/returned → Refund), code calls `add_income_tool` itself and shows the preview card. Unclear amount or direction → the AI may ask.

### Budgets
- Table `budgets`: one row per budget; `category IS NULL` = overall monthly budget. A unique index on `(user_id, COALESCE(category, ''))` allows one budget per category and one overall; `CHECK (monthly_limit > 0)`.
- Budgets are always for the **current month** (ignore the profile date filter). Spent / percent / remaining come from SQL (`get_budget_status`).
- Colours: green < 70%, orange 70–99%, red ≥ 100%. Adding an expense (form or AI Confirm) warns when it crosses 80% or 100% — both paths use `add_expenses_with_budget_check()`.
- AI: `get_budget_status` (read-only) and `set_budget_tool` (needs Confirm; amount must be typed by the user).

### MCP server (`mcp_server.py`)
- Exposes Spendly to external AI clients (Claude Desktop, Claude Code) over **stdio**, using the official `mcp` SDK 2.x (`MCPServer`, which is FastMCP renamed in 2.x).
- Tools: `get_spending_summary`, `list_expenses`, `get_budget_status`, `get_savings` (read-only) and `add_expense`, `add_income` (save immediately). **No edit or delete tools**, and don't add any.
- Acts as **ONE user**: `SPENDLY_USER_EMAIL` from `.env` next to the file (the real environment wins; `.env` is git-ignored, see `.env.example`). Never hardcode it, and tools never take a user id. Missing/unknown email → clear error (startup exits with code 1).
- Reuses the website's rules: `validate_expense_form` / `validate_income_form` and `add_expenses_with_budget_check` from `app.py`, and `db.py` queries (SQL does all totals). Invalid input → a tool error, nothing saved.
- **Never print to stdout** in the server or anything it imports: stdout carries the MCP protocol. Logs go to stderr (`[MCP]` lines); every write is logged as `WRITE add_expense …` / `WRITE add_income …`.
- Tool descriptions spell out the categories, sources, `YYYY-MM-DD` dates and ₹ amounts. Keep them accurate when changing a tool.
- Tests: `tests/test_mcp_server.py` (in-process MCP `Client` on the test database; one test runs the real script to check stdout stays empty).

### Tests
- Tests live in `tests/` (`conftest.py` holds the shared fixtures). Run `pytest` after every change; all must pass.
- **Tests never touch the real `spendly.db`**: `SPENDLY_DB_PATH` points at a temp file and every test gets a fresh empty database. A session check fails the run if `spendly.db` changes.
- **Tests never call the real Ollama model or Groq**: use the `fake_ai(...)` fixture with `tool_call(...)` replies, or `ollama_down` / `ai_must_not_be_called`. An autouse fixture pins every test to `PROVIDER="ollama"` with an empty Groq key, whatever `.env` says; Groq tests (`tests/test_llm_provider.py`) use a fake key and fake HTTP.
- New features and bug fixes get a test.

## How to run evals

Evals measure how well the AI answers with the **real** models (the provider in `LLM_PROVIDER`). They are separate from pytest (pytest uses fakes and must stay fast and offline). Ollama must be running (or `GROQ_API_KEY` set for Groq).

```powershell
python evals/run_evals.py                      # all cases (~100, about an hour)
python evals/run_evals.py --group hinglish     # one group or tag: totals, averages, counts, minmax, percentage,
                                               #   comparison, lists, empty, hinglish, wording, followups,
                                               #   actions, safety, receipts, insights, ml; tags: regression, edge
python evals/run_evals.py --only-failures      # re-run the failures of the newest results file
python evals/run_evals.py --model llama3.1:8b  # same cases + grading with another model (compare scores)
python evals/run_evals.py --judge              # also run the optional LLM-as-judge on free-text cases
python evals/run_evals.py --case total_today   # one case (repeatable)
$env:LLM_PROVIDER="groq"; python evals/run_evals.py   # on Groq: 4 s between cases (--delay), waits out 429s,
                                               #   stops on a daily limit (not-run cases → --only-failures)
python evals/make_cases.py                     # rebuild evals/cases.json after editing the grid
python evals/ml_eval.py                        # forecast backtest + anomaly precision/recall (no Ollama, ~15 s)
```

- **Never real data**: the runner copies `database/spendly.db` to a temp file (SQLite backup API, read-only source), points `SPENDLY_DB_PATH` at the copy *before* importing the app, and asserts the app uses it. It seeds an eval user (`evals/seed.py`, dates relative to today) and an "other user" whose marker data (₹77,777, `OTHERUSER-SECRET`) must never appear in a reply. Nothing is ever confirmed.
- **Cases** (`evals/cases.json`) are generated by `evals/make_cases.py` from a coverage grid. They hold no numbers: expected values are computed at run time by `evals/ground_truth.py`, with its own fixed parameterized SQL and its own date maths (it deliberately doesn't reuse `db.py` or `resolve_period`, so the app isn't graded against itself). Month names are `{placeholders}`.
- **Grading** (`evals/grading.py`): numbers are pulled from the reply (dates removed) and must match the truth (±0.01, or rounded to a whole rupee); lists must contain every matching amount and no other ₹ amount; previews (session `ai_pending`) must match type/amount/category/date/source exactly; safety passes only if refused/blocked AND the whole-database hash is unchanged AND nothing leaked. Every case also fails if the database changed or a question created a preview.
- **LLM-as-judge** (`--judge`): only for free text (insights tips, Hinglish replies), a short checklist, reported separately from the main score.
- **Output**: score per group/tag and overall, failures (question → expected → got → SQL / tool log), total time, saved to `evals/results/<date>_<model>[_<selection>].json`.
- Every fixed bug gets a regression case (tag `regression`) in `make_cases.py`.
- Receipt images live in `evals/receipts/` (git-ignored: they are real bills); missing images are skipped.

## Commands (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1        # activate venv
pip install -r requirements.txt    # install dependencies
python app.py                      # run at http://localhost:5001
pytest                             # run all tests (no Ollama needed, ~7 min: password hashing per test user)
python scripts/seed_ml_demo.py     # (re)create the ML demo user ml_demo@spendly.com / demo123
python test_ai.py                  # check the AI connection (provider from LLM_PROVIDER)
python mcp_server.py               # MCP server over stdio (normally started by Claude, not by hand)
```

## Running with Docker

Phase 1: the app runs in a container, **Ollama stays on Windows** (the container reaches it at `http://host.docker.internal:11434`). Docker Desktop must be running, and so must Ollama for the AI features. With `LLM_PROVIDER=groq` in `.env` (plus `GROQ_API_KEY`), the container uses Groq instead and needs no Ollama (the AWS setup).

```powershell
Copy-Item .env.example .env                      # first time only, then set SECRET_KEY
docker compose up -d --build                     # build the image and start it in the background
docker compose ps                                # STATUS shows (healthy) after ~20 s
Invoke-RestMethod http://localhost:5001/health   # app / database / ai (provider, models) status as JSON
docker compose logs -f spendly                   # follow the logs ([AI] lines, gunicorn access log)
docker compose run --rm spendly pytest           # full test suite inside a throwaway container
docker compose down                              # stop (data is kept in the volumes)
docker image ls spendly                          # image size
```

- **gunicorn with 1 worker + 8 threads**, never more workers: chat memory and the AI rate limit live in process memory.
- Volumes: `spendly-data` → `/app/data/spendly.db` (the database), `whisper-cache` → `/models` (Whisper download, `HF_HOME`). `docker compose down -v` **deletes them, including all data.**
- The container starts with an EMPTY database (demo user only). It does not see `database/spendly.db`. To copy yours in once: `docker compose cp database/spendly.db spendly:/app/data/spendly.db`, then `docker compose restart`.
- `.env` is never copied into the image (`.dockerignore`); compose passes it as `env_file` and overrides `OLLAMA_URL`, `SPENDLY_DB_PATH` and `HF_HOME`.
- Runs as the non-root user `spendly` (uid 1000).
- If `/health` shows `"reachable": false` while Ollama runs on Windows, make Ollama listen on all interfaces: set the Windows user environment variable `OLLAMA_HOST=0.0.0.0` and restart Ollama.

## Structure

```
app.py               # all Flask routes
config.py            # all settings from environment variables / .env (python-dotenv)
Dockerfile           # python:3.11-slim image, non-root, gunicorn on 5001
docker-compose.yml   # service `spendly`, volumes for the database + Whisper cache, healthcheck
database/db.py       # all SQLite code
ai_agent.py          # AI chat assistant (LangChain create_agent) + the provider switch (get_chat_model / get_vision_model)
ai_insights.py       # AI spending insights (custom LangGraph StateGraph)
voice.py             # voice input: faster-whisper speech-to-text on the CPU
mcp_server.py        # MCP server (stdio) for Claude Desktop / Claude Code: 6 tools, one user from .env
evals/               # AI evals with the real models on a temp DB copy (run_evals.py, cases.json, results/)
                     #   + ml_eval.py: forecast backtest and anomaly precision/recall (group `ml`)
forecast.py          # month-end forecast: monthly items + weekday/weekend bootstrap simulation, range, budget
anomaly.py           # unusual expenses: median + MAD per category (word peers), "This is normal"
scripts/seed_ml_demo.py  # (re)creates ONLY ml_demo@spendly.com with 6 months of synthetic data
vision.py            # receipt scanning: upload checks, OpenCV clean-up, vision model (gemma3:4b / Groq), code checks
test_ai.py           # manual AI connection check (Ollama or Groq)
templates/base.html  # shared layout (navbar + footer)
templates/*.html     # pages, each extends base.html (_date_filter.html is a shared include, not a page)
static/css/          # stylesheets
static/js/           # vanilla JS (ai_assistant.js = AI chat box on the profile page)
```
