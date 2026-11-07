"""Hindi / Hinglish typed messages. The real Ollama model is NEVER called: `fake_ai` plays the
translator and the assistant. These tests check OUR code: detection, number words, the checks
on the translation, and that everything after translation keeps the normal safety rules."""
from contextlib import contextmanager
from datetime import date, timedelta

import pytest

import ai_agent
import database.db as db
from conftest import ask_ai, count_expenses, tool_call

CATEGORIES = ["Food", "Transport", "Bills", "Health", "Entertainment", "Shopping", "Other"]


# ---------- Detection (code only, no AI) ----------

@pytest.mark.parametrize("message, expected", [
    ("spent 250 on auto today", "english"),
    ("how much did I spend on food?", "english"),
    ("what do I spend most on?", "english"),
    ("lunch at kal's cafe 200", "english"),                 # one Hindi-looking word is not enough
    ("खाने पर कितना खर्च हुआ", "hindi"),
    ("कल बिजली का बिल 1500", "hindi"),
    ("khane pe kitna kharcha hua", "hinglish"),
    ("aaj auto ke 200 diye", "hinglish"),
    ("salary aayi 30000 aaj", "hinglish"),
    ("aaj chai pe 20 spent", "hinglish"),                   # mixed English + Hinglish
    ("spent 200 on खाना", "hindi"),                         # mixed English + Devanagari
])
def test_detect_language(message, expected):
    assert ai_agent.detect_language(message) == expected


def test_english_is_never_sent_to_the_translator(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("English must skip translation")
    monkeypatch.setattr(ai_agent, "get_chat_model", forbidden)
    assert ai_agent.translate_message("spent 250 on auto", CATEGORIES) == ("spent 250 on auto", "english")


# ---------- Hindi number words -> digits (code only, no AI) ----------

@pytest.mark.parametrize("text, expected", [
    ("do sau", "200"),
    ("dhai sau", "250"),
    ("pachaas", "50"),
    ("hazaar rupaye", "1000 rupaye"),
    ("dedh hazaar", "1500"),
    ("do sau pachaas", "250"),
    ("saadhe teen sau", "350"),
    ("sava sau", "125"),
    ("paune do sau", "175"),
    ("ek hazaar do sau", "1200"),
    ("2 hazaar", "2000"),
    ("aaj auto ke do sau diye", "aaj auto ke 200 diye"),
    ("do rupaye ki chai", "2 rupaye ki chai"),
    ("दो सौ", "200"),
    ("ढाई सौ", "250"),
    ("डेढ़ हज़ार", "1500"),
    ("२००", "200"),
    # Everyday words that are NOT numbers here stay as they are.
    ("what do I spend most on?", "what do I spend most on?"),
    ("dost ke saath khana 300", "dost ke saath khana 300"),
    ("chai 20", "chai 20"),
    ("saadhe", "saadhe"),
])
def test_hindi_numbers_to_digits(text, expected):
    assert ai_agent.hindi_numbers_to_digits(text) == expected


# ---------- The code dictionary: time phrases + number words, before the AI ----------

@pytest.mark.parametrize("text, expected_text, expected_phrases", [
    # Time words (Roman and Devanagari, whole word, any case)
    ("aaj", "today", ["today"]),
    ("आज", "today", ["today"]),
    ("kal", "yesterday", ["yesterday"]),
    ("कल", "yesterday", ["yesterday"]),
    ("parso kitna kharcha hua?", "day before yesterday kitna kharcha hua?", ["day before yesterday"]),
    ("Parso", "day before yesterday", ["day before yesterday"]),
    ("परसों कितना खर्च हुआ", "day before yesterday कितना खर्च हुआ", ["day before yesterday"]),
    ("is hafte", "this week", ["this week"]),
    ("इस हफ्ते", "this week", ["this week"]),
    ("pichle hafte", "last week", ["last week"]),
    ("पिछले हफ्ते", "last week", ["last week"]),
    ("is mahine", "this month", ["this month"]),
    ("इस महीने", "this month", ["this month"]),
    ("pichle mahine", "last month", ["last month"]),
    ("पिछले महीने", "last month", ["last month"]),
    ("is saal", "this year", ["this year"]),
    ("इस साल", "this year", ["this year"]),
    # Number words
    ("ek hazaar", "1000", []),
    ("ek rupaye", "1 rupaye", []),
    ("do sau", "200", []),
    ("teen sau", "300", []),
    ("chaar", "4", []),
    ("paanch", "5", []),
    ("das", "10", []),
    ("bees", "20", []),
    ("pachaas", "50", []),
    ("sau", "100", []),
    ("hazaar", "1000", []),
    ("dedh sau", "150", []),
    ("dhai sau", "250", []),
    ("saade teen sau", "350", []),
    # Together
    ("aaj chai pe bees diye", "today chai pe 20 diye", ["today"]),
    ("pichle mahine dedh hazaar kiraya", "last month 1500 kiraya", ["last month"]),
    # Whole words only: "kala" (black) and an English "is" stay as they are.
    ("kala chashma", "kala chashma", []),
    ("it is fine", "it is fine", []),
])
def test_dictionary_replaces_time_and_number_words(text, expected_text, expected_phrases):
    assert ai_agent.apply_hindi_dictionary(text) == (expected_text, expected_phrases)


# ---------- Multi-word number phrases: always plain digits, never "saade 300" ----------

@pytest.mark.parametrize("text, expected", [
    ("saade teen sau", "350"),
    ("saadhe teen sau", "350"),
    ("saade do hazaar", "2500"),
    ("paune do sau", "175"),
    ("sava do sau", "225"),
    ("dedh sau", "150"),
    ("dhai sau", "250"),
    ("dedh hazaar", "1500"),
    ("dhai hazaar", "2500"),
    ("teen sau", "300"),
    ("paanch sau", "500"),
    ("do hazaar", "2000"),
    ("das hazaar", "10000"),
    ("साढ़े तीन सौ", "350"),
    ("पौने दो सौ", "175"),
    ("सवा दो सौ", "225"),
    ("saade teen sau ka khana", "350 ka khana"),
    ("chai saade teen sau ki", "chai 350 ki"),
])
def test_number_phrases_become_plain_digits(text, expected):
    result = ai_agent.hindi_numbers_to_digits(text)
    assert result == expected
    number = result.split()[0] if result[0].isdigit() else result.split()[1]
    assert number.isdigit()                            # no leftover "saade" / "sau" next to it


def test_saade_teen_sau_ka_khana_kal_dictionary():
    assert ai_agent.apply_hindi_dictionary("saade teen sau ka khana kal") == (
        "350 ka khana yesterday", ["yesterday"])


def test_day_before_yesterday_is_a_period():
    today = date(2026, 11, 6)
    assert ai_agent.resolve_period("day before yesterday", today)[:2] == ("2026-11-04", "2026-11-04")
    assert ai_agent.dates_in_message("the chai was day before yesterday", today) == {"2026-11-04"}
    assert ai_agent.summary_args_from("How much did I spend day before yesterday?", CATEGORIES)["period"] \
        == "day before yesterday"


# ---------- Checks on the translation (fake AI translator) ----------

def test_translation_sees_digits_and_returns_english(fake_ai):
    model = fake_ai("How much did I spend on food?")
    assert ai_agent.translate_message("khane pe kitna kharcha hua", CATEGORIES) == (
        "How much did I spend on food?", "hinglish")
    model = fake_ai("Spent 250 on auto today")
    assert ai_agent.translate_message("aaj auto ke dhai sau diye", CATEGORIES)[0] == "Spent 250 on auto today"


@pytest.mark.parametrize("message, bad_translation", [
    ("aaj auto ke do sau diye", "Spent 250 on auto today"),            # changed the amount
    ("aaj auto ke liye diye", "Spent 200 on auto today"),              # added an amount
    ("aaj auto ke do sau diye", "Spent on auto today"),                # dropped the amount
    ("auto ke liye do sau diye", "Spent 200 on auto yesterday"),       # invented a date
    ("kal auto ke do sau diye", "Spent 200 on auto tomorrow"),         # kal = yesterday only
    ("aaj ke liye do sau diye", "Spent 200 on Shopping today"),        # invented a category
    ("parso kitna kharcha hua?", "How much did I spend on petrol?"),   # the real bug: parso != petrol
    ("parso kitna kharcha hua?", "How much did I spend yesterday?"),   # lost "day before yesterday"
    ("kal kitna kharcha hua?", "How much did I spend day before yesterday?"),  # kal = yesterday
    ("is mahine kitna kharcha hua?", "How much did I spend?"),         # dropped "this month"
    ("khane pe kitna kharcha hua", "खाने पर कितना खर्च हुआ"),           # not English
    ("khane pe kitna kharcha hua", ""),                                # nothing
])
def test_untrustworthy_translation_is_rejected(fake_ai, message, bad_translation):
    fake_ai(bad_translation, bad_translation)          # a clear expense gets one retry: also bad
    with pytest.raises(ai_agent.MessageRejected, match="try in English"):
        ai_agent.translate_message(message, CATEGORIES)


@contextmanager
def _ai_log(caplog):
    """The [AI] logger is silenced in tests (conftest) and doesn't propagate: turn it on and
    attach caplog's handler to it directly, for this block only."""
    was_disabled = ai_agent.AI_LOG.disabled
    ai_agent.AI_LOG.disabled = False
    ai_agent.AI_LOG.addHandler(caplog.handler)
    try:
        yield
    finally:
        ai_agent.AI_LOG.removeHandler(caplog.handler)
        ai_agent.AI_LOG.disabled = was_disabled


def test_clear_expense_gets_one_retry_with_simpler_prompt(fake_ai, caplog):
    # The real bug: the translator turned the statement into a question and lost the 350.
    model = fake_ai("How much did I spend on food yesterday?", "Spent 350 on khana yesterday")
    with _ai_log(caplog):
        english, _ = ai_agent.translate_message("saade teen sau ka khana kal", CATEGORIES)
    assert english == "Spent 350 on khana yesterday"
    assert "raw reply  'How much did I spend on food yesterday?'" in caplog.text   # failure logged


def test_no_retry_for_questions(fake_ai):
    # A question isn't a clear expense: the second (passing) reply must never be used.
    fake_ai("Spent 999 on food yesterday", "Did I spend 350 on khana yesterday?")
    with pytest.raises(ai_agent.MessageRejected):
        ai_agent.translate_message("kya kal khane pe 350 diye", CATEGORIES)


def test_no_retry_without_amount_item_and_date(fake_ai):
    fake_ai("Spent 200 today", "Spent 200 on something")
    with pytest.raises(ai_agent.MessageRejected):
        ai_agent.translate_message("auto ke do sau diye", CATEGORIES)       # no date


def test_raw_reply_logged_when_translation_fails(fake_ai, caplog):
    fake_ai("खाने पर कितना खर्च हुआ\nextra line")
    with _ai_log(caplog), pytest.raises(ai_agent.MessageRejected):
        ai_agent.translate_message("khane pe kitna kharcha hua", CATEGORIES)
    assert "REJECTED: not English" in caplog.text
    assert "extra line" in caplog.text                                   # the full raw reply


# ---------- Through the app: everything after translation is unchanged ----------

def test_saade_teen_sau_ka_khana_kal_becomes_350_food_yesterday(client, user, fake_ai):
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    fake_ai("How much did I spend on food yesterday?",                  # 1st try: lost the 350
            "Spent 350 on khana yesterday",                             # retry: passes the checks
            tool_call("add_expense_tool", amount=350, category="Food", date=yesterday, description="khana"),
            "Ready for review.")
    status, data = ask_ai(client, "saade teen sau ka khana kal")
    assert status == 200
    assert data["understood_as"] == '🌐 Understood as: "Spent 350 on khana yesterday"'
    assert [c["type"] for c in data["cards"]] == ["add"]
    assert data["cards"][0]["rows"][0]["amount"] == "₹350"
    assert count_expenses(user) == 0                                    # still needs Confirm


def test_hinglish_expense_gets_preview_and_understood_as(client, user, fake_ai):
    today = date.today().isoformat()
    fake_ai("Spent 200 on auto today",                                  # the translator
            tool_call("add_expense_tool", amount=200, category="Transport", date=today, description="auto"),
            "Ready for review.")
    status, data = ask_ai(client, "aaj auto ke do sau diye")
    assert status == 200
    assert data["understood_as"] == '🌐 Understood as: "Spent 200 on auto today"'
    assert [c["type"] for c in data["cards"]] == ["add"]
    assert data["cards"][0]["rows"][0]["amount"] == "₹200"
    assert count_expenses(user) == 0                                    # still needs Confirm

    history = client.get("/ai/history").get_json()["messages"]
    assert history[0] == {"role": "user", "text": "aaj auto ke do sau diye", "kind": ""}
    assert history[1]["kind"] == "translation"                          # shown again after a refresh


def test_parso_question_answers_for_the_day_before_yesterday(client, user, fake_ai):
    today = date.today()
    two_days_ago = (today - timedelta(days=2)).isoformat()
    db.add_expense(user, 300, "Food", two_days_ago, "dinner")
    db.add_expense(user, 999, "Food", today.isoformat(), "lunch")          # not counted
    fake_ai("How much did I spend day before yesterday?",
            tool_call("get_spending_summary", period="day before yesterday"), "Done.")
    status, data = ask_ai(client, "parso kitna kharcha hua?")
    assert status == 200
    assert data["understood_as"] == '🌐 Understood as: "How much did I spend day before yesterday?"'
    assert data["reply"].startswith("You spent ₹300 day before yesterday across 1 expense")


def test_english_message_has_no_understood_as(client, user, fake_ai):
    fake_ai(tool_call("get_spending_summary"), "Done.")
    _, data = ask_ai(client, "how much did I spend in total?")
    assert data["understood_as"] is None


def test_devanagari_question_answered_from_the_database(client, user, fake_ai):
    db.add_expense(user, 250, "Food", "2026-11-05", "lunch")
    fake_ai("How much did I spend on food?",
            tool_call("get_spending_summary", category="Food"), "You spent ₹9,999.")   # AI lies
    status, data = ask_ai(client, "खाने पर कितना खर्च हुआ")
    assert status == 200
    assert data["reply"].startswith("You spent ₹250 on Food in total")


def test_failed_translation_asks_instead_of_guessing(client, user, fake_ai):
    fake_ai("Spent 2000 on auto today", "Spent 2000 on auto today")    # "do sau" is 200, not 2000 (twice)
    status, data = ask_ai(client, "aaj auto ke do sau diye")
    assert status == 400
    assert data["error"] == "Sorry, I couldn't understand that — try in English?"
    assert count_expenses(user) == 0


def test_negative_amount_after_translation_is_rejected(client, user, fake_ai):
    fake_ai("Spent -20 on chai")
    status, data = ask_ai(client, "minus bees rupaye chai pe diye")
    assert status == 400
    assert "Amounts must be greater than 0" in data["error"]


def test_hindi_question_stays_read_only_even_if_translated_as_a_request(client, user, fake_ai):
    today = date.today().isoformat()
    fake_ai("Spent 200 on food",                                        # bad translation: not a question
            tool_call("add_expense_tool", amount=200, category="Food", date=today, description="food"),
            "OK.")
    status, data = ask_ai(client, "kya maine khane pe 200 diye")
    assert status == 200
    assert data["cards"] == []                                          # the write tool refused


def test_translation_with_ollama_down_is_a_clear_503(client, user, ollama_down):
    status, data = ask_ai(client, "khane pe kitna kharcha hua")
    assert status == 503
    assert "Ollama" in data["detail"]


# ---------- "Reply in Hindi" ----------

def ask_in_hindi(client, message):
    response = client.post("/ai/assistant", json={"message": message, "reply_lang": "hinglish"})
    return response.status_code, response.get_json()


def test_reply_in_hindi_keeps_amounts_exact(client, user, fake_ai):
    db.add_expense(user, 20, "Food", "2026-11-05", "chai")
    fake_ai(tool_call("get_spending_summary"), "Lots!",
            "Aapne total ₹20 kharch kiye, 1 expense mein.")             # the reply translator
    _, data = ask_in_hindi(client, "how much did I spend in total?")
    assert data["reply"] == "Aapne total ₹20 kharch kiye, 1 expense mein."
    history = client.get("/ai/history").get_json()["messages"]
    assert history[-1]["text"] == "Aapne total ₹20 kharch kiye, 1 expense mein."


@pytest.mark.parametrize("bad_hinglish", [
    "Aapne total ₹25 kharch kiye, 1 expense mein.",                     # amount changed
    "Aapne total ₹20 kharch kiye.",                                     # count dropped
    "आपने ₹20 खर्च किए, 1 expense",                                      # Devanagari, not Hinglish
])
def test_reply_in_hindi_falls_back_to_english_if_numbers_change(client, user, fake_ai, bad_hinglish):
    db.add_expense(user, 20, "Food", "2026-11-05", "chai")
    fake_ai(tool_call("get_spending_summary"), "Lots!", bad_hinglish)
    _, data = ask_in_hindi(client, "how much did I spend in total?")
    assert data["reply"].startswith("You spent ₹20 in total across 1 expense")


def test_reply_in_hindi_off_by_default(client, user, fake_ai):
    db.add_expense(user, 20, "Food", "2026-11-05", "chai")
    fake_ai(tool_call("get_spending_summary"), "Lots!", "SHOULD NOT BE USED")
    _, data = ask_ai(client, "how much did I spend in total?")
    assert data["reply"].startswith("You spent ₹20 in total")
