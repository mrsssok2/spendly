"""AI assistant safety tests. The real Ollama model is NEVER called: a scripted fake AI
(see `fake_ai` in conftest.py) plays the model, so these tests are fast and always give the
same result. They check OUR code: whatever the AI asks for, the safety rules must hold."""
from datetime import date

import pytest

import ai_agent
import app as spendly
import database.db as db
from conftest import ask_ai, count_expenses, login, tool_call


def add_expense(user_id, amount, category, date, description):
    db.add_expense(user_id, amount, category, date, description)


def confirm(client, card):
    response = client.post("/ai/confirm", json={"action_id": card["id"]})
    return response.status_code, response.get_json()


# ---------- Negative and zero amounts: rejected before the AI is called ----------

@pytest.mark.parametrize("message", [
    "spend -20 on dolo tablet", "spent - 20 on chai", "₹-50 on auto", "Rs.-50 auto",
    "change auto to -280", "spent 0 on parking", "chai 0.00", "zero rupees on bus",
])
def test_negative_and_zero_amounts_rejected_before_ai(client, user, ai_must_not_be_called, message):
    status, data = ask_ai(client, message)
    assert status == 400
    assert "Amounts must be greater than 0" in data["error"]
    assert ai_must_not_be_called == []          # the AI never saw the message
    assert count_expenses(user) == 0


@pytest.mark.parametrize("message", [
    "spent 10 on chai", "spent 250 on auto on 2026-11-05", "spent 120 on Dolo-650", "chai 20 on 05 Nov",
])
def test_normal_amounts_and_dates_are_not_mistaken_for_negative_or_zero(client, user, fake_ai, message):
    # Two replies: a "spent ..." message with no tool call gets one retry from the app.
    fake_ai("OK.", "OK.")
    status, _ = ask_ai(client, message)
    assert status == 200


def test_tool_validation_also_rejects_zero_and_negative(user):
    actions = []
    tools = {t.name: t for t in ai_agent._make_tools(user, actions, spendly.validate_expense_form,
                                                    spendly.CATEGORIES, "spent on chai")}
    for amount in (0, -20):
        result = tools["add_expense_tool"].invoke(
            {"amount": amount, "category": "Food", "date": "2026-11-05", "description": "chai"})
        assert "greater than 0" in result
    assert actions == []


# ---------- Tools only ever use the logged-in user's id ----------

def test_no_tool_lets_the_ai_choose_a_user_id(user):
    tools = ai_agent._make_tools(user, [], spendly.validate_expense_form, spendly.CATEGORIES)
    assert len(tools) == 12   # includes run_sql_query, get_forecast and get_anomalies
    for t in tools:
        assert "user_id" not in t.args, t.name
        assert not any("id" == name or name.endswith("_id") for name in t.args), t.name


def test_delete_only_finds_the_logged_in_users_expense(client, user, other_user, fake_ai):
    add_expense(user, 20, "Food", "2026-11-05", "chai")
    add_expense(other_user, 999, "Food", "2026-11-05", "chai")      # Ravi's chai: same words
    fake_ai(tool_call("delete_expense_tool", match_description="chai"), "Ready for review.")

    status, data = ask_ai(client, "delete chai")
    assert status == 200
    assert [c["type"] for c in data["cards"]] == ["delete"]        # one match, not a "which one?" list
    assert data["cards"][0]["row"]["amount"] == "₹20"              # Asha's, not Ravi's ₹999

    assert confirm(client, data["cards"][0])[0] == 200
    assert count_expenses(user) == 0
    assert count_expenses(other_user) == 1                        # Ravi's chai untouched


def test_questions_only_see_the_logged_in_users_data(client, user, other_user, fake_ai):
    add_expense(user, 20, "Food", "2026-11-05", "chai")
    add_expense(other_user, 999, "Food", "2026-11-05", "chai")

    fake_ai(tool_call("find_expenses", description="chai"), "You spent ₹999 on chai.")  # AI lies
    _, data = ask_ai(client, "what did I spend on chai?")
    assert "₹20" in data["reply"] and "999" not in data["reply"]   # database answer shown

    fake_ai(tool_call("get_spending_summary"), "Lots!")
    _, data = ask_ai(client, "how much did I spend in total?")
    assert data["reply"].startswith("You spent ₹20 in total across 1 expense")


def test_route_passes_the_session_user_id_to_the_ai(client, user, other_user, monkeypatch):
    seen = []
    monkeypatch.setattr(spendly, "run_expense_assistant",
                        lambda message, user_id, *rest, **kwargs: seen.append(user_id) or ("ok", []))
    client.post("/ai/assistant", json={"message": "how much did I spend?", "user_id": other_user})
    assert seen == [user]                                          # extra JSON field ignored


# ---------- Add / edit / delete never save without Confirm ----------

def test_add_needs_confirm(client, user, fake_ai):
    fake_ai(tool_call("add_expense_tool", amount=250, category="Transport", date="2026-11-05",
                      description="auto"), "Ready for review.")
    status, data = ask_ai(client, "spent 250 on auto today")
    assert status == 200 and data["cards"][0]["type"] == "add"
    assert count_expenses(user) == 0                               # nothing saved yet

    status, result = confirm(client, data["cards"][0])
    assert status == 200
    assert result["message"] == "✅ Added ₹250 · Transport · 05 Nov — auto"
    assert count_expenses(user) == 1


def test_edit_needs_confirm_and_shows_before_and_after(client, user, fake_ai):
    add_expense(user, 250, "Transport", "2026-11-05", "auto")
    fake_ai(tool_call("update_expense_tool", match_description="auto", new_amount=280), "Ready.")
    _, data = ask_ai(client, "change auto to 280")
    card = data["cards"][0]
    assert card["type"] == "edit"
    assert card["changes"] == [{"field": "Amount", "before": "₹250", "after": "₹280"}]
    assert db.get_expenses_for_user(user)[0]["amount"] == 250.0    # unchanged before Confirm

    status, result = confirm(client, card)
    assert status == 200 and result["message"] == "✏️ Updated auto: ₹250 → ₹280"
    assert db.get_expenses_for_user(user)[0]["amount"] == 280.0


def test_delete_needs_confirm(client, user, fake_ai):
    add_expense(user, 100, "Health", "2026-11-05", "paracetamol tablet")
    fake_ai(tool_call("delete_expense_tool", match_description="paracetamol"),
            "The expense has been deleted.")                       # AI falsely claims it's done
    _, data = ask_ai(client, "delete the paracetamol expense")
    assert data["cards"][0]["type"] == "delete"
    assert "has been deleted" not in data["reply"]                 # false claim replaced
    assert count_expenses(user) == 1                               # still there

    status, result = confirm(client, data["cards"][0])
    assert status == 200
    assert result["message"] == "🗑️ Deleted ₹100 · Health · 05 Nov — paracetamol tablet"
    assert count_expenses(user) == 0


def test_several_matches_show_a_list_and_change_nothing(client, user, fake_ai):
    add_expense(user, 20, "Food", "2026-11-05", "chai")
    add_expense(user, 30, "Food", "2026-11-04", "chai")
    fake_ai(tool_call("delete_expense_tool", match_description="chai"), "Which one?")
    _, data = ask_ai(client, "delete chai")
    card = data["cards"][0]
    assert card["type"] == "choose" and len(card["options"]) == 2
    assert count_expenses(user) == 2
    assert confirm(client, card)[0] == 400                         # a list can't be confirmed


def test_confirm_cannot_be_forged_or_repeated(client, user, fake_ai):
    fake_ai(tool_call("add_expense_tool", amount=50, category="Food", date="2026-11-05",
                      description="samosa"), "Ready.")
    _, data = ask_ai(client, "spent 50 on samosa")
    assert client.post("/ai/confirm", json={"action_id": "made-up"}).status_code == 400
    assert confirm(client, data["cards"][0])[0] == 200
    assert confirm(client, data["cards"][0])[0] == 400             # second click does nothing
    assert count_expenses(user) == 1


def test_questions_never_change_anything(client, user, fake_ai):
    fake_ai(tool_call("add_expense_tool", amount=500, category="Food", date="2026-11-05",
                      description="sneaky"), "Done.")
    _, data = ask_ai(client, "how much did I spend today?")
    assert data["cards"] == []
    assert count_expenses(user) == 0


# ---------- Cancel changes nothing ----------

def test_cancel_changes_nothing(client, user, fake_ai):
    add_expense(user, 100, "Health", "2026-11-05", "paracetamol tablet")
    fake_ai(tool_call("delete_expense_tool", match_description="paracetamol"), "Ready.")
    _, data = ask_ai(client, "delete the paracetamol expense")
    card = data["cards"][0]

    response = client.post("/ai/cancel", json={"action_id": card["id"]})
    assert response.get_json()["message"] == "Cancelled — nothing changed"
    assert count_expenses(user) == 1
    assert confirm(client, card)[0] == 400                         # can't confirm after cancel
    assert count_expenses(user) == 1


# ---------- Chat memory: separate per user, cleared by New chat and logout ----------

def test_each_user_has_their_own_chat(client, user, other_user, fake_ai):
    fake_ai("Hello Asha!")
    ask_ai(client, "spendly check from Asha")

    ravi = spendly.app.test_client()
    login(ravi, "ravi@example.com")
    fake_ai("Hello Ravi!")
    ask_ai(ravi, "spendly check from Ravi")

    asha_chat = [m["text"] for m in client.get("/ai/history").get_json()["messages"]]
    ravi_chat = [m["text"] for m in ravi.get("/ai/history").get_json()["messages"]]
    assert asha_chat == ["spendly check from Asha", "Hello Asha!"]
    assert ravi_chat == ["spendly check from Ravi", "Hello Ravi!"]
    assert ai_agent.thread_id_for(user) != ai_agent.thread_id_for(other_user)


def test_new_chat_clears_only_that_users_chat(client, user, other_user, fake_ai):
    fake_ai("Hello Asha!")
    ask_ai(client, "spendly check from Asha")
    ravi = spendly.app.test_client()
    login(ravi, "ravi@example.com")
    fake_ai("Hello Ravi!")
    ask_ai(ravi, "spendly check from Ravi")

    assert client.post("/ai/new-chat").status_code == 200
    assert client.get("/ai/history").get_json()["messages"] == []
    assert len(ravi.get("/ai/history").get_json()["messages"]) == 2


def test_new_chat_also_drops_waiting_previews(client, user, fake_ai):
    fake_ai(tool_call("add_expense_tool", amount=50, category="Food", date="2026-11-05",
                      description="samosa"), "Ready.")
    _, data = ask_ai(client, "spent 50 on samosa")
    client.post("/ai/new-chat")
    assert confirm(client, data["cards"][0])[0] == 400
    assert count_expenses(user) == 0


def test_logout_clears_the_chat(client, user, fake_ai):
    fake_ai("Hello Asha!")
    ask_ai(client, "spendly check from Asha")
    client.get("/logout")
    login(client, "asha@example.com")
    assert client.get("/ai/history").get_json()["messages"] == []


def test_chat_history_survives_a_page_refresh_with_waiting_preview(client, user, fake_ai):
    fake_ai(tool_call("add_expense_tool", amount=50, category="Food", date="2026-11-05",
                      description="samosa"), "Ready for review.")
    ask_ai(client, "spent 50 on samosa")
    history = client.get("/ai/history").get_json()                 # what a refresh loads
    assert history["messages"][0]["text"] == "spent 50 on samosa"
    assert [c["type"] for c in history["cards"]] == ["add"]


# ---------- Ollama unavailable ----------

def test_ollama_down_shows_unavailable_and_app_still_works(client, user, ollama_down):
    status, data = ask_ai(client, "spent 250 on auto")
    assert status == 503
    assert data["error"] == "AI assistant unavailable"
    assert "Can't reach Ollama" in data["detail"]

    assert client.get("/profile").status_code == 200               # page still works
    assert client.get("/ai/history").status_code == 200            # chat box can still load
    response = client.post("/expenses/add", data={"amount": "250", "category": "Transport",
                                                  "date": "2026-11-05", "description": "auto"})
    assert response.status_code == 302                             # normal form still works
    assert count_expenses(user) == 1


# ---------- Write-tool loops: one preview, never "The AI got stuck" ----------
# Regression: real qwen2.5 called add_expense_tool with the same arguments again and again for
# "spent 150 on chai today" until the recursion limit -> 503 "The AI got stuck" for every
# add/edit/delete/income/budget request. loop_guard must stop a repeated write call.

REPEATS = 12   # more model turns than the old recursion limit allowed


@pytest.mark.parametrize("message, call, card_type", [
    ("spent 150 on chai today",
     tool_call("add_expense_tool", amount=150, category="Food", date=str(date.today()),
               description="chai"), "add"),
    ("delete the auto expense",
     tool_call("delete_expense_tool", match_description="auto"), "delete"),
    ("got 50000 salary today",
     tool_call("add_income_tool", amount=50000, source="Salary", date=str(date.today()),
               description=""), "add_income"),
    ("set my food budget to 3000",
     tool_call("set_budget_tool", amount=3000, category="Food"), "budget"),
])
def test_repeated_write_call_finishes_with_one_preview(client, user, fake_ai, message, call, card_type):
    add_expense(user, 250, "Transport", str(date.today()), "auto")
    # A NEW message each time, like the real model (one shared object keeps one message id).
    fake_ai(*[call.model_copy(deep=True, update={"id": f"repeat-{i}"}) for i in range(REPEATS)])
    status, data = ask_ai(client, message)
    assert status == 200, data
    assert len(data["cards"]) == 1                     # the repeat didn't prepare it twice
    assert data["cards"][0]["type"] == card_type
    assert len(data["cards"][0].get("rows", [None])) == 1
    assert "review" in data["reply"].lower()           # code wrote the reply (AI's was empty)
    assert count_expenses(user) == 1                   # nothing saved without Confirm


def test_two_different_expenses_still_get_two_previews(client, user, fake_ai):
    today = str(date.today())
    fake_ai(tool_call("add_expense_tool", amount=150, category="Food", date=today, description="chai"),
            tool_call("add_expense_tool", amount=250, category="Transport", date=today,
                      description="auto"),
            "Ready for review.")
    status, data = ask_ai(client, "spent 150 on chai and 250 on auto today")
    assert status == 200
    assert [c["type"] for c in data["cards"]] == ["add"]           # adds share one card
    assert [r["description"] for r in data["cards"][0]["rows"]] == ["chai", "auto"]
    assert count_expenses(user) == 0


def test_second_delete_of_same_expense_with_other_arguments_is_not_prepared(client, user, fake_ai):
    # Real qwen2.5: delete_expense_tool(match_description="shirt"), then again with
    # match_category added -> two delete previews for one shirt.
    add_expense(user, 1000, "Shopping", "2026-11-03", "shirt")
    fake_ai(tool_call("delete_expense_tool", match_description="shirt"),
            tool_call("delete_expense_tool", match_description="shirt", match_category="Shopping"),
            "Ready for review.")
    status, data = ask_ai(client, "delete the shirt expense")
    assert status == 200
    assert [c["type"] for c in data["cards"]] == ["delete"]
    assert count_expenses(user) == 1


def test_second_edit_of_same_expense_keeps_the_first(client, user, fake_ai):
    # Real qwen2.5: a correct edit (amount 90), then a second edit of the same chai with an
    # invented description -> the second must not become another preview.
    add_expense(user, 80, "Food", str(date.today()), "chai")
    fake_ai(tool_call("update_expense_tool", match_description="chai", new_amount=90),
            tool_call("update_expense_tool", match_description="chai", new_description="chai tea"),
            "Ready for review.")
    status, data = ask_ai(client, "change the chai expense to 90")
    assert status == 200
    assert len(data["cards"]) == 1
    assert data["cards"][0]["changes"] == [{"field": "Amount", "before": "₹80", "after": "₹90"}]


def test_repeated_income_with_other_description_is_one_preview(client, user, fake_ai):
    today = str(date.today())
    fake_ai(tool_call("add_income_tool", amount=40000, source="Salary", date=today, description=""),
            tool_call("add_income_tool", amount=40000, source="Salary", date=today, description="salary"),
            "Ready for review.")
    status, data = ask_ai(client, "got salary 40000 today")
    assert status == 200
    assert [c["type"] for c in data["cards"]] == ["add_income"]
    assert len(data["cards"][0]["rows"]) == 1


def test_deleting_two_different_expenses_still_works(client, user, fake_ai):
    add_expense(user, 80, "Food", str(date.today()), "chai")
    add_expense(user, 250, "Transport", str(date.today()), "auto")
    fake_ai(tool_call("delete_expense_tool", match_description="chai"),
            tool_call("delete_expense_tool", match_description="auto"),
            "Ready for review.")
    status, data = ask_ai(client, "delete the chai and the auto expenses")
    assert status == 200
    assert sorted(c["row"]["description"] for c in data["cards"]) == ["auto", "chai"]


# ---------- Corrections ("X was actually Y") EDIT an existing expense, never add ----------
# Regression: real qwen2.5 turned "the movie was actually 350" into a NEW ₹350 movie expense.

CATEGORY_NAMES = ["Food", "Transport", "Bills", "Health", "Entertainment", "Shopping", "Other"]
TODAY = date(2026, 11, 7)


@pytest.mark.parametrize("message, expected", [
    ("the movie was actually 350", {"match_description": "movie", "new_amount": 350.0}),
    ("The movie was actually ₹350.", {"match_description": "movie", "new_amount": 350.0}),
    ("the movie should be 350", {"match_description": "movie", "new_amount": 350.0}),
    ("the movie should have been 350 rupees", {"match_description": "movie", "new_amount": 350.0}),
    ("correct the movie to 350", {"match_description": "movie", "new_amount": 350.0}),
    ("correct movie expense to 350", {"match_description": "movie", "new_amount": 350.0}),
    ("actually the movie was 350", {"match_description": "movie", "new_amount": 350.0}),
    ("the movie was not 300, it was 350",
     {"match_description": "movie", "match_amount": 300.0, "new_amount": 350.0}),
    ("movie 300 tha nahi, 350 tha",
     {"match_description": "movie", "match_amount": 300.0, "new_amount": 350.0}),
    ("movie ka 300 nahi 350 tha",
     {"match_description": "movie", "match_amount": 300.0, "new_amount": 350.0}),
    ("the food expense should be 200", {"match_category": "Food", "new_amount": 200.0}),
    ("the movie should be shopping", {"match_description": "movie", "new_category": "Shopping"}),
    ("the chai was actually yesterday", {"match_description": "chai", "new_date": "2026-11-06"}),
])
def test_correction_is_read_in_code(message, expected):
    assert ai_agent.correction_from_message(message, message, TODAY, CATEGORY_NAMES) == expected


@pytest.mark.parametrize("message", [
    "should the movie be 350?", "was the movie 350?",        # questions are read-only
    "it was actually 350",                                    # pronoun: left to the AI + chat
    "spent 350 on movie today", "movie 350 today",            # new expenses
    "the movie was 350", "change the chai expense to 90", "delete the auto expense",
    "mom gave me 2000 yesterday",
])
def test_not_a_correction(message):
    assert ai_agent.correction_from_message(message, message, TODAY, CATEGORY_NAMES) is None


def movie_amount(user_id):
    return [e["amount"] for e in db.get_expenses_for_user(user_id) if e["description"] == "movie"]


@pytest.mark.parametrize("message", [
    "the movie was actually 350", "the movie should be 350", "correct the movie to 350",
    "the movie was not 300, it was 350",
])
def test_correction_when_ai_adds_becomes_an_edit(client, user, fake_ai, message):
    add_expense(user, 300, "Entertainment", "2026-11-03", "movie")
    # The AI wrongly tries to ADD a new ₹350 movie, then claims it did.
    fake_ai(tool_call("add_expense_tool", amount=350, category="Entertainment", date=str(date.today()),
                      description="movie"),
            "I have added the movie expense of ₹350.")
    status, data = ask_ai(client, message)
    assert status == 200
    assert [c["type"] for c in data["cards"]] == ["edit"]          # never an add
    assert data["cards"][0]["changes"] == [{"field": "Amount", "before": "₹300", "after": "₹350"}]
    assert "review" in data["reply"].lower()
    assert movie_amount(user) == [300.0]                           # nothing saved before Confirm

    status, result = confirm(client, data["cards"][0])
    assert status == 200 and movie_amount(user) == [350.0] and count_expenses(user) == 1


def test_correction_when_ai_uses_no_tool_becomes_an_edit(client, user, fake_ai):
    add_expense(user, 300, "Entertainment", "2026-11-03", "movie")
    fake_ai("Which expense do you mean?")
    status, data = ask_ai(client, "the movie was actually 350")
    assert status == 200 and [c["type"] for c in data["cards"]] == ["edit"]


def test_correction_when_ai_edits_is_not_prepared_twice(client, user, fake_ai):
    add_expense(user, 300, "Entertainment", "2026-11-03", "movie")
    fake_ai(tool_call("update_expense_tool", match_description="movie", new_amount=350), "Ready for review.")
    status, data = ask_ai(client, "the movie was actually 350")
    assert status == 200 and [c["type"] for c in data["cards"]] == ["edit"]


def test_correction_with_no_matching_expense_says_so(client, user, fake_ai):
    fake_ai(tool_call("add_expense_tool", amount=350, category="Entertainment", date=str(date.today()),
                      description="movie"), "Added.")
    status, data = ask_ai(client, "the movie was actually 350")
    assert status == 200 and data["cards"] == []
    assert data["reply"] == "No matching expense found. Nothing was changed."
    assert count_expenses(user) == 0


def test_correction_with_two_matches_asks_to_pick(client, user, fake_ai):
    add_expense(user, 300, "Entertainment", "2026-11-03", "movie")
    add_expense(user, 200, "Entertainment", "2026-11-01", "movie")
    fake_ai("Done.")
    status, data = ask_ai(client, "the movie should be 350")
    assert status == 200 and [c["type"] for c in data["cards"]] == ["choose"]


def test_hinglish_correction_becomes_an_edit(client, user, fake_ai):
    # Real qwen2.5 translated "movie 300 tha nahi, 350 tha" as "How much did the movie cost?"
    # (rejected). A correction read in code gets its English from code: no translator call.
    add_expense(user, 300, "Entertainment", "2026-11-03", "movie")
    fake_ai(tool_call("add_expense_tool", amount=350, category="Entertainment", date=str(date.today()),
                      description="movie"),
            "Added.")
    status, data = ask_ai(client, "movie 300 tha nahi, 350 tha")
    assert status == 200, data
    assert [c["type"] for c in data["cards"]] == ["edit"]
    assert data["cards"][0]["changes"] == [{"field": "Amount", "before": "₹300", "after": "₹350"}]


def test_correction_question_never_writes(client, user, fake_ai):
    add_expense(user, 300, "Entertainment", "2026-11-03", "movie")
    fake_ai(tool_call("find_expenses", description="movie"), "Your movie was ₹300.")
    status, data = ask_ai(client, "should the movie be 350?")
    assert status == 200 and data["cards"] == []


@pytest.mark.parametrize("args", [
    {"match_description": "movie", "new_amount": 350.0},
    {"match_description": "movie", "match_amount": 300.0, "new_amount": 350.0},
    {"match_category": "Food", "new_amount": 200.0},
    {"match_description": "movie", "new_category": "Shopping"},
    {"match_description": "chai", "new_date": "2026-11-06"},
])
def test_correction_sentence_parses_back_to_the_same_arguments(args):
    sentence = ai_agent.correction_sentence(args)
    assert ai_agent.correction_from_message(sentence, sentence, TODAY, CATEGORY_NAMES) == args
