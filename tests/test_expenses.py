"""Adding, editing, deleting and filtering expenses, and keeping users' data apart."""
import database.db as db
from conftest import count_expenses


def add(client, amount="250", category="Transport", date="2026-11-05", description="auto"):
    return client.post("/expenses/add", data={
        "amount": amount, "category": category, "date": date, "description": description})


def only_expense(user_id):
    rows = db.get_expenses_for_user(user_id)
    assert len(rows) == 1
    return rows[0]


# ---------- Add ----------

def test_add_expense(client, user):
    response = add(client)
    assert response.status_code == 302
    expense = only_expense(user)
    assert (expense["amount"], expense["category"], expense["date"], expense["description"]) == \
           (250.0, "Transport", "2026-11-05", "auto")
    assert b"auto" in client.get("/profile").data


def test_add_expense_rejects_bad_amounts(client, user):
    for bad in ["0", "-5", "abc", "", "nan", "inf"]:
        response = add(client, amount=bad)
        assert response.status_code == 400, bad
    assert count_expenses(user) == 0


def test_add_expense_rejects_bad_category_and_date(client, user):
    assert add(client, category="Crypto").status_code == 400
    assert add(client, date="2026-02-30").status_code == 400
    assert add(client, date="yesterday").status_code == 400
    assert count_expenses(user) == 0


def test_add_expense_rejects_too_long_description(client, user):
    assert add(client, description="x" * 201).status_code == 400
    assert count_expenses(user) == 0


def test_user_id_comes_from_session_not_the_form(client, user, other_user):
    # A sneaky extra form field must be ignored.
    client.post("/expenses/add", data={"amount": "99", "category": "Food", "date": "2026-11-05",
                                       "description": "sneaky", "user_id": str(other_user)})
    assert count_expenses(user) == 1
    assert count_expenses(other_user) == 0


# ---------- Edit / delete ----------

def test_edit_expense(client, user):
    add(client)
    expense_id = only_expense(user)["id"]
    response = client.post(f"/expenses/{expense_id}/edit", data={
        "amount": "280", "category": "Transport", "date": "2026-11-04", "description": "auto ride"})
    assert response.status_code == 302
    expense = only_expense(user)
    assert (expense["amount"], expense["date"], expense["description"]) == (280.0, "2026-11-04", "auto ride")


def test_edit_expense_with_invalid_values_changes_nothing(client, user):
    add(client)
    expense_id = only_expense(user)["id"]
    response = client.post(f"/expenses/{expense_id}/edit", data={
        "amount": "-1", "category": "Transport", "date": "2026-11-05", "description": "auto"})
    assert response.status_code == 400
    assert only_expense(user)["amount"] == 250.0


def test_delete_expense(client, user):
    add(client)
    expense_id = only_expense(user)["id"]
    assert client.post(f"/expenses/{expense_id}/delete").status_code == 302
    assert count_expenses(user) == 0


# ---------- Another user's expense: always 404, never changed ----------

def test_cannot_view_edit_or_delete_another_users_expense(client, user, other_user):
    db.add_expense(other_user, 999, "Food", "2026-11-05", "Ravi's dinner")
    ravi_expense = only_expense(other_user)["id"]

    assert client.get(f"/expenses/{ravi_expense}/edit").status_code == 404
    edit = client.post(f"/expenses/{ravi_expense}/edit", data={
        "amount": "1", "category": "Food", "date": "2026-11-05", "description": "hacked"})
    assert edit.status_code == 404
    assert client.post(f"/expenses/{ravi_expense}/delete").status_code == 404

    expense = only_expense(other_user)
    assert (expense["amount"], expense["description"]) == (999.0, "Ravi's dinner")
    assert b"Ravi's dinner" not in client.get("/profile").data


def test_missing_expense_is_404(client, user):
    assert client.get("/expenses/12345/edit").status_code == 404
    assert client.post("/expenses/12345/delete").status_code == 404


# ---------- Date filter ----------

def test_date_filter_shows_only_expenses_in_range(client, user):
    add(client, amount="100", date="2026-10-05", description="october-thing")
    add(client, amount="200", date="2026-11-05", description="november-thing")

    page = client.get("/profile?date_from=2026-11-01&date_to=2026-11-30").data
    assert b"november-thing" in page
    assert b"october-thing" not in page
    assert "₹200.00".encode() in page

    page = client.get("/profile").data  # no filter = all time
    assert b"november-thing" in page and b"october-thing" in page


def test_invalid_date_filter_shows_error_and_all_expenses(client, user):
    add(client, description="still-shown")
    page = client.get("/profile?date_from=2026-11-30&date_to=2026-11-01").data
    assert b"on or before" in page
    assert b"still-shown" in page

    page = client.get("/profile?date_from=not-a-date").data
    assert b"Please enter valid dates" in page
