"""Registration, login, logout and pages that need a login."""
import database.db as db
from conftest import login


def register(client, name="Asha", email="asha@example.com", password="secret123", confirm=None):
    return client.post("/register", data={
        "name": name, "email": email, "password": password,
        "confirm_password": password if confirm is None else confirm,
    })


# ---------- Registration ----------

def test_register_creates_account_and_redirects_to_login(client):
    response = register(client)
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login")
    user = db.get_user_by_email("asha@example.com")
    assert user is not None
    assert user["password_hash"] != "secret123"  # stored hashed, never as plain text


def test_register_rejects_duplicate_email(client):
    register(client)
    response = register(client, name="Someone else")
    assert response.status_code == 400
    assert b"already registered" in response.data


def test_register_rejects_mismatched_passwords(client):
    response = register(client, confirm="different1")
    assert response.status_code == 400
    assert b"Passwords do not match" in response.data
    assert db.get_user_by_email("asha@example.com") is None


def test_register_rejects_short_password_and_bad_email(client):
    response = register(client, email="not-an-email", password="123")
    assert response.status_code == 400
    assert b"valid email" in response.data
    assert b"at least 6 characters" in response.data


# ---------- Login / logout ----------

def test_login_with_correct_password(client, make_user):
    user_id = make_user("asha@example.com")
    response = login(client, "asha@example.com")
    assert response.status_code == 302
    with client.session_transaction() as session:
        assert session["user_id"] == user_id


def test_login_with_wrong_password_fails(client, make_user):
    make_user("asha@example.com")
    response = login(client, "asha@example.com", password="wrong-password")
    assert response.status_code == 401
    assert b"Invalid email or password" in response.data
    with client.session_transaction() as session:
        assert "user_id" not in session


def test_login_with_unknown_email_gives_same_message(client):
    response = login(client, "nobody@example.com")
    assert response.status_code == 401
    assert b"Invalid email or password" in response.data


def test_logout_clears_the_session(client, user):
    client.get("/logout")
    with client.session_transaction() as session:
        assert "user_id" not in session
    assert client.get("/profile").status_code == 302  # back to needing a login


# ---------- Pages that need a login ----------

def test_profile_requires_login(client):
    response = client.get("/profile")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_add_expense_page_requires_login(client):
    assert "/login" in client.get("/expenses/add").headers["Location"]
    response = client.post("/expenses/add", data={"amount": "10", "category": "Food",
                                                  "date": "2026-11-05", "description": "x"})
    assert "/login" in response.headers["Location"]


def test_ai_routes_require_login(client):
    assert client.post("/ai/assistant", json={"message": "hi"}).status_code == 401
    assert client.get("/ai/history").status_code == 401
    assert client.post("/ai/confirm", json={"action_id": "x"}).status_code == 401


def test_profile_works_when_logged_in(client, user):
    response = client.get("/profile")
    assert response.status_code == 200
    assert b"Asha" in response.data
