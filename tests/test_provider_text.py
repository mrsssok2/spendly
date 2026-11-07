"""Page text about the AI must match the provider in use.

"Local / private / on your computer / no API keys" is only true with Ollama. With Groq the AI
runs in the cloud, so no page may claim it (app.AI_TEXT + the provider_text context processor).
"""
import re
from datetime import date

import pytest

import ai_agent
import app as spendly
import database.db as db

# Claims that are false when the AI runs on Groq.
LOCAL_CLAIMS = re.compile(
    r"privately|private ai|private by design|your (own )?computer|local ai|no api keys?|"
    r"free, private",
    re.IGNORECASE,
)

PUBLIC_PAGES = ["/", "/login", "/register"]
LOGGED_IN_PAGES = ["/", "/profile", "/analytics"]


def _visible_text(html):
    """Page text without <script>/<style> blocks and tags (so CSS/JS words can't match)."""
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    return re.sub(r"<[^>]+>", " ", html)


@pytest.fixture
def use_provider(monkeypatch):
    def _use(name):
        monkeypatch.setattr(ai_agent, "PROVIDER", name)
    return _use


@pytest.fixture
def user_with_expense(user):
    """Asha, logged in, with one expense (the analytics insights card needs data)."""
    db.add_expense(user, 120, "Food", date.today().isoformat(), "Chai and snacks")
    return user


def _get(client, path):
    response = client.get(path)
    assert response.status_code == 200, path
    return _visible_text(response.get_data(as_text=True))


def test_every_provider_has_the_same_text_keys():
    assert set(spendly.AI_TEXT) == {"ollama", "groq"}
    assert spendly.AI_TEXT["ollama"].keys() == spendly.AI_TEXT["groq"].keys()


def test_groq_texts_never_claim_local_or_private_ai():
    for key, text in spendly.AI_TEXT["groq"].items():
        assert not LOCAL_CLAIMS.search(text), (key, text)


@pytest.mark.parametrize("path", PUBLIC_PAGES)
def test_groq_public_pages_make_no_local_claims(client, use_provider, path):
    use_provider("groq")
    text = _get(client, path)
    assert not LOCAL_CLAIMS.search(text), LOCAL_CLAIMS.search(text).group(0)


@pytest.mark.parametrize("path", LOGGED_IN_PAGES)
def test_groq_logged_in_pages_make_no_local_claims(client, use_provider, user_with_expense, path):
    use_provider("groq")
    text = _get(client, path)
    assert not LOCAL_CLAIMS.search(text), LOCAL_CLAIMS.search(text).group(0)


def test_groq_landing_page_says_cloud_ai(client, use_provider):
    use_provider("groq")
    text = _get(client, "/")
    assert "Fast cloud AI (Groq)" in text
    assert "Only you can see your data" in text
    assert "Your data stays in your account" not in text


def test_groq_assistant_header_and_insights_card(client, use_provider, user_with_expense):
    use_provider("groq")
    assert "Fast cloud AI (Groq)" in _get(client, "/profile")
    assert "written by AI (Groq, in the cloud)" in _get(client, "/analytics")


def test_groq_login_and_register_art(client, use_provider):
    use_provider("groq")
    assert "Fast cloud AI (Groq)" in _get(client, "/login")
    register = _get(client, "/register")
    assert "Fast cloud AI (Groq)" in register
    assert "Free, simple, and yours" in register


def test_ollama_keeps_the_local_texts(client, use_provider, user_with_expense):
    use_provider("ollama")
    home = _get(client, "/")
    assert "runs privately on your own computer" in home
    assert "Private by design" in home
    assert "Local AI, no API keys" in home
    assert "Runs privately on your computer" in _get(client, "/profile")
    assert "written by your local AI" in _get(client, "/analytics")


def test_ollama_login_and_register_art(client, use_provider):
    use_provider("ollama")
    assert "A private AI on your own computer" in _get(client, "/login")
    assert "Free, private, and yours" in _get(client, "/register")


def test_text_follows_a_provider_switch_without_restart(client, use_provider):
    use_provider("ollama")
    assert "Local AI, no API keys" in _get(client, "/")
    use_provider("groq")
    text = _get(client, "/")
    assert "Local AI, no API keys" not in text
    assert "Fast cloud AI (Groq)" in text
