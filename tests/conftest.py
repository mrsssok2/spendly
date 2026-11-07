"""Shared test setup (pytest "fixtures") for Spendly.

Safety first: before the app is imported, SPENDLY_DB_PATH points at a temporary file,
so importing app.py (which creates tables and seeds a demo user) never touches the real
database/spendly.db. Every test then gets its OWN brand-new empty database.
"""
import hashlib
import os
import sys
import tempfile

# ---- Must run before `import app` (app.py sets up the database on import) ----
_SESSION_DIR = tempfile.mkdtemp(prefix="spendly-tests-")
os.environ["SPENDLY_DB_PATH"] = os.path.join(_SESSION_DIR, "import.db")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
REAL_DB = os.path.join(PROJECT_DIR, "database", "spendly.db")

import httpx  # noqa: E402
import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import ai_agent  # noqa: E402
import app as spendly  # noqa: E402
import database.db as db  # noqa: E402


def _fingerprint(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return os.path.getmtime(path), hashlib.sha256(f.read()).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def real_database_is_never_touched():
    """Runs around the WHOLE test session: fails if the real spendly.db changed at all."""
    before = _fingerprint(REAL_DB)
    yield
    assert _fingerprint(REAL_DB) == before, "The real database/spendly.db was modified by the tests!"


@pytest.fixture(autouse=True)
def fresh_database(tmp_path, monkeypatch):
    """Every test gets a brand-new, empty database in its own temporary folder."""
    path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    assert os.path.abspath(db.DB_PATH) != os.path.abspath(REAL_DB)
    return path


@pytest.fixture(autouse=True)
def local_provider_no_cloud(monkeypatch):
    """Tests behave the same whatever LLM_PROVIDER / GROQ_API_KEY the local .env has: provider
    "ollama" and NO Groq key, so nothing can ever reach Groq by accident. Groq tests opt in
    with a fake key and fake HTTP (tests/test_llm_provider.py)."""
    import config
    monkeypatch.setattr(ai_agent, "PROVIDER", "ollama")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    ai_agent._groq_models_cache.update(at=None, models=None)
    yield
    ai_agent._groq_models_cache.update(at=None, models=None)


@pytest.fixture(autouse=True)
def fresh_chat_memory(monkeypatch):
    """Every test gets empty AI chat memory (a new LangGraph checkpointer)."""
    monkeypatch.setattr(ai_agent, "CHECKPOINTER", InMemorySaver())


@pytest.fixture(autouse=True)
def fresh_rate_limit():
    """Every test starts with no AI requests counted against the rate limit."""
    spendly._ai_requests.clear()
    yield
    spendly._ai_requests.clear()


@pytest.fixture(autouse=True)
def quiet_ai_log():
    ai_agent.AI_LOG.disabled = True
    yield
    ai_agent.AI_LOG.disabled = False


@pytest.fixture
def client():
    """A fake browser for sending requests to the app."""
    spendly.app.config["TESTING"] = True
    return spendly.app.test_client()


@pytest.fixture
def make_user():
    """Create a user and return their id: make_user("asha@x.com")."""
    def _make(email, name="Test User", password="secret123"):
        user_id = db.create_user(name, email, password)
        assert user_id is not None
        return user_id
    return _make


def login(client, email, password="secret123"):
    return client.post("/login", data={"email": email, "password": password})


@pytest.fixture
def user(client, make_user):
    """A user called Asha, already logged in on `client`. Returns her user id."""
    user_id = make_user("asha@example.com", name="Asha")
    login(client, "asha@example.com")
    return user_id


@pytest.fixture
def other_user(make_user):
    """A second user, Ravi (not logged in). Returns his user id."""
    return make_user("ravi@example.com", name="Ravi")


# ---------- Fake AI ----------

class FakeToolModel(GenericFakeChatModel):
    """A pretend chat model that replies with scripted messages, in order.
    bind_tools() is accepted (and ignored) so create_agent can use it like a real model."""

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name, **args):
    """A scripted AI message that calls one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{name}"}])


@pytest.fixture
def fake_ai(monkeypatch):
    """Replace the real Ollama model with a scripted fake.

    Usage: fake_ai(tool_call("add_expense_tool", amount=250, ...), "Ready for review.")
    Each argument is the model's next reply (a tool call or plain text).
    """
    def _script(*replies):
        messages = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
        model = FakeToolModel(messages=iter(messages))
        monkeypatch.setattr(ai_agent, "get_chat_model", lambda *a, **k: model)
        return model
    return _script


class _OllamaDown(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        raise httpx.ConnectError("connection refused")


@pytest.fixture
def ollama_down(monkeypatch):
    """Pretend Ollama isn't running: every model call fails to connect."""
    monkeypatch.setattr(ai_agent, "get_chat_model", lambda *a, **k: _OllamaDown(messages=iter([])))


@pytest.fixture
def ai_must_not_be_called(monkeypatch):
    """Fail the test if anything tries to use the AI."""
    calls = []

    def _forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("The AI was called, but it should not have been!")

    monkeypatch.setattr(spendly, "run_expense_assistant", _forbidden)
    return calls


def ask_ai(client, message):
    """Send a message to the AI assistant; returns (status_code, json)."""
    response = client.post("/ai/assistant", json={"message": message})
    return response.status_code, response.get_json()


def count_expenses(user_id):
    return db.get_expense_summary(user_id)["count"]
