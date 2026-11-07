"""LLM provider switch (config.LLM_PROVIDER = ollama | groq).

Groq is NEVER contacted: the key is fake, httpx.get is faked for /health, and model calls use
fake models or a patched ChatGroq._generate. These tests check the wiring, the 429 handling,
receipt scanning without a vision model, and that the API key never leaks.
"""
import io

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from PIL import Image

import ai_agent
import config
import vision
from conftest import FakeToolModel, ask_ai

FAKE_KEY = "gsk_FAKE_KEY_for_tests_0123456789"


@pytest.fixture
def groq_provider(monkeypatch):
    """Switch the app to Groq with a fake key (no network: model calls must be faked)."""
    monkeypatch.setattr(ai_agent, "PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setattr(ai_agent, "GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(ai_agent, "GROQ_VISION_MODEL", "qwen/qwen3.8-27b")


@pytest.fixture
def groq_models_list(monkeypatch):
    """Groq's GET /openai/v1/models answers with these model ids. Returns the requests seen."""
    def _serve(*ids, status=200):
        seen = []

        def fake_get(url, headers=None, timeout=None):
            seen.append({"url": url, "headers": headers, "timeout": timeout})
            request = httpx.Request("GET", url)
            return httpx.Response(status, json={"data": [{"id": i} for i in ids]}, request=request)
        monkeypatch.setattr(ai_agent.httpx, "get", fake_get)
        return seen
    return _serve


def rate_limit_error(retry_after="1"):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, headers=headers, request=request)
    return groq.RateLimitError("Rate limit reached", response=response, body=None)


# ---------- Which model class each provider builds ----------

def test_ollama_is_the_default_provider():
    assert config.LLM_PROVIDER in {"ollama", "groq"}
    assert ai_agent.PROVIDER == "ollama"           # conftest pins tests to ollama
    model = ai_agent.get_chat_model()
    assert isinstance(model, ChatOllama) and model.model == ai_agent.OLLAMA_MODEL
    assert ai_agent.get_chat_model(json_mode=True).format == "json"
    assert ai_agent.get_vision_model().model == config.OLLAMA_VISION_MODEL


def test_groq_chat_model(groq_provider):
    model = ai_agent.get_chat_model()
    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"
    assert model.temperature < 0.001 and model.max_retries == 0   # ChatGroq sends 0 as 1e-8
    assert model.reasoning_effort == ai_agent.GROQ_REASONING_EFFORT     # gpt-oss only
    assert model.model_kwargs == {}
    assert ai_agent.get_chat_model(json_mode=True).model_kwargs == {"response_format": {"type": "json_object"}}


def test_groq_non_reasoning_model_gets_no_reasoning_effort(groq_provider, monkeypatch):
    monkeypatch.setattr(ai_agent, "GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
    assert ai_agent.get_chat_model().reasoning_effort is None


def test_groq_vision_model_is_json(groq_provider):
    model = ai_agent.get_vision_model()
    assert isinstance(model, ChatGroq) and model.model_name == "qwen/qwen3.8-27b"
    assert model.model_kwargs == {"response_format": {"type": "json_object"}}


def test_groq_without_key_is_a_friendly_error(groq_provider, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(ai_agent.AIError, match="GROQ_API_KEY is missing"):
        ai_agent.get_chat_model()


def test_api_key_never_shows_in_the_model_object(groq_provider):
    model = ai_agent.get_chat_model()
    assert FAKE_KEY not in repr(model) and FAKE_KEY not in str(model)
    assert FAKE_KEY not in str(model.model_dump())


# ---------- 429: one short retry, then "AI is busy" ----------

def test_429_short_wait_is_retried_once(groq_provider, monkeypatch):
    calls, sleeps = [], []

    def flaky(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise rate_limit_error("1")
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])
    monkeypatch.setattr(ChatGroq, "_generate", flaky)
    monkeypatch.setattr(ai_agent.time, "sleep", sleeps.append)
    assert ai_agent.ask_ai("hello") == "ok"
    assert len(calls) == 2 and sleeps == [1.0]


def test_429_long_wait_is_not_retried(groq_provider, monkeypatch):
    calls = []

    def limited(self, *args, **kwargs):
        calls.append(1)
        raise rate_limit_error("45")          # e.g. the per-minute token limit
    monkeypatch.setattr(ChatGroq, "_generate", limited)
    monkeypatch.setattr(ai_agent.time, "sleep", lambda s: pytest.fail("must not wait 45 s"))
    with pytest.raises(ai_agent.AIBusy, match="AI is busy, try again in a moment."):
        ai_agent.ask_ai("hello")
    assert len(calls) == 1


def test_429_twice_gives_up(groq_provider, monkeypatch):
    calls = []

    def limited(self, *args, **kwargs):
        calls.append(1)
        raise rate_limit_error(None)          # no retry-after: short default wait
    monkeypatch.setattr(ChatGroq, "_generate", limited)
    monkeypatch.setattr(ai_agent.time, "sleep", lambda s: None)
    with pytest.raises(ai_agent.AIBusy):
        ai_agent.ask_ai("hello")
    assert len(calls) == 1 + ai_agent.GROQ_RATE_LIMIT_RETRIES


class _RateLimited(FakeToolModel):
    def _generate(self, *args, **kwargs):
        raise rate_limit_error("60")


def test_chat_route_answers_429_ai_busy(client, user, monkeypatch):
    monkeypatch.setattr(ai_agent, "get_chat_model", lambda *a, **k: _RateLimited(messages=iter([])))
    status, data = ask_ai(client, "how much did I spend on food this month?")
    assert status == 429
    assert data["error"] == "AI is busy, try again in a moment."


def test_insights_route_answers_429_ai_busy(client, user, monkeypatch):
    import ai_insights
    from test_insights import sample_data
    sample_data(user)
    monkeypatch.setattr(ai_insights, "get_chat_model", lambda *a, **k: _RateLimited(messages=iter([])))
    response = client.post("/analytics/insights")
    assert response.status_code == 429
    assert response.get_json()["error"] == "AI is busy, try again in a moment."


def test_other_groq_errors_are_friendly_and_hide_the_key(groq_provider):
    request = httpx.Request("POST", "https://api.groq.com/x")
    errors = [
        groq.AuthenticationError(f"Invalid API Key {FAKE_KEY}", response=httpx.Response(401, request=request), body=None),
        groq.NotFoundError("model not found", response=httpx.Response(404, request=request), body=None),
        groq.BadRequestError("tool_use_failed", response=httpx.Response(400, request=request), body=None),
        groq.InternalServerError("oops", response=httpx.Response(500, request=request), body=None),
        groq.APIConnectionError(request=request),
    ]
    for error in errors:
        def boom(e=error):
            raise e
        with pytest.raises(ai_agent.AIError) as caught:
            ai_agent._call_ai(boom)
        assert FAKE_KEY not in str(caught.value)
        assert not isinstance(caught.value, ai_agent.AIBusy)


# ---------- /health with Groq ----------

def test_health_groq_ok(client, groq_provider, groq_models_list):
    seen = groq_models_list("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.1-8b-instant")
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["ai"] == {
        "provider": "groq", "reachable": True, "url": config.GROQ_BASE_URL,
        "chat_model": {"name": "openai/gpt-oss-120b", "available": True},
        "vision_model": {"name": "qwen/qwen3.8-27b", "available": True},
    }
    assert FAKE_KEY not in response.get_data(as_text=True)
    assert seen[0]["url"] == f"{config.GROQ_BASE_URL}/openai/v1/models"
    assert seen[0]["timeout"] == ai_agent.AI_HEALTH_TIMEOUT_SECONDS
    # Cached: Docker's healthcheck every 30 s doesn't ask Groq every time.
    client.get("/health")
    assert len(seen) == 1


def test_health_groq_model_missing(client, groq_provider, groq_models_list):
    groq_models_list("llama-3.1-8b-instant")
    ai = client.get("/health").get_json()["ai"]
    assert ai["reachable"] is True
    assert ai["chat_model"]["available"] is False and ai["vision_model"]["available"] is False


def test_health_groq_bad_key_is_degraded_not_503(client, groq_provider, groq_models_list):
    groq_models_list(status=401)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "degraded"
    assert body["ai"]["reachable"] is False and body["ai"]["error"] == "Groq rejected the API key"
    assert FAKE_KEY not in response.get_data(as_text=True)


def test_health_groq_no_key(client, groq_provider, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(ai_agent.httpx, "get", lambda *a, **k: pytest.fail("no request without a key"))
    body = client.get("/health").get_json()
    assert body["ai"]["reachable"] is False and body["ai"]["error"] == "GROQ_API_KEY is not set"


# ---------- Receipt scanning on Groq ----------

def _png():
    buffer = io.BytesIO()
    Image.new("RGB", (400, 600), "white").save(buffer, "PNG")
    return buffer.getvalue()


def test_no_vision_model_shows_receipts_unavailable(client, user, groq_provider, monkeypatch):
    monkeypatch.setattr(ai_agent, "GROQ_VISION_MODEL", "")
    monkeypatch.setattr(vision, "get_vision_model", lambda *a, **k: pytest.fail("AI must not be called"))
    page = client.get("/expenses/scan")
    assert page.status_code == 200
    text = page.get_data(as_text=True)
    assert "Receipt scanning isn&#39;t available on this server" in text
    assert 'id="scan-form"' not in text                     # no upload form
    response = client.post("/expenses/scan", data={"receipt": (io.BytesIO(_png()), "r.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 503
    assert "Receipt scanning isn&#39;t available on this server" in response.get_data(as_text=True)


def test_vision_model_not_on_groq_shows_receipts_unavailable(client, user, groq_provider, groq_models_list):
    groq_models_list("openai/gpt-oss-120b")                 # no vision model on this account
    text = client.get("/expenses/scan").get_data(as_text=True)
    assert "Receipt scanning isn&#39;t available on this server" in text


def test_groq_vision_model_reads_receipts(client, user, groq_provider, groq_models_list, monkeypatch):
    groq_models_list("openai/gpt-oss-120b", "qwen/qwen3.8-27b")
    reading = '{"shop_name": "Cafe", "final_amount": 250, "date": "05/11/2026", "business_type": "cafe"}'
    monkeypatch.setattr(vision, "get_vision_model",
                        lambda *a, **k: FakeToolModel(messages=iter([AIMessage(content=reading)])))
    page = client.get("/expenses/scan").get_data(as_text=True)
    assert 'id="scan-form"' in page
    response = client.post("/expenses/scan", data={"receipt": (io.BytesIO(_png()), "r.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 200
    assert 'value="250' in response.get_data(as_text=True)


def test_voice_stays_local():
    """Voice never uses the LLM provider: faster-whisper on the CPU."""
    import voice
    source = open(voice.__file__, encoding="utf-8").read()
    assert "groq" not in source.lower() and "get_chat_model" not in source
