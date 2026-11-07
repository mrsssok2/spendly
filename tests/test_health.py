"""/health (Docker healthcheck) and config.py (settings from environment variables).
The real Ollama is never contacted: httpx.get is replaced with fakes. (Groq health: test_llm_provider.py)"""
import importlib
import sqlite3

import httpx
import pytest

import ai_agent
import app as spendly
import config
import database.db as db


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def ollama_up(monkeypatch):
    """Ollama answers /api/tags with the given model names."""
    def _up(*names):
        seen = []

        def fake_get(url, timeout):
            seen.append((url, timeout))
            return _FakeResponse({"models": [{"name": n} for n in names]})
        monkeypatch.setattr(ai_agent.httpx, "get", fake_get)
        return seen
    return _up


@pytest.fixture
def ollama_offline(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(ai_agent.httpx, "get", fake_get)


def test_health_all_ok(client, ollama_up):
    seen = ollama_up(ai_agent.OLLAMA_MODEL, config.OLLAMA_VISION_MODEL)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["app"] == "ok" and body["database"] == "ok" and body["status"] == "ok"
    assert body["ai"]["provider"] == "ollama"
    assert body["ai"]["reachable"] is True
    assert body["ai"]["chat_model"] == {"name": ai_agent.OLLAMA_MODEL, "available": True}
    assert body["ai"]["vision_model"] == {"name": config.OLLAMA_VISION_MODEL, "available": True}
    # Uses the configured URL, with a short timeout (a healthcheck must not hang).
    assert seen == [(f"{ai_agent.OLLAMA_URL}/api/tags", ai_agent.OLLAMA_HEALTH_TIMEOUT_SECONDS)]


def test_health_reports_missing_model(client, ollama_up):
    ollama_up(ai_agent.OLLAMA_MODEL)              # vision model not pulled
    body = client.get("/health").get_json()
    assert body["ai"]["chat_model"]["available"] is True
    assert body["ai"]["vision_model"]["available"] is False


def test_health_ollama_down_is_still_200(client, ollama_offline):
    """Ollama off = the AI features don't work, but the app does: Docker must not restart it."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["database"] == "ok"
    assert body["ai"] == {"provider": "ollama", "reachable": False, "url": ai_agent.OLLAMA_URL}
    assert body["status"] == "degraded"


def test_health_database_broken_is_503(client, ollama_offline, monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "no-such-folder" / "x.db"))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.get_json()["database"] == "error"


def test_health_needs_no_login_and_no_ai(client, ollama_offline, ai_must_not_be_called):
    assert client.get("/health").status_code == 200
    assert ai_must_not_be_called == []


def test_check_database():
    assert db.check_database() is True


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://host.docker.internal:11434/")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "llava:7b")
    monkeypatch.setenv("WHISPER_MODEL", "base")
    monkeypatch.setenv("SECRET_KEY", "from-env")
    monkeypatch.setenv("SPENDLY_DEBUG", "false")
    monkeypatch.setenv("SPENDLY_PORT", "6000")
    monkeypatch.setenv("LLM_PROVIDER", " Groq ")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_real")
    monkeypatch.setenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_VISION_MODEL", "")
    try:
        cfg = importlib.reload(config)
        assert cfg.OLLAMA_URL == "http://host.docker.internal:11434"     # trailing / removed
        assert cfg.OLLAMA_CHAT_MODEL == "llama3.1:8b"
        assert cfg.OLLAMA_VISION_MODEL == "llava:7b"
        assert cfg.WHISPER_MODEL == "base"
        assert cfg.SECRET_KEY == "from-env"
        assert cfg.DEBUG is False and cfg.PORT == 6000
        assert cfg.LLM_PROVIDER == "groq"
        assert cfg.GROQ_API_KEY == "gsk_test_not_real"
        assert cfg.GROQ_CHAT_MODEL == "llama-3.3-70b-versatile"
        assert cfg.GROQ_VISION_MODEL == ""            # empty = no receipt scanning on this server
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_config_defaults(monkeypatch):
    for name in ["OLLAMA_URL", "OLLAMA_CHAT_MODEL", "OLLAMA_VISION_MODEL", "WHISPER_MODEL",
                 "SECRET_KEY", "SPENDLY_DEBUG", "SPENDLY_PORT", "LLM_PROVIDER", "GROQ_API_KEY",
                 "GROQ_CHAT_MODEL", "GROQ_VISION_MODEL", "GROQ_BASE_URL"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)   # ignore a local .env
    try:
        cfg = importlib.reload(config)
        assert cfg.OLLAMA_URL == "http://localhost:11434"
        assert cfg.OLLAMA_CHAT_MODEL == "qwen2.5:7b"
        assert cfg.OLLAMA_VISION_MODEL == "gemma3:4b"
        assert cfg.WHISPER_MODEL == "small"
        assert cfg.PORT == 5001
        assert cfg.LLM_PROVIDER == "ollama"
        assert cfg.GROQ_API_KEY == ""
        assert cfg.GROQ_CHAT_MODEL == "openai/gpt-oss-120b"
        assert cfg.GROQ_VISION_MODEL == "qwen/qwen3.8-27b"
        assert cfg.GROQ_BASE_URL == "https://api.groq.com"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_modules_use_config():
    """No hardcoded Ollama URL or model names left: everything comes from config.py."""
    import vision
    import voice
    assert ai_agent.OLLAMA_URL == config.OLLAMA_URL
    assert ai_agent.OLLAMA_MODEL == config.OLLAMA_CHAT_MODEL
    assert ai_agent.OLLAMA_VISION_MODEL == config.OLLAMA_VISION_MODEL
    assert ai_agent.GROQ_CHAT_MODEL == config.GROQ_CHAT_MODEL
    assert ai_agent.GROQ_VISION_MODEL == config.GROQ_VISION_MODEL
    assert vision.get_vision_model is ai_agent.get_vision_model
    assert voice.WHISPER_MODEL == config.WHISPER_MODEL
    assert spendly.app.secret_key == config.SECRET_KEY


def test_invalid_provider_is_refused(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    try:
        with pytest.raises(ValueError, match="LLM_PROVIDER"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_check_database_returns_false_on_error(monkeypatch):
    def broken():
        raise sqlite3.OperationalError("unable to open database file")
    monkeypatch.setattr(db, "get_db", broken)
    assert db.check_database() is False
