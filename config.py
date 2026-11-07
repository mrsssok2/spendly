"""All Spendly settings, read from environment variables (with sensible defaults).

Order of precedence: real environment variables  >  the .env file next to this file  >  defaults.
In Docker, docker-compose.yml passes the settings as environment variables (the .env file
is NOT copied into the image), so the same code works on Windows and in a container.

Import this module before anything reads a setting: database.db reads DB_PATH from here.
"""
import os

from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# override=False: a variable that is already set (by Docker, the shell, or the tests) wins.
load_dotenv(os.path.join(PROJECT_DIR, ".env"), override=False)


def _bool(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


# --- Which LLM runs the AI features: "ollama" (local, default) or "groq" (cloud, e.g. on AWS). ---
# Voice input always stays on local faster-whisper (CPU), whatever the provider.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower() or "ollama"
if LLM_PROVIDER not in {"ollama", "groq"}:
    raise ValueError(f"LLM_PROVIDER must be 'ollama' or 'groq', not {LLM_PROVIDER!r}")

# --- Groq (only used when LLM_PROVIDER=groq). The key is a secret: never print or log it. ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com").rstrip("/")
GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-120b").strip()
# Empty = no vision model on this server: receipt scanning is switched off.
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()

# --- Ollama (local AI). In Docker: http://host.docker.internal:11434 (Ollama on the Windows host) ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "gemma3:4b")

# --- Voice (faster-whisper on the CPU). The model files are cached by Hugging Face (HF_HOME). ---
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

# --- Database. SPENDLY_DB_PATH is also what the tests and evals set to a temporary file. ---
DB_PATH = os.environ.get("SPENDLY_DB_PATH") or os.path.join(PROJECT_DIR, "database", "spendly.db")

# --- Flask ---
SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-only-change-me"
DEBUG = _bool("SPENDLY_DEBUG", True)          # only used by `python app.py` (gunicorn never debugs)
PORT = _int("SPENDLY_PORT", 5001)
