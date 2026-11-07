"""Manual check that LangChain can talk to the configured model (LLM_PROVIDER: Ollama or Groq).

Run with:  python test_ai.py
(It has no test_ functions, so pytest doesn't run it: the model may be offline.)
"""

import sys

from ai_agent import PROVIDER, AIError, ask_ai, chat_model_name

if __name__ == "__main__":
    print(f"Provider: {PROVIDER}, model: {chat_model_name()}")   # never the API key
    try:
        print(ask_ai("Say hi in 5 words"))
    except AIError as e:
        print(f"Error: {e}")
        sys.exit(1)
