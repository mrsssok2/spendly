"""Voice input: turn a short recording from the browser into text with faster-whisper.

    browser mic ──► MediaRecorder clip (webm/ogg/mp4, max 30 s, max 5 MB)
                ──► /ai/voice (logged-in users only)
                ──► decode()      PyAV (inside faster-whisper) reads the clip IN MEMORY → 16 kHz samples
                ──► checks        CODE: not empty, not too long, not silent
                ──► transcribe    Whisper "small" on the CPU (int8), in the language the USER chose:
                                  EN = transcribe English, हिंदी = translate Hindi into English
                ──► script check  CODE: the text must be in Roman letters, else "try again"
                ──► text goes into the chat box; the USER reads/fixes it and presses Send

Why the CPU: the 6 GB GPU is already busy with qwen2.5:7b in Ollama. If Whisper took GPU
memory too, Ollama could be pushed partly onto the CPU, which made tool calling unreliable.
"small" + int8 needs about 0.5 GB of normal RAM and is fast enough for 30-second clips.

Recordings are never stored: the audio only lives in memory for one request (no temp files),
and nothing is saved to the database here. The transcript is only ever text in the input box.
"""

import io
import re
import threading
import time
import unicodedata

import numpy as np

import config
from ai_agent import AI_LOG

WHISPER_MODEL = config.WHISPER_MODEL   # "small" unless WHISPER_MODEL is set
WHISPER_DEVICE = "cpu"            # keep the GPU free for Ollama
WHISPER_COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16_000              # Whisper always works on 16 kHz mono audio
MAX_AUDIO_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_SECONDS = 30
SECONDS_GRACE = 1.5               # the browser's 30 s auto-stop can overshoot a little
SILENCE_PEAK = 0.01               # loudest sample below this (out of 1.0) = nothing was said
NO_SPEECH_LIMIT = 0.6             # Whisper's own "this part was probably silence" score
# The USER picks the language (EN / हिंदी toggle). We don't trust auto-detection: an English
# sentence in an Indian accent was detected as Hindi and came out as garbled Devanagari.
#   en → transcribe as English (Hinglish words come out in Roman letters: "auto mein 250 diye")
#   hi → translate Hindi speech to English, so the assistant always receives English text
MODES = {
    "en": {"language": "en", "task": "transcribe"},
    "hi": {"language": "hi", "task": "translate"},
}
DEFAULT_MODE = "en"
# Words the user is likely to say: makes Whisper pick "electricity bill" over "a lecture city bill".
EXPENSE_PROMPT = ("Spendly expenses in rupees: spent, paid, salary, income, budget, electricity bill, "
                  "auto, uber, Swiggy, Zomato, groceries, rent, yesterday, today.")

DIDNT_CATCH = "I didn't catch that, try again."
WRONG_SCRIPT = "That didn't sound right — try again or type it."
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


class VoiceError(Exception):
    """A problem with the recording. The message is safe to show to the user."""


class VoiceUnavailable(Exception):
    """The speech model couldn't be loaded. The message is safe to show to the user."""


# ---------- The model: loaded once, reused for every request ----------

_MODEL = None
_MODEL_LOCK = threading.Lock()       # only one thread loads the model
_TRANSCRIBE_LOCK = threading.Lock()  # one transcription at a time, so CPU use stays predictable


def model_is_downloaded():
    """True if the Whisper files are already on this computer (no internet needed)."""
    if _MODEL is not None:
        return True
    try:
        from faster_whisper.utils import download_model
        download_model(WHISPER_MODEL, local_files_only=True)
        return True
    except Exception:
        return False


def _get_model():
    """Load Whisper the first time it's needed, then reuse it. The first load downloads
    the model (about 480 MB) once; after that everything works offline."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            try:
                from faster_whisper import WhisperModel   # imported here so app start-up stays fast
            except ImportError:
                raise VoiceUnavailable("Voice input isn't installed. Run: pip install -r requirements.txt")
            if not model_is_downloaded():
                AI_LOG.info("voice: downloading Whisper '%s' model (first time only)...", WHISPER_MODEL)
            started = time.perf_counter()
            try:
                _MODEL = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
            except Exception as e:
                AI_LOG.warning("voice: couldn't load Whisper: %s", e)
                raise VoiceUnavailable(
                    "Couldn't load the voice model. The first use needs an internet connection to "
                    "download it (once). You can still type your message.")
            AI_LOG.info("voice: Whisper '%s' ready on %s/%s in %.1fs", WHISPER_MODEL, WHISPER_DEVICE,
                        WHISPER_COMPUTE_TYPE, time.perf_counter() - started)
    return _MODEL


# ---------- Step 1: read the recording (in memory) ----------

def decode(data):
    """Recording bytes → 16 kHz mono float samples. Raises VoiceError for bad input."""
    if not data:
        raise VoiceError(DIDNT_CATCH)
    if len(data) > MAX_AUDIO_BYTES:
        raise VoiceError("That recording is too large. Please keep it under 5 MB (about 30 seconds).")
    try:
        from faster_whisper import decode_audio
    except ImportError:
        raise VoiceUnavailable("Voice input isn't installed. Run: pip install -r requirements.txt")
    try:
        audio = decode_audio(io.BytesIO(data), sampling_rate=SAMPLE_RATE)
    except Exception:
        raise VoiceError("Couldn't read that recording. Please try again.")
    if audio is None or len(audio) == 0:
        raise VoiceError(DIDNT_CATCH)
    return audio


def check_audio(audio):
    """Code checks before the model runs: length and silence."""
    seconds = len(audio) / SAMPLE_RATE
    if seconds > MAX_SECONDS + SECONDS_GRACE:
        raise VoiceError(f"Please keep recordings under {MAX_SECONDS} seconds.")
    if float(np.max(np.abs(audio))) < SILENCE_PEAK:
        raise VoiceError(DIDNT_CATCH)
    return seconds


# ---------- Step 2: speech → text ----------

def _detect(model, audio):
    """Whisper's own language guess, ONLY for the log (it never chooses the language)."""
    try:
        language, probability, _ = model.detect_language(audio, vad_filter=True)
        return language, probability
    except Exception:
        return "?", 0.0


def clean_text(text):
    text = text.translate(DEVANAGARI_DIGITS)       # "२५०" → "250"
    return re.sub(r"\s+", " ", text).strip()


def is_roman_text(text):
    """Both modes must give Roman (English-alphabet) letters: EN transcribes, हिंदी translates.
    Any letter from another script (Devanagari, Urdu, …) means Whisper went wrong."""
    return all(not ch.isalpha() or unicodedata.name(ch, "").startswith("LATIN") for ch in text)


def transcribe_audio(data, mode=DEFAULT_MODE):
    """Recording bytes → {"text", "mode", "seconds", "took"}. `mode` is "en" or "hi" (the
    user's toggle). Raises VoiceError / VoiceUnavailable with user-safe messages.
    Writes nothing to disk."""
    if mode not in MODES:
        raise VoiceError("Please choose EN or हिंदी for voice input.")
    audio = decode(data)
    seconds = check_audio(audio)
    model = _get_model()

    started = time.perf_counter()
    with _TRANSCRIBE_LOCK:
        detected, probability = _detect(model, audio)
        segments, _ = model.transcribe(
            audio,
            **MODES[mode],                     # the user's choice, never auto-detect
            beam_size=5,
            vad_filter=True,                   # skip silent parts (stops "Thank you." hallucinations)
            condition_on_previous_text=False,
            initial_prompt=EXPENSE_PROMPT,
        )
        parts = [s.text for s in segments if s.no_speech_prob < NO_SPEECH_LIMIT]
    took = time.perf_counter() - started

    text = clean_text(" ".join(parts))
    AI_LOG.info("voice: mode=%s, %.1fs of audio, Whisper guessed %s (%.0f%%) → %r (transcribed in %.1fs)",
                mode, seconds, detected, probability * 100, text, took)
    if not re.search(r"\w", text):
        raise VoiceError(DIDNT_CATCH)
    if not is_roman_text(text):
        AI_LOG.info("voice: transcript isn't in Roman letters, not filling the box")
        raise VoiceError(WRONG_SCRIPT)
    return {"text": text, "mode": mode, "seconds": round(seconds, 1), "took": round(took, 1)}


if __name__ == "__main__":
    # Quick manual check: python voice.py some_recording.webm [en|hi]
    import sys
    with open(sys.argv[1], "rb") as f:
        print(transcribe_audio(f.read(), sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODE))
