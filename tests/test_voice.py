"""Voice input (voice.py + /ai/voice). The real Whisper model is never loaded: a fake
model replies instead. Test recordings are tiny WAV files generated with numpy."""
import io
import wave
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

import voice
from conftest import ask_ai, count_expenses, tool_call


def make_wav(seconds=1.0, loud=True, rate=16_000):
    """A short mono WAV: a 440 Hz tone (stands in for speech) or pure silence."""
    t = np.arange(int(seconds * rate)) / rate
    samples = 0.3 * np.sin(2 * np.pi * 440 * t) if loud else np.zeros_like(t)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((samples * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def segment(text, no_speech_prob=0.05):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob)


class FakeWhisper:
    """Pretends to be faster_whisper.WhisperModel: returns the scripted segments, and
    `detected` as its (ignored) language guess."""

    def __init__(self, segments, detected=("en", 0.9)):
        self.segments = segments
        self.detected = detected
        self.calls = []

    def detect_language(self, audio, **options):
        return self.detected[0], self.detected[1], [self.detected]

    def transcribe(self, audio, **options):
        self.calls.append({"audio": audio, **options})
        info = SimpleNamespace(language=options.get("language"), language_probability=1.0)
        return iter(self.segments), info


@pytest.fixture
def fake_whisper(monkeypatch):
    def _install(segments, detected=("en", 0.9)):
        model = FakeWhisper(segments, detected)
        monkeypatch.setattr(voice, "_get_model", lambda: model)
        return model
    return _install


@pytest.fixture
def whisper_must_not_be_called(monkeypatch):
    def _forbidden():
        raise AssertionError("Whisper was called, but it should not have been!")
    monkeypatch.setattr(voice, "_get_model", _forbidden)


def send_audio(client, data, filename="speech.webm", lang=None):
    form = {"audio": (io.BytesIO(data), filename)}
    if lang is not None:
        form["lang"] = lang
    response = client.post("/ai/voice", data=form, content_type="multipart/form-data")
    return response.status_code, response.get_json()


# ---------- EN mode (default) ----------

def test_english_speech_becomes_text_only(client, user, fake_whisper, ai_must_not_be_called):
    model = fake_whisper([segment(" Spent 250 on auto"), segment(" today.")])
    status, data = send_audio(client, make_wav(), lang="en")
    assert status == 200
    assert data == {"text": "Spent 250 on auto today.", "lang": "en"}
    # Speech is only text in the box: nothing sent to the AI, nothing saved.
    assert ai_must_not_be_called == []
    assert count_expenses(user) == 0
    call = model.calls[0]
    assert call["language"] == "en" and call["task"] == "transcribe"
    assert call["initial_prompt"] == voice.EXPENSE_PROMPT
    assert "electricity bill" in call["initial_prompt"]
    # The audio went to the model as in-memory samples (no file path), 16 kHz mono.
    assert isinstance(call["audio"], np.ndarray) and len(call["audio"]) == 16_000
    assert call["vad_filter"] is True


def test_no_lang_field_means_english(client, user, fake_whisper):
    model = fake_whisper([segment("paid rent")])
    status, data = send_audio(client, make_wav())
    assert status == 200 and data["lang"] == "en"
    assert model.calls[0]["language"] == "en"


def test_english_is_used_even_when_whisper_guesses_hindi(client, user, fake_whisper):
    # The real bug: Indian-accented English was auto-detected as Hindi. The guess is only logged.
    model = fake_whisper([segment(" What's yesterday's electricity bill?")], detected=("hi", 0.71))
    status, data = send_audio(client, make_wav(), lang="en")
    assert status == 200
    assert data["text"] == "What's yesterday's electricity bill?"
    assert [c["language"] for c in model.calls] == ["en"]


def test_hinglish_in_english_mode_keeps_roman_letters(client, user, fake_whisper):
    fake_whisper([segment(" aaj auto mein 250 diye")])
    status, data = send_audio(client, make_wav(), lang="en")
    assert status == 200 and data["text"] == "aaj auto mein 250 diye"


def test_devanagari_text_in_english_mode_is_refused(client, user, fake_whisper):
    fake_whisper([segment(" बाज से स्तरे से लग्री सी दीम लिएं")], detected=("hi", 0.6))
    status, data = send_audio(client, make_wav(), lang="en")
    assert status == 400
    assert data["error"] == "That didn't sound right — try again or type it."
    assert "text" not in data                                     # nothing goes into the box


def test_mixed_script_is_refused(client, user, fake_whisper):
    fake_whisper([segment(" spent 250 on ऑटो")])
    status, data = send_audio(client, make_wav(), lang="en")
    assert status == 400 and "didn't sound right" in data["error"]


def test_devanagari_digits_become_normal_digits(client, user, fake_whisper):
    fake_whisper([segment("auto ₹२५०")])
    status, data = send_audio(client, make_wav(), lang="en")
    assert status == 200
    assert data["text"] == "auto ₹250"


# ---------- हिंदी mode: Hindi speech translated to English ----------

def test_hindi_mode_translates_to_english(client, user, fake_whisper):
    model = fake_whisper([segment(" Gave 250 in the auto today.")], detected=("hi", 0.95))
    status, data = send_audio(client, make_wav(), lang="hi")
    assert status == 200
    assert data == {"text": "Gave 250 in the auto today.", "lang": "hi"}
    call = model.calls[0]
    assert call["language"] == "hi" and call["task"] == "translate"
    assert call["initial_prompt"] == voice.EXPENSE_PROMPT


def test_hindi_mode_untranslated_devanagari_is_refused(client, user, fake_whisper):
    fake_whisper([segment(" आज ऑटो में 250 दिए")], detected=("hi", 0.95))
    status, data = send_audio(client, make_wav(), lang="hi")
    assert status == 400 and "didn't sound right" in data["error"]


def test_unknown_language_is_refused(client, user, whisper_must_not_be_called):
    status, data = send_audio(client, make_wav(), lang="fr")
    assert status == 400
    assert "EN" in data["error"]


# ---------- Silence / bad recordings ----------

def test_silence_is_caught_before_the_model(client, user, whisper_must_not_be_called):
    status, data = send_audio(client, make_wav(loud=False))
    assert status == 400
    assert data["error"] == "I didn't catch that, try again."


def test_no_speech_found_by_the_model(client, user, fake_whisper):
    fake_whisper([])
    status, data = send_audio(client, make_wav())
    assert status == 400
    assert data["error"] == "I didn't catch that, try again."


def test_probable_silence_segments_are_dropped(client, user, fake_whisper):
    # Whisper's classic hallucination on noise: "Thank you." with a high no-speech score.
    fake_whisper([segment(" Thank you.", no_speech_prob=0.9)])
    status, data = send_audio(client, make_wav())
    assert status == 400
    assert data["error"] == "I didn't catch that, try again."


def test_empty_upload(client, user, whisper_must_not_be_called):
    status, data = send_audio(client, b"")
    assert status == 400
    assert data["error"] == "I didn't catch that, try again."


def test_missing_file_field(client, user, whisper_must_not_be_called):
    response = client.post("/ai/voice", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_garbage_bytes_are_refused(client, user, whisper_must_not_be_called):
    status, data = send_audio(client, b"this is not audio at all" * 50)
    assert status == 400
    assert "Couldn't read that recording" in data["error"]


def test_recording_over_30_seconds_is_refused(client, user, whisper_must_not_be_called):
    status, data = send_audio(client, make_wav(seconds=33, rate=8_000))
    assert status == 400
    assert "30 seconds" in data["error"]


def test_recording_over_5_mb_is_refused(client, user, whisper_must_not_be_called):
    status, data = send_audio(client, b"\0" * (5 * 1024 * 1024 + 1))
    assert status == 400
    assert "5 MB" in data["error"]


def test_huge_request_gets_json_not_a_redirect(client, user, whisper_must_not_be_called):
    status, data = send_audio(client, b"\0" * (7 * 1024 * 1024))
    assert status == 413
    assert "5 MB" in data["error"]


# ---------- Security and failures ----------

def test_logged_out_users_cannot_use_voice(client, whisper_must_not_be_called):
    status, data = send_audio(client, make_wav())
    assert status == 401
    assert client.get("/ai/voice/status").status_code == 401


def test_model_unavailable_is_a_clear_error_and_chat_still_works(client, user, monkeypatch, fake_ai):
    def _broken():
        raise voice.VoiceUnavailable("Couldn't load the voice model. You can still type your message.")
    monkeypatch.setattr(voice, "_get_model", _broken)
    status, data = send_audio(client, make_wav())
    assert status == 503
    assert data["error"] == "Voice input unavailable"
    assert "still type" in data["detail"]

    # The typed assistant is unaffected, and adding still needs Confirm.
    fake_ai(tool_call("add_expense_tool", amount=250, category="Transport",
                      date=date.today().isoformat(), description="auto"), "Please review and confirm.")
    status, data = ask_ai(client, "spent 250 on auto today")
    assert status == 200
    assert data["cards"][0]["type"] == "add"
    assert count_expenses(user) == 0


def test_status_route(client, user, monkeypatch):
    import app as spendly
    monkeypatch.setattr(spendly, "model_is_downloaded", lambda: False)
    assert client.get("/ai/voice/status").get_json() == {"model_ready": False}


def test_model_is_loaded_once_on_the_cpu(monkeypatch):
    created = []

    class FakeWhisperModel:
        def __init__(self, size, device, compute_type):
            created.append((size, device, compute_type))

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(voice, "model_is_downloaded", lambda: True)
    monkeypatch.setattr(voice, "_MODEL", None)
    first = voice._get_model()
    second = voice._get_model()
    assert first is second
    assert created == [("small", "cpu", "int8")]   # CPU + int8: the GPU stays free for Ollama


def test_model_load_failure_becomes_voice_unavailable(monkeypatch):
    class Offline:
        def __init__(self, *args, **kwargs):
            raise OSError("no internet")

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", Offline)
    monkeypatch.setattr(voice, "model_is_downloaded", lambda: False)
    monkeypatch.setattr(voice, "_MODEL", None)
    with pytest.raises(voice.VoiceUnavailable, match="internet"):
        voice._get_model()
    assert voice._MODEL is None


# ---------- Page ----------

def test_profile_page_has_voice_controls(client, user):
    page = client.get("/profile").get_data(as_text=True)
    assert 'id="ai-mic"' in page
    assert 'id="ai-speak"' in page
    assert 'id="ai-voice-status"' in page
    assert 'data-voice-url="/ai/voice"' in page
    assert 'id="ai-voice-lang" data-lang="en"' in page        # EN is the default
    # Auto-send is OFF by default: the user reads the transcript first.
    autosend = page[page.index('id="ai-autosend"') - 40: page.index('id="ai-autosend"') + 40]
    assert "checked" not in autosend
