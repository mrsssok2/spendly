// AI assistant box on the profile page (vanilla JS).
// Flow: send message -> AI reply + preview cards -> Confirm / Cancel / Select.
// Nothing changes in the database until a Confirm button is clicked.
// All text is inserted with textContent, never innerHTML, so AI output can't inject HTML.

(function () {
    const box = document.getElementById("ai-assistant");
    if (!box) return;

    const chat = document.getElementById("ai-chat");
    const form = document.getElementById("ai-form");
    const input = document.getElementById("ai-message");
    const sendBtn = document.getElementById("ai-send");
    const newChatBtn = document.getElementById("ai-new-chat");
    const urls = {
        ask: box.dataset.askUrl,
        select: box.dataset.selectUrl,
        confirm: box.dataset.confirmUrl,
        cancel: box.dataset.cancelUrl,
        history: box.dataset.historyUrl,
        newChat: box.dataset.newChatUrl,
        voice: box.dataset.voiceUrl,
        voiceStatus: box.dataset.voiceStatusUrl,
    };
    const kindClass = { success: "ai-msg-success", warning: "ai-msg-warning", note: "ai-msg-note", error: "ai-msg-error",
                        translation: "ai-msg-translation" };

    // ---------- Small helpers ----------

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function button(text, className) {
        const b = el("button", "btn btn-sm " + (className || ""), text);
        b.type = "button";
        return b;
    }

    function scrollDown() {
        chat.scrollTop = chat.scrollHeight;
    }

    function addBubble(who, text, extraClass) {
        const bubble = el("div", "ai-msg ai-msg-" + who + (extraClass ? " " + extraClass : ""), text);
        chat.appendChild(bubble);
        scrollDown();
        return bubble;
    }

    // "Show SQL" under an answer that came from a read-only database query.
    function addSqlToggle(bubble, sql) {
        const details = el("details", "ai-sql");
        details.appendChild(el("summary", "", "Show SQL"));
        details.appendChild(el("pre")).appendChild(el("code", "", sql));
        bubble.appendChild(details);
    }

    function setBusy(busy) {
        input.disabled = busy;
        sendBtn.disabled = busy;
        voice.setBusy(busy);
    }

    // POST JSON and always return {ok, data}, even if the server sent HTML or nothing.
    async function postJson(url, body) {
        try {
            const res = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body || {}),
            });
            let data = {};
            try { data = await res.json(); } catch (e) { data = { error: "Unexpected response from the server." }; }
            return { ok: res.ok, data: data };
        } catch (e) {
            return { ok: false, data: { error: "Couldn't reach Spendly. Is the app still running?" } };
        }
    }

    // A new message replaces all old previews on the server, so disable their buttons here too.
    function closeOldCards() {
        chat.querySelectorAll(".ai-card button").forEach(function (b) { b.disabled = true; });
    }

    // Reload the totals, filters, categories and transactions without losing the chat:
    // fetch this same page (keeping any date filter) and swap in the new #profile-data.
    async function refreshProfileData() {
        try {
            const res = await fetch(window.location.href);
            const html = await res.text();
            const fresh = new DOMParser().parseFromString(html, "text/html").getElementById("profile-data");
            const current = document.getElementById("profile-data");
            if (fresh && current) {
                current.replaceWith(fresh);
                if (window.spendlyEnhance) window.spendlyEnhance(fresh);  // count-up, close buttons
            }
        } catch (e) {
            window.location.reload();
        }
    }

    function expenseTable(rows) {
        const table = el("table", "transactions ai-preview-table");
        const headRow = el("tr");
        ["Date", "Category", "Description", "Amount"].forEach(function (h, i) {
            headRow.appendChild(el("th", i === 3 ? "num" : "", h));
        });
        table.appendChild(el("thead")).appendChild(headRow);
        const tbody = table.appendChild(el("tbody"));
        rows.forEach(function (r) {
            tbody.appendChild(expenseTr(r));
        });
        const wrap = el("div", "table-wrap");
        wrap.appendChild(table);
        return wrap;
    }

    function expenseTr(r) {
        const tr = el("tr");
        tr.appendChild(el("td", "", r.date));
        tr.appendChild(el("td")).appendChild(el("span", "tag", r.category));
        tr.appendChild(el("td", "", r.description));
        tr.appendChild(el("td", "num", r.amount));
        return tr;
    }

    // ---------- Cards ----------

    // Confirm / Cancel buttons. Every click goes to the server, which re-checks everything.
    function confirmButtons(card, confirmText, confirmClass) {
        const actions = el("div", "ai-preview-actions");
        const confirmBtn = button(confirmText, confirmClass);
        const cancelBtn = button("Cancel", "btn-secondary");
        actions.append(confirmBtn, cancelBtn);

        async function finish(url) {
            confirmBtn.disabled = cancelBtn.disabled = true;
            const result = await postJson(url, { action_id: card.id });
            if (result.ok) {
                addBubble("bot", result.data.message, kindClass[result.data.kind] || "ai-msg-note");
                speak(result.data.message);
                if (url === urls.confirm) refreshProfileData();
            } else {
                addBubble("bot", result.data.error || "Something went wrong.", "ai-msg-error");
            }
        }
        confirmBtn.addEventListener("click", function () { finish(urls.confirm); });
        cancelBtn.addEventListener("click", function () { finish(urls.cancel); });
        return actions;
    }

    function addCard(card) {
        const wrap = el("div", "ai-msg ai-msg-bot ai-card");
        wrap.appendChild(el("p", "ai-preview-title",
            card.rows.length === 1 ? "Add this expense?" : "Add these " + card.rows.length + " expenses?"));
        wrap.appendChild(expenseTable(card.rows));
        wrap.appendChild(confirmButtons(card, "Confirm"));
        return wrap;
    }

    function editCard(card) {
        const wrap = el("div", "ai-msg ai-msg-bot ai-card");
        wrap.appendChild(el("p", "ai-preview-title", "Edit this expense?"));
        wrap.appendChild(expenseTable([card.row]));
        const list = el("ul", "ai-changes");
        card.changes.forEach(function (c) {
            const li = el("li");
            li.appendChild(el("span", "ai-change-field", c.field + ": "));
            li.appendChild(el("span", "ai-change-before", c.before));
            li.appendChild(el("span", "ai-change-arrow", " → "));
            li.appendChild(el("span", "ai-change-after", c.after));
            list.appendChild(li);
        });
        wrap.appendChild(list);
        wrap.appendChild(confirmButtons(card, "Confirm"));
        return wrap;
    }

    function deleteCard(card) {
        const wrap = el("div", "ai-msg ai-msg-bot ai-card ai-card-danger");
        wrap.appendChild(el("p", "ai-preview-title", "Delete this expense? This can't be undone."));
        wrap.appendChild(expenseTable([card.row]));
        wrap.appendChild(confirmButtons(card, "Confirm delete", "btn-danger"));
        return wrap;
    }

    // Several expenses matched: list them all and let the user pick. The AI never guesses.
    function chooseCard(card) {
        const wrap = el("div", "ai-msg ai-msg-bot ai-card");
        const what = card.op === "delete" ? "delete" : "edit" + (card.wanted ? " (" + card.wanted + ")" : "");
        wrap.appendChild(el("p", "ai-preview-title",
            card.options.length + " expenses match. Which one do you want to " + what + "?"));

        const list = el("ul", "ai-choices");
        const buttons = [];
        card.options.forEach(function (option) {
            const li = el("li", "ai-choice");
            const r = option.row;
            li.appendChild(el("span", "ai-choice-text",
                r.date + " · " + r.category + " · " + r.description + " · " + r.amount));
            const selectBtn = button("Select", "btn-secondary");
            buttons.push(selectBtn);
            selectBtn.addEventListener("click", async function () {
                buttons.forEach(function (b) { b.disabled = true; });
                cancelBtn.disabled = true;
                const result = await postJson(urls.select, { action_id: card.id, expense_id: option.expense_id });
                if (result.ok) {
                    showCard(result.data.card);
                } else {
                    addBubble("bot", result.data.error || "Something went wrong.", "ai-msg-error");
                }
            });
            li.appendChild(selectBtn);
            list.appendChild(li);
        });
        wrap.appendChild(list);

        const cancelBtn = button("Cancel", "btn-secondary");
        cancelBtn.addEventListener("click", async function () {
            buttons.forEach(function (b) { b.disabled = true; });
            cancelBtn.disabled = true;
            const result = await postJson(urls.cancel, { action_id: card.id });
            addBubble("bot", result.data.message || result.data.error, "ai-msg-note");
        });
        const actions = el("div", "ai-preview-actions");
        actions.appendChild(cancelBtn);
        wrap.appendChild(actions);
        return wrap;
    }

    function budgetCard(card) {
        const wrap = el("div", "ai-msg ai-msg-bot ai-card");
        wrap.appendChild(el("p", "ai-preview-title", "Set this budget?"));
        const list = el("ul", "ai-changes");
        const li = el("li");
        li.appendChild(el("span", "ai-change-field", card.name + ": "));
        li.appendChild(el("span", "ai-change-before", card.before));
        li.appendChild(el("span", "ai-change-arrow", " → "));
        li.appendChild(el("span", "ai-change-after", card.after));
        list.appendChild(li);
        wrap.appendChild(list);
        wrap.appendChild(confirmButtons(card, "Confirm"));
        return wrap;
    }

    function incomeCard(card) {
        const wrap = el("div", "ai-msg ai-msg-bot ai-card");
        wrap.appendChild(el("p", "ai-preview-title",
            card.rows.length === 1 ? "💵 Add this income?" : "💵 Add these " + card.rows.length + " income entries?"));
        const table = el("table", "transactions ai-preview-table");
        const headRow = el("tr");
        ["Date", "Source", "Description", "Amount"].forEach(function (h, i) {
            headRow.appendChild(el("th", i === 3 ? "num" : "", h));
        });
        table.appendChild(el("thead")).appendChild(headRow);
        const tbody = table.appendChild(el("tbody"));
        card.rows.forEach(function (r) {
            const tr = el("tr");
            tr.appendChild(el("td", "", r.date));
            tr.appendChild(el("td")).appendChild(el("span", "tag tag-income", r.source));
            tr.appendChild(el("td", "", r.description));
            tr.appendChild(el("td", "num money-positive", "+" + r.amount));
            tbody.appendChild(tr);
        });
        const wrapTable = el("div", "table-wrap");
        wrapTable.appendChild(table);
        wrap.appendChild(wrapTable);
        wrap.appendChild(confirmButtons(card, "Confirm"));
        return wrap;
    }

    const builders = { add: addCard, edit: editCard, delete: deleteCard, choose: chooseCard, budget: budgetCard,
                       add_income: incomeCard };

    function showCard(card) {
        const build = builders[card.type];
        if (!build) return;
        chat.appendChild(build(card));
        scrollDown();
    }

    // ---------- Chat history (remembered on the server per user) ----------

    async function loadHistory() {
        try {
            const res = await fetch(urls.history);
            if (!res.ok) return;
            const data = await res.json();
            (data.messages || []).forEach(function (m) {
                const bubble = addBubble(m.role === "user" ? "user" : "bot", m.text, kindClass[m.kind] || "");
                if (m.sql) addSqlToggle(bubble, m.sql);
            });
            // Previews still waiting for Confirm come back too.
            (data.cards || []).forEach(showCard);
        } catch (e) {
            // No history is fine: the chat just starts empty.
        }
    }

    newChatBtn.addEventListener("click", async function () {
        newChatBtn.disabled = true;
        const result = await postJson(urls.newChat, {});
        newChatBtn.disabled = false;
        if (result.ok) {
            chat.replaceChildren();
            input.focus();
        } else {
            addBubble("bot", result.data.error || "Couldn't start a new chat.", "ai-msg-error");
        }
    });

    loadHistory();

    // ---------- Settings (remembered in this browser only) ----------

    function loadSetting(name) {
        try { return localStorage.getItem("spendly:" + name) === "on"; } catch (e) { return false; }
    }

    function saveSetting(name, on) {
        try { localStorage.setItem("spendly:" + name, on ? "on" : "off"); } catch (e) { /* private mode: fine */ }
    }

    // ---------- Reply in Hindi (off by default) ----------
    // Typed Hindi/Hinglish is always understood; this only changes the language of replies.
    // The server keeps every ₹ amount and date exactly as in the English reply.

    const replyHindiBox = document.getElementById("ai-reply-hindi");
    if (replyHindiBox) {
        replyHindiBox.checked = loadSetting("reply-hindi");
        replyHindiBox.addEventListener("change", function () { saveSetting("reply-hindi", replyHindiBox.checked); });
    }

    // ---------- 🔊 Read replies aloud (browser's built-in speechSynthesis, off by default) ----------

    const speakBtn = document.getElementById("ai-speak");
    const canSpeak = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
    let speakOn = canSpeak && loadSetting("speak-replies");

    function showSpeakState() {
        speakBtn.setAttribute("aria-pressed", speakOn ? "true" : "false");
        speakBtn.firstChild.textContent = speakOn ? "🔊" : "🔈";
        speakBtn.title = speakOn ? "Reading replies aloud (click to stop)" : "Read replies aloud";
    }

    function speak(text) {
        if (!speakOn || !text) return;
        try {
            window.speechSynthesis.cancel();   // only the newest reply
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = /[ऀ-ॿ]/.test(text) ? "hi-IN" : "en-IN";
            window.speechSynthesis.speak(utterance);
        } catch (e) {
            // Reading aloud is a nice extra; the text is already on screen.
        }
    }

    if (speakBtn && canSpeak) {
        speakBtn.hidden = false;
        showSpeakState();
        speakBtn.addEventListener("click", function () {
            speakOn = !speakOn;
            saveSetting("speak-replies", speakOn);
            if (!speakOn) window.speechSynthesis.cancel();
            showSpeakState();
        });
    }

    // ---------- 🎤 Voice input ----------
    // Speech becomes TEXT in the box. It is not sent unless the user presses Send
    // (or turned on "Auto-send"), and every change still needs the Confirm button.

    const voice = (function () {
        const micBtn = document.getElementById("ai-mic");
        const status = document.getElementById("ai-voice-status");
        const autoSend = document.getElementById("ai-autosend");
        const langBtn = document.getElementById("ai-voice-lang");
        const MAX_SECONDS = 30;
        const MAX_BYTES = 5 * 1024 * 1024;
        const noop = { setBusy: function () {} };
        if (!micBtn || !status) return noop;

        if (autoSend) {
            autoSend.checked = loadSetting("auto-send");   // off unless the user turned it on
            autoSend.addEventListener("change", function () { saveSetting("auto-send", autoSend.checked); });
        }

        // EN / हिंदी toggle. The server never guesses the language: it uses this choice.
        // हिंदी speech is translated to English, so the assistant always gets English text.
        const LANGS = {
            en: { label: "EN", title: "Voice language: English (click for Hindi)" },
            hi: { label: "हिंदी", title: "Voice language: Hindi, translated to English (click for English)" },
        };
        let voiceLang = "en";
        try { if (localStorage.getItem("spendly:voice-lang") === "hi") voiceLang = "hi"; } catch (e) { /* fine */ }

        function showLang() {
            if (!langBtn) return;
            langBtn.firstChild.textContent = LANGS[voiceLang].label;
            langBtn.dataset.lang = voiceLang;
            langBtn.title = LANGS[voiceLang].title;
        }

        if (langBtn) {
            showLang();
            langBtn.addEventListener("click", function () {
                voiceLang = voiceLang === "en" ? "hi" : "en";
                try { localStorage.setItem("spendly:voice-lang", voiceLang); } catch (e) { /* fine */ }
                showLang();
            });
        }

        const supported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
        let recorder = null;
        let stream = null;
        let chunks = [];
        let timer = null;
        let stopTimer = null;
        let started = 0;
        let working = false;      // recording or transcribing
        let chatBusy = false;
        let modelReady = null;    // null = not asked the server yet

        function setStatus(text, kind) {
            status.textContent = text || "";
            status.className = "ai-voice-status" + (kind ? " is-" + kind : "");
        }

        function refreshButton() {
            const listening = !!recorder;
            micBtn.classList.toggle("is-listening", listening);
            micBtn.setAttribute("aria-pressed", listening ? "true" : "false");
            micBtn.title = listening ? "Stop recording" : "Speak your message";
            micBtn.disabled = !listening && (working || chatBusy);
            if (langBtn) langBtn.disabled = working || listening;   // the language can't change mid-recording
        }

        function pickType() {
            const types = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/webm", "audio/mp4"];
            for (let i = 0; i < types.length; i++) {
                if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(types[i])) return types[i];
            }
            return "";
        }

        function extensionFor(type) {
            if (type.indexOf("ogg") !== -1) return "ogg";
            if (type.indexOf("mp4") !== -1) return "m4a";
            return "webm";
        }

        function clock(seconds) {
            return "0:" + String(seconds).padStart(2, "0");
        }

        function releaseMic() {
            clearInterval(timer);
            clearTimeout(stopTimer);
            if (stream) stream.getTracks().forEach(function (t) { t.stop(); });   // mic light goes off
            stream = null;
        }

        function done(message, kind) {
            working = false;
            refreshButton();
            setStatus(message, kind);
        }

        async function start() {
            if (!window.isSecureContext) {
                setStatus("Voice needs the app opened at http://localhost:5001. You can still type.", "error");
                return;
            }
            working = true;
            refreshButton();
            try {
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            } catch (e) {
                const blocked = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
                const missing = e && (e.name === "NotFoundError" || e.name === "OverconstrainedError");
                done(blocked
                    ? "Microphone access is blocked. Allow it from the icon in the address bar to use voice. You can still type."
                    : missing ? "No microphone found. You can still type."
                    : "Couldn't start the microphone. You can still type.", "error");
                return;
            }
            if (speakOn) window.speechSynthesis.cancel();   // don't record our own voice

            const type = pickType();
            try {
                recorder = type ? new MediaRecorder(stream, { mimeType: type }) : new MediaRecorder(stream);
            } catch (e) {
                recorder = null;
                releaseMic();
                done("This browser can't record audio. You can still type.", "error");
                return;
            }
            chunks = [];
            recorder.addEventListener("dataavailable", function (event) {
                if (event.data && event.data.size) chunks.push(event.data);
            });
            recorder.addEventListener("stop", finish);
            recorder.start();
            started = Date.now();
            setStatus("Listening... " + clock(0) + " (click 🎤 to stop)", "listening");
            timer = setInterval(function () {
                const seconds = Math.min(MAX_SECONDS, Math.floor((Date.now() - started) / 1000));
                setStatus("Listening... " + clock(seconds) + " (click 🎤 to stop)", "listening");
            }, 250);
            stopTimer = setTimeout(stop, MAX_SECONDS * 1000);   // auto-stop after 30 seconds
            refreshButton();
        }

        function stop() {
            if (recorder && recorder.state !== "inactive") recorder.stop();   // fires "stop" → finish()
        }

        async function finish() {
            const type = (recorder && recorder.mimeType) || "audio/webm";
            recorder = null;
            releaseMic();
            refreshButton();
            const clip = new Blob(chunks, { type: type });
            chunks = [];

            if (!clip.size) {
                done("I didn't catch that, try again.", "error");
                return;
            }
            if (clip.size > MAX_BYTES) {
                done("That recording is too large. Please keep it under 30 seconds.", "error");
                return;
            }

            if (modelReady === null) {
                try {
                    const res = await fetch(urls.voiceStatus);
                    modelReady = res.ok ? (await res.json()).model_ready : true;
                } catch (e) { modelReady = true; }
            }
            setStatus(modelReady ? "Transcribing..." : "Downloading voice model (first time only)...", "");

            const body = new FormData();
            body.append("audio", clip, "speech." + extensionFor(type));
            body.append("lang", voiceLang);
            let ok = false;
            let data = {};
            try {
                const res = await fetch(urls.voice, { method: "POST", body: body });
                ok = res.ok;
                try { data = await res.json(); } catch (e) { data = { error: "Unexpected response from the server." }; }
            } catch (e) {
                data = { error: "Couldn't reach Spendly. Is the app still running?" };
            }

            if (!ok) {
                done(data.detail || data.error || "Voice input failed. You can still type.", "error");
                return;
            }
            modelReady = true;
            input.value = data.text;   // a plain value: the transcript can't inject HTML
            done("Check the text, fix anything misheard, then press Send.", "");
            if (autoSend && autoSend.checked && !chatBusy) {
                setStatus("", "");
                form.requestSubmit();
            } else {
                input.focus();
                input.setSelectionRange(input.value.length, input.value.length);
            }
        }

        if (!supported) {
            micBtn.disabled = true;
            micBtn.title = "Voice input isn't supported in this browser";
            return noop;
        }

        micBtn.addEventListener("click", function () {
            if (recorder) stop();
            else if (!working) start();
        });

        // Typing clears the voice hint or error.
        input.addEventListener("input", function () {
            if (!working) setStatus("", "");
        });

        return {
            setBusy: function (busy) {
                chatBusy = busy;
                refreshButton();
            },
        };
    })();

    // ---------- Sending a message ----------

    form.addEventListener("submit", async function (event) {
        event.preventDefault();
        const message = input.value.trim();
        if (!message) return;

        closeOldCards();
        addBubble("user", message);
        input.value = "";
        setBusy(true);
        const thinking = addBubble("bot", "Thinking...", "ai-msg-thinking");

        const replyLang = replyHindiBox && replyHindiBox.checked ? "hinglish" : "en";
        const result = await postJson(urls.ask, { message: message, reply_lang: replyLang });
        thinking.remove();
        setBusy(false);
        input.focus();

        // A Hindi/Hinglish message was translated: show what the assistant understood.
        if (result.ok && result.data.understood_as) {
            addBubble("bot", result.data.understood_as, "ai-msg-translation");
        }
        if (!result.ok) {
            const bubble = addBubble("bot", result.data.error || "Something went wrong.", "ai-msg-error");
            if (result.data.detail) bubble.appendChild(el("small", "ai-msg-detail", result.data.detail));
            return;
        }
        if (result.data.reply) {
            // Refused by a guardrail (off-topic / prompt injection): shown as a note.
            const refused = result.data.guard === "off_topic" || result.data.guard === "injection";
            const bubble = addBubble("bot", result.data.reply, refused ? "ai-msg-note" : "");
            if (result.data.sql) addSqlToggle(bubble, result.data.sql);
            speak(result.data.reply);
        }
        // Cards come from our code, so they're the truth. False "added/updated/deleted"
        // claims in the AI's text are replaced on the server (see ai_agent.py).
        (result.data.cards || []).forEach(showCard);
    });
})();
