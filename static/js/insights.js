// "Get insights" button on the analytics page (vanilla JS).
// The server runs the read-only insights workflow; this file only shows the result.
// All text is inserted with textContent, never innerHTML.

(function () {
    const panel = document.getElementById("insights");
    if (!panel) return;
    const button = document.getElementById("insights-button");
    const output = document.getElementById("insights-output");

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function showMessage(text, className, detail) {
        const box = el("p", "insights-message " + className, text);
        if (detail) box.appendChild(el("small", "insights-detail", detail));
        output.replaceChildren(box);
    }

    function showResult(data) {
        const card = el("div", "insights-card");

        if (data.key_numbers && data.key_numbers.length) {
            const numbers = el("div", "insights-numbers");
            data.key_numbers.forEach(function (n) {
                const item = el("div", "insights-number");
                item.appendChild(el("span", "insights-number-label", n.label));
                item.appendChild(el("span", "insights-number-value", n.value));
                if (n.note) item.appendChild(el("span", "insights-number-note", n.note));   // e.g. "1–7 Nov vs 1–7 Oct"
                numbers.appendChild(item);
            });
            card.appendChild(numbers);
        }

        card.appendChild(el("p", "insights-summary", data.summary));
        card.appendChild(el("h3", "insights-tips-title", "3 ways to save"));
        const list = el("ol", "insights-tips");
        data.tips.forEach(function (tip) { list.appendChild(el("li", "", tip)); });
        card.appendChild(list);

        if (data.source === "fallback") {
            card.appendChild(el("p", "insights-note",
                "The AI's answer didn't match your numbers, so this summary was written directly from your data."));
        }
        output.replaceChildren(card);
    }

    button.addEventListener("click", async function () {
        button.disabled = true;
        showMessage("Analysing your spending...", "insights-loading");
        const skeleton = el("div", "insights-skeleton");
        skeleton.setAttribute("aria-hidden", "true");
        for (let i = 0; i < 4; i++) skeleton.appendChild(el("span"));
        output.appendChild(skeleton);
        try {
            const res = await fetch(panel.dataset.url, { method: "POST" });
            let data = {};
            try { data = await res.json(); } catch (e) { data = {}; }

            if (res.ok && data.status === "ok") {
                showResult(data);
            } else if (res.ok && data.status === "not_enough") {
                showMessage(data.message, "insights-empty");
            } else if (data.status === "unavailable") {
                showMessage("Insights unavailable", "insights-error", data.detail);
            } else {
                showMessage(data.error || "Something went wrong. Please try again.", "insights-error");
            }
        } catch (e) {
            showMessage("Couldn't reach Spendly. Is the app still running?", "insights-error");
        }
        button.disabled = false;
    });
})();
