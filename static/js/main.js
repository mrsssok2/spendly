// Spendly client-side scripts (vanilla JS).

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Ask for confirmation before submitting any form that has a data-confirm attribute,
// e.g. <form data-confirm="Delete this expense?">. Cancelling stops the submit.
document.addEventListener("submit", function (event) {
    const form = event.target;
    const message = form.dataset.confirm;
    if (message && !window.confirm(message)) {
        event.preventDefault();
    }
});

// ---------- Flash messages: a close button, and success messages fade away ----------
function setUpAlerts(root) {
    root.querySelectorAll(".alert").forEach(function (alert) {
        if (alert.querySelector(".alert-close")) return;
        const close = document.createElement("button");
        close.type = "button";
        close.className = "alert-close";
        close.setAttribute("aria-label", "Dismiss message");
        close.textContent = "×";
        close.addEventListener("click", function () { dismiss(alert); });
        alert.appendChild(close);
        if (alert.classList.contains("alert-success")) {
            setTimeout(function () { dismiss(alert); }, 6000);
        }
    });
}

function dismiss(alert) {
    alert.classList.add("is-leaving");
    alert.addEventListener("animationend", function () { alert.remove(); }, { once: true });
    if (reduceMotion) alert.remove();
}

// ---------- Count-up animation for big numbers ----------
// <p data-count="1234.5">₹1,234.50</p>: animates from ₹0 and always ENDS on the exact text
// the server sent. The browser only animates; it never calculates any totals.
function countUp(root) {
    root.querySelectorAll("[data-count]").forEach(function (el) {
        const target = parseFloat(el.dataset.count);
        const finalText = el.textContent;
        if (reduceMotion || !isFinite(target) || target <= 0) return;
        const hasDecimals = /\.\d\d/.test(finalText);
        const prefix = finalText.trim().startsWith("₹") ? "₹" : "";
        const duration = 900;
        const start = performance.now();
        function frame(now) {
            const t = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - t, 3);
            el.textContent = prefix + (target * eased).toLocaleString("en-IN", {
                minimumFractionDigits: hasDecimals ? 2 : 0, maximumFractionDigits: hasDecimals ? 2 : 0,
            });
            if (t < 1) requestAnimationFrame(frame);
            else el.textContent = finalText;
        }
        requestAnimationFrame(frame);
    });
}

// ---------- Light / dark mode toggle ----------
// No saved choice = follow the system setting. Clicking saves "light" or "dark" (theme.js
// applies it on the next page load before anything is drawn).
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

function currentTheme() {
    const chosen = document.documentElement.getAttribute("data-theme");
    if (chosen === "light" || chosen === "dark") return chosen;
    return systemDark.matches ? "dark" : "light";
}

function announceTheme() {
    const theme = currentTheme();
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#0f1120" : "#4f46e5");
    const toggle = document.getElementById("theme-toggle");
    if (toggle) {
        toggle.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
        toggle.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
    }
    // Charts (analytics.js) listen for this to recolour themselves.
    document.dispatchEvent(new CustomEvent("spendly:themechange", { detail: { theme: theme } }));
}

const themeToggle = document.getElementById("theme-toggle");
if (themeToggle) {
    themeToggle.addEventListener("click", function () {
        const next = currentTheme() === "dark" ? "light" : "dark";
        const root = document.documentElement;
        if (!reduceMotion) {
            root.classList.add("theme-switching");
            setTimeout(function () { root.classList.remove("theme-switching"); }, 400);
        }
        root.setAttribute("data-theme", next);
        try { localStorage.setItem("spendly-theme", next); } catch (e) { /* not saved; still switches */ }
        announceTheme();
    });
}

// Follow the system when the user hasn't picked a theme themselves.
systemDark.addEventListener("change", function () {
    if (!document.documentElement.hasAttribute("data-theme")) announceTheme();
});
announceTheme();

// Pages that swap in new HTML (e.g. after an AI change) call this again on the new part.
window.spendlyEnhance = function (root) {
    setUpAlerts(root);
    countUp(root);
};

window.spendlyEnhance(document);
