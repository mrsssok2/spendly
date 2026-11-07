// Runs in <head> BEFORE the page is drawn, so a saved theme is applied without a white flash.
// No saved choice = follow the system setting (handled by CSS).
(function () {
    try {
        const saved = localStorage.getItem("spendly-theme");
        if (saved === "light" || saved === "dark") {
            document.documentElement.setAttribute("data-theme", saved);
        }
    } catch (e) {
        // Storage blocked (private mode etc.): just follow the system setting.
    }
})();
