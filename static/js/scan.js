// Receipt scanning page (vanilla JS): file preview, drag & drop, loading state,
// and the Original / Cleaned (OpenCV) image toggle.

(function () {
    const MAX_BYTES = 5 * 1024 * 1024;

    // ---------- Upload step ----------
    const form = document.getElementById("scan-form");
    if (form) {
        const input = document.getElementById("receipt");
        const zone = document.getElementById("drop-zone");
        const preview = document.getElementById("drop-preview");
        const name = document.getElementById("drop-name");
        const submit = document.getElementById("scan-submit");
        const loading = document.getElementById("scan-loading");

        function showFile(file) {
            if (!file) return;
            name.textContent = file.name + " · " + (file.size / 1024 / 1024).toFixed(1) + " MB";
            if (file.size > MAX_BYTES) {
                name.textContent += " — too large (max 5 MB)";
                zone.classList.add("has-error");
            } else {
                zone.classList.remove("has-error");
            }
            zone.classList.add("has-file");
            if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) {
                // Browsers can't show a PDF in an <img>; the server draws page 1 instead.
                preview.hidden = true;
                name.textContent = "📄 " + name.textContent + " · page 1 will be read";
                return;
            }
            const reader = new FileReader();
            reader.onload = function () { preview.src = reader.result; preview.hidden = false; };
            reader.readAsDataURL(file);
        }

        input.addEventListener("change", function () { showFile(input.files[0]); });

        ["dragenter", "dragover"].forEach(function (type) {
            zone.addEventListener(type, function (e) { e.preventDefault(); zone.classList.add("dragging"); });
        });
        ["dragleave", "drop"].forEach(function (type) {
            zone.addEventListener(type, function (e) { e.preventDefault(); zone.classList.remove("dragging"); });
        });
        zone.addEventListener("drop", function (e) {
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                showFile(input.files[0]);
            }
        });

        form.addEventListener("submit", function (e) {
            const file = input.files[0];
            if (file && file.size > MAX_BYTES) {
                e.preventDefault();
                return;
            }
            submit.disabled = true;
            loading.hidden = false;   // "Reading your receipt..." while the server works
        });
    }

    // ---------- Review step: Original / Cleaned toggle ----------
    document.querySelectorAll(".toggle-btn").forEach(function (button) {
        button.addEventListener("click", function () {
            const show = button.dataset.show;
            document.querySelectorAll(".toggle-btn").forEach(function (b) {
                const on = b === button;
                b.classList.toggle("active", on);
                b.setAttribute("aria-pressed", on ? "true" : "false");
            });
            document.querySelectorAll(".receipt-img").forEach(function (img) {
                img.hidden = img.dataset.image !== show;
            });
        });
    });
})();
