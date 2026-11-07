// Analytics page charts (vanilla JS + Chart.js from the CDN).
// The numbers come from SQLite via Flask as JSON. This file only DRAWS them and formats
// them as rupees: it never adds anything up.

(function () {
    const dataTag = document.getElementById("analytics-data");
    if (!dataTag) return;
    const charts = JSON.parse(dataTag.textContent);

    if (typeof Chart === "undefined") {
        // The CDN couldn't be reached (offline?). Say so instead of leaving blank boxes.
        document.querySelectorAll(".chart-box").forEach(function (box) {
            box.replaceWith(Object.assign(document.createElement("p"), {
                className: "chart-empty",
                textContent: "Charts couldn't load (no internet connection to the chart library).",
            }));
        });
        return;
    }

    const BRAND = "#4f46e5";
    // Same colours as the category tags and bars on the profile page (profile.css).
    const CATEGORY_COLOURS = {
        Food: "#f97316", Transport: "#0ea5e9", Bills: "#eab308", Health: "#10b981",
        Entertainment: "#ec4899", Shopping: "#8b5cf6", Other: "#64748b",
    };

    // Vertical gradient for bars / the area under the line.
    function gradient(ctx, area, top, bottom) {
        if (!area) return top;
        const g = ctx.createLinearGradient(0, area.bottom, 0, area.top);
        g.addColorStop(0, bottom);
        g.addColorStop(1, top);
        return g;
    }

    function rupees(value) {
        return "₹" + Number(value).toLocaleString("en-IN", { maximumFractionDigits: 2 });
    }

    // Colours that change with light/dark mode come from the CSS variables in style.css.
    function cssVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    function applyThemeColours() {
        Chart.defaults.color = cssVar("--muted");
        Chart.defaults.borderColor = cssVar("--divider");
    }

    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    applyThemeColours();
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.animation.duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 1100;
    Chart.defaults.animation.easing = "easeOutQuart";
    Chart.defaults.plugins.tooltip.backgroundColor = "#1d2140";
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 10;

    const rupeeAxis = { beginAtZero: true, ticks: { callback: rupees } };
    const rupeeTooltip = { callbacks: { label: function (ctx) { return " " + rupees(ctx.parsed.y ?? ctx.parsed); } } };

    const drawn = [];

    function draw(id, config) {
        const canvas = document.getElementById(id);
        if (canvas) drawn.push(new Chart(canvas, config));
    }

    // Light/dark toggle (main.js): recolour axes, grid lines and the doughnut's gaps, then redraw.
    document.addEventListener("spendly:themechange", function () {
        applyThemeColours();
        const muted = cssVar("--muted"), divider = cssVar("--divider");
        drawn.forEach(function (chart) {
            Object.values(chart.options.scales || {}).forEach(function (scale) {
                scale.ticks.color = muted;
                scale.grid.color = divider;
                if (scale.border) scale.border.color = divider;
            });
            if (chart.options.plugins.legend && chart.options.plugins.legend.labels) {
                chart.options.plugins.legend.labels.color = muted;
            }
            if (chart.config.type === "doughnut") {
                chart.data.datasets[0].borderColor = cssVar("--surface");
            }
            if (chart.config.type === "line") {
                chart.data.datasets[0].borderColor = cssVar("--primary");
            }
            chart.update("none");
        });
    });

    draw("chart-monthly", {
        type: "bar",
        data: {
            labels: charts.monthly.labels,
            datasets: [{
                data: charts.monthly.totals,
                backgroundColor: function (c) { return gradient(c.chart.ctx, c.chart.chartArea, "#7c3aed", "#818cf8"); },
                hoverBackgroundColor: BRAND,
                borderRadius: 10,
                borderSkipped: false,
                maxBarThickness: 46,
            }],
        },
        options: {
            scales: { y: rupeeAxis, x: { grid: { display: false } } },
            plugins: { legend: { display: false }, tooltip: rupeeTooltip },
        },
    });

    if (charts.income_vs_expenses) {
        draw("chart-income", {
            type: "bar",
            data: {
                labels: charts.income_vs_expenses.labels,
                datasets: [
                    { label: "Income", data: charts.income_vs_expenses.income, backgroundColor: "#22c55e",
                      borderRadius: 8, borderSkipped: false, maxBarThickness: 34 },
                    { label: "Expenses",
                      data: charts.income_vs_expenses.expenses,
                      backgroundColor: function (c) { return gradient(c.chart.ctx, c.chart.chartArea, "#7c3aed", "#818cf8"); },
                      borderRadius: 8, borderSkipped: false, maxBarThickness: 34 },
                ],
            },
            options: {
                scales: { y: rupeeAxis, x: { grid: { display: false } } },
                plugins: {
                    legend: { position: "top", align: "end", labels: { usePointStyle: true, pointStyle: "circle" } },
                    tooltip: { callbacks: { label: function (ctx) { return " " + ctx.dataset.label + ": " + rupees(ctx.parsed.y); } } },
                },
            },
        });
    }

    draw("chart-categories", {
        type: "doughnut",
        data: {
            labels: charts.categories.labels,
            datasets: [{
                data: charts.categories.totals,
                backgroundColor: charts.categories.labels.map(function (name) { return CATEGORY_COLOURS[name] || "#94a3b8"; }),
                borderColor: cssVar("--surface"),   // gaps between slices match the card
                borderWidth: 3,
                hoverOffset: 10,
            }],
        },
        options: {
            cutout: "64%",
            animation: { animateRotate: true, animateScale: true },
            plugins: {
                legend: { position: "bottom", labels: { usePointStyle: true, pointStyle: "circle", padding: 14 } },
                tooltip: { callbacks: { label: function (ctx) { return " " + ctx.label + ": " + rupees(ctx.parsed); } } },
            },
        },
    });

    // This month: actual running total vs the forecast (forecast.py). Every value arrives ready
    // from the server; this only draws lines. Colours are CSS tokens, so dark mode works.
    if (charts.forecast) {
        const f = charts.forecast;
        const forecastColours = function () {
            return { actual: cssVar("--primary"), forecast: cssVar("--accent"), band: cssVar("--primary-soft"),
                     budget: cssVar("--danger") };
        };
        const c = forecastColours();
        const datasets = [
            { label: "Spent so far", data: f.actual, borderColor: c.actual, backgroundColor: c.actual,
              borderWidth: 2.5, pointRadius: 0, pointHoverRadius: 5, tension: 0, spanGaps: false },
            { label: "Most likely", data: f.expected, borderColor: c.forecast, borderDash: [6, 5],
              borderWidth: 2, pointRadius: 0, pointHoverRadius: 5, spanGaps: false },
            { label: "Likely range (low)", data: f.low, borderColor: "transparent", pointRadius: 0,
              pointHoverRadius: 0, spanGaps: false },
            { label: "Likely range", data: f.high, borderColor: "transparent", backgroundColor: c.band,
              fill: "-1", pointRadius: 0, pointHoverRadius: 0, spanGaps: false },
        ];
        if (f.budget) {
            datasets.push({ label: "Budget", data: f.labels.map(function () { return f.budget; }),
                            borderColor: c.budget, borderDash: [2, 4], borderWidth: 1.5, pointRadius: 0,
                            pointHoverRadius: 0 });
        }
        draw("chart-forecast", {
            type: "line",
            data: { labels: f.labels, datasets: datasets },
            options: {
                interaction: { mode: "index", intersect: false },
                scales: { y: rupeeAxis, x: { grid: { display: false } } },
                plugins: {
                    legend: { position: "top", align: "end",
                              labels: { usePointStyle: true, pointStyle: "line",
                                        filter: function (item) { return item.text !== "Likely range (low)"; } } },
                    tooltip: { filter: function (item) { return item.raw !== null && item.dataset.label !== "Likely range (low)"; },
                               callbacks: { label: function (ctx) {
                                   if (ctx.dataset.label === "Likely range") {
                                       return " Likely range: " + rupees(f.low[ctx.dataIndex]) + " – " + rupees(ctx.parsed.y);
                                   }
                                   return " " + ctx.dataset.label + ": " + rupees(ctx.parsed.y);
                               } } },
                },
            },
        });
        document.addEventListener("spendly:themechange", function () {
            const chart = drawn.find(function (ch) { return ch.canvas.id === "chart-forecast"; });
            if (!chart) return;
            const col = forecastColours();
            chart.data.datasets[0].borderColor = col.actual;
            chart.data.datasets[0].backgroundColor = col.actual;
            chart.data.datasets[1].borderColor = col.forecast;
            chart.data.datasets[3].backgroundColor = col.band;
            if (chart.data.datasets[4]) chart.data.datasets[4].borderColor = col.budget;
            chart.update("none");
        });
    }

    draw("chart-daily", {
        type: "line",
        data: {
            labels: charts.daily.labels,
            datasets: [{
                data: charts.daily.totals, borderColor: cssVar("--primary") || BRAND, borderWidth: 2.5,
                backgroundColor: function (c) {
                    return gradient(c.chart.ctx, c.chart.chartArea, "rgba(124, 58, 237, 0.28)", "rgba(124, 58, 237, 0)");
                },
                fill: true, cubicInterpolationMode: "monotone",  // smooth, never dips below ₹0
                pointRadius: 0, pointHoverRadius: 6,
                pointHoverBackgroundColor: BRAND, pointHoverBorderColor: "#fff", pointHoverBorderWidth: 2,
            }],
        },
        options: {
            interaction: { mode: "index", intersect: false },
            scales: { y: rupeeAxis, x: { grid: { display: false } } },
            plugins: { legend: { display: false }, tooltip: rupeeTooltip },
        },
    });
})();
