// ---------------------------------------------------------------------------
// Вкладки
// ---------------------------------------------------------------------------

const tabs = document.querySelectorAll(".tab");
const tabContents = document.querySelectorAll(".tab-content");

tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
        const target = tab.dataset.tab;

        tabs.forEach((item) => {
            item.classList.remove("active");
        });

        tabContents.forEach((item) => {
            item.classList.remove("active");
        });

        tab.classList.add("active");
        document
            .getElementById(target)
            .classList.add("active");
    });
});


// ---------------------------------------------------------------------------
// Настройки API
// ---------------------------------------------------------------------------

// true: ручки заменены заглушками (данные в памяти браузера).
// false: запросы уходят на бэкенд.
const USE_MOCK = false;

const API = {
    uploadApplications: "/v1/applications/upload",
    unscoredApplications: "/v1/applications/unscored",
    batchPredict: "/v1/predict/batch",
    predictions: "/v1/predictions",
    predictionsExport: "/v1/predictions/export",
    review: "/v1/review",
    reviewDecision: (id) =>
        `/v1/review/${encodeURIComponent(id)}/decision`
};

const APPLICATION_COLUMNS = [
    "loan_amnt", "term", "emp_title", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "purpose", "zip_code",
    "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "collections_12_mths_ex_med",
    "application_type", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
    "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal",
    "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "delinq_amnt",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit"
];

// Должен совпадать с SUMMARY_COLUMNS из app.py
const SUMMARY_COLUMNS = [
    "loan_amnt", "term", "purpose", "annual_inc", "dti", "fico_range_low"
];

const UNSCORED_COLUMNS = ["application_id", ...APPLICATION_COLUMNS];

const PREDICTION_COLUMNS = [
    "application_id", "ts", "decision", "score",
    ...SUMMARY_COLUMNS,
    "model_version", "latency_ms", "response_code"
];

const REVIEW_COLUMNS = [
    "application_id", "ts", "decision", "score", "model_version"
];


// ---------------------------------------------------------------------------
// Общие функции
// ---------------------------------------------------------------------------

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));


// В режиме USE_MOCK вместо запроса вызывается mock().
async function request(url, options, mock) {
    if (USE_MOCK) {
        await sleep(400);
        return mock();
    }

    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(
            data.detail
                ? JSON.stringify(data.detail)
                : "Request failed"
        );
    }

    return data;
}


function showError(elementId, message) {
    const element = document.getElementById(
        elementId
    );

    element.textContent = message;
    element.classList.remove("hidden");
}


const showSuccess = showError;


function hideError(elementId) {
    document
        .getElementById(elementId)
        .classList.add("hidden");
}


function setLoading(prefix, isLoading) {
    document
        .getElementById(`${prefix}-loader`)
        .classList.toggle("hidden", !isLoading);

    document.getElementById(
        `${prefix}-submit`
    ).disabled = isLoading;
}


function formatNumber(value) {
    return Number(value).toFixed(4);
}


function formatTimestamp(value) {
    return String(value).replace("T", " ").slice(0, 19);
}


function makeCell(text, className) {
    const cell = document.createElement("td");

    cell.textContent = text;

    if (className) {
        cell.className = className;
    }

    return cell;
}


// Цвет решения: DECLINE красный, REVIEW жёлтый, APPROVE зелёный
function decisionClass(value) {
    return `decision decision-${String(value).toLowerCase()}`;
}


function renderTable(
    headId, bodyId, columns, rows, formatValue, cellClass
) {
    const head = document.getElementById(headId);
    const body = document.getElementById(bodyId);

    head.innerHTML = "";
    body.innerHTML = "";

    const headRow = document.createElement("tr");

    columns.forEach((column) => {
        const th = document.createElement("th");
        th.textContent = column;
        headRow.appendChild(th);
    });

    head.appendChild(headRow);

    if (!rows.length) {
        const row = document.createElement("tr");
        const cell = makeCell("No data", "empty");

        cell.colSpan = columns.length;
        row.appendChild(cell);
        body.appendChild(row);

        return;
    }

    rows.forEach((item) => {
        const row = document.createElement("tr");

        columns.forEach((column) => {
            const value = item[column];

            if (value === null || value === undefined) {
                row.appendChild(makeCell("—", "null"));
            } else {
                row.appendChild(
                    makeCell(
                        formatValue(column, value),
                        cellClass ? cellClass(column, value) : undefined
                    )
                );
            }
        });

        body.appendChild(row);
    });
}


function toCsv(rows, columns) {
    const escape = (value) => {
        if (value === null || value === undefined) {
            return "";
        }

        const text = typeof value === "object"
            ? JSON.stringify(value)
            : String(value);

        return /[",\r\n]/.test(text)
            ? `"${text.replace(/"/g, '""')}"`
            : text;
    };

    const lines = [columns.map(escape).join(",")];

    rows.forEach((row) => {
        lines.push(
            columns.map((column) => escape(row[column])).join(",")
        );
    });

    return "\ufeff" + lines.join("\n");
}


function csvToBlobUrl(csv) {
    return URL.createObjectURL(
        new Blob([csv], { type: "text/csv;charset=utf-8" })
    );
}


function triggerDownload(url, filename) {
    const link = document.createElement("a");

    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
}


function buildQuery(params) {
    const query = new URLSearchParams();

    Object.entries(params).forEach(([key, value]) => {
        if (value) {
            query.set(key, value);
        }
    });

    return query.toString();
}


// ---------------------------------------------------------------------------
// Вкладка Applications: загрузка CSV, необработанные заявки, запуск скоринга
// ---------------------------------------------------------------------------

async function loadUnscored() {
    hideError("unscored-error");
    setLoading("unscored", true);

    try {
        const data = await request(
            API.unscoredApplications,
            {},
            mockFetchUnscored
        );

        document.getElementById(
            "unscored-count"
        ).textContent = data.items.length;

        renderTable(
            "unscored-head",
            "unscored-body",
            UNSCORED_COLUMNS,
            data.items,
            (column, value) => String(value)
        );

    } catch (error) {
        showError("unscored-error", error.message);
    } finally {
        setLoading("unscored", false);
    }
}


document
    .getElementById("unscored-submit")
    .addEventListener("click", loadUnscored);


document
    .getElementById("upload-form")
    .addEventListener("submit", async (event) => {

        event.preventDefault();

        hideError("upload-error");
        hideError("upload-success");

        const fileInput =
            document.getElementById(
                "csv-file"
            );

        if (!fileInput.files.length) {
            showError(
                "upload-error",
                "Select a CSV file."
            );

            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();

        formData.append("file", file);

        setLoading("upload", true);

        try {
            const data = await request(
                API.uploadApplications,
                {
                    method: "POST",
                    body: formData
                },
                () => mockUploadApplications(file)
            );

            let message = `Added ${data.added} applications.`;

            if (data.skipped) {
                message += ` Skipped ${data.skipped}.`;
            }

            showSuccess("upload-success", message);
            event.target.reset();

            await loadUnscored();

        } catch (error) {
            showError(
                "upload-error",
                error.message
            );
        } finally {
            setLoading("upload", false);
        }
    });


// Скоринг всех необработанных заявок
document
    .getElementById("predict-submit")
    .addEventListener("click", async () => {

        hideError("predict-error");
        hideError("predict-success");

        setLoading("predict", true);

        try {
            const data = await request(
                API.batchPredict,
                { method: "POST" },
                mockBatchPredict
            );

            if (!data.total) {
                throw new Error(
                    "There are no unscored applications."
                );
            }

            showSuccess(
                "predict-success",
                `Scored ${data.total}: `
                + `${data.approved} approved, `
                + `${data.review} sent to review, `
                + `${data.declined} declined.`
            );

            // Оценённые заявки уходят из очереди и появляются
            // в predictions, а часть из них в review
            loadUnscored();
            loadPredictions();
            loadReview();

        } catch (error) {
            showError("predict-error", error.message);
        } finally {
            setLoading("predict", false);
        }
    });


// ---------------------------------------------------------------------------
// Вкладка Predictions: таблица и выгрузка в CSV
// ---------------------------------------------------------------------------

let currentPeriod = "week";


function formatPredictionValue(column, value) {
    if (column === "score") {
        return formatNumber(value);
    }

    if (column === "ts") {
        return formatTimestamp(value);
    }

    if (column === "latency_ms") {
        return Number(value).toFixed(1);
    }

    return String(value);
}


function predictionCellClass(column, value) {
    return column === "decision" ? decisionClass(value) : undefined;
}


async function loadPredictions() {
    hideError("pred-error");

    document.getElementById("pred-loader").classList.remove("hidden");

    try {
        const query = buildQuery({ period: currentPeriod });

        const data = await request(
            `${API.predictions}?${query}`,
            {},
            () => mockFetchPredictions(currentPeriod)
        );

        document.getElementById(
            "pred-count"
        ).textContent = data.items.length;

        // Признаки из SUMMARY_COLUMNS приходят вложенным объектом features
        const rows = data.items.map((item) => ({
            ...(item.features || {}),
            ...item
        }));

        renderTable(
            "pred-head",
            "pred-body",
            PREDICTION_COLUMNS,
            rows,
            formatPredictionValue,
            predictionCellClass
        );

    } catch (error) {
        showError("pred-error", error.message);
    } finally {
        document.getElementById("pred-loader").classList.add("hidden");
    }
}


document
    .querySelectorAll("#period-buttons .period")
    .forEach((button) => {
        button.addEventListener("click", () => {
            currentPeriod = button.dataset.period;

            document
                .querySelectorAll("#period-buttons .period")
                .forEach((item) => {
                    item.classList.toggle("active", item === button);
                });

            loadPredictions();
        });
    });


document
    .getElementById("pred-download")
    .addEventListener("click", async (event) => {
        const button = event.currentTarget;

        hideError("pred-error");

        const filename = `predictions_${currentPeriod}.csv`;

        button.disabled = true;

        try {
            if (USE_MOCK) {
                await sleep(400);
                triggerDownload(
                    mockExportPredictions(currentPeriod),
                    filename
                );
            } else {
                triggerDownload(
                    `${API.predictionsExport}?${buildQuery({
                        period: currentPeriod
                    })}`,
                    filename
                );
            }
        } catch (error) {
            showError("pred-error", error.message);
        } finally {
            button.disabled = false;
        }
    });


// ---------------------------------------------------------------------------
// Вкладка Review: заявки с decision = REVIEW
// ---------------------------------------------------------------------------

let reviewItems = [];


function updateReviewCount() {
    const count = reviewItems.length;

    document.getElementById("review-count").textContent = count;
    document.getElementById("review-tab-count").textContent = count;
}


function showEmptyReview() {
    const body = document.getElementById("review-body");
    const row = document.createElement("tr");
    const cell = makeCell("No applications for review", "empty");

    cell.colSpan = REVIEW_COLUMNS.length + 1;
    row.appendChild(cell);
    body.appendChild(row);
}


function renderReviewHead() {
    const head = document.getElementById("review-head");
    const row = document.createElement("tr");

    // Первый столбец под стрелку раскрытия
    ["", ...REVIEW_COLUMNS].forEach((column) => {
        const th = document.createElement("th");
        th.textContent = column;
        row.appendChild(th);
    });

    head.innerHTML = "";
    head.appendChild(row);
}


function formatFeatureValue(value) {
    if (value === null || value === undefined) {
        return "—";
    }

    return typeof value === "number"
        ? String(Number(value.toFixed(4)))
        : String(value);
}


// Диаграмма вкладов: вправо красным (повышает P(default)),
// влево зелёным (понижает). Длина считается от максимального вклада.
function buildImportanceChart(importance) {
    const chart = document.createElement("div");

    chart.className = "importance";

    const maxImpact = Math.max(
        ...importance.map((item) => Math.abs(item.impact)),
        1e-9
    );

    importance.forEach((item) => {
        const row = document.createElement("div");
        row.className = "importance-row";

        const name = document.createElement("span");
        name.className = "importance-name";
        name.textContent = item.feature;

        const value = document.createElement("span");
        value.className = "importance-value";
        value.textContent = formatFeatureValue(item.value);

        const track = document.createElement("div");
        track.className = "bar-track";

        const fill = document.createElement("div");
        const positive = item.impact > 0;

        fill.className = `bar-fill ${positive ? "positive" : "negative"}`;
        fill.style.width = `${(Math.abs(item.impact) / maxImpact) * 50}%`;
        fill.title = item.direction || "";

        track.appendChild(fill);

        const impact = document.createElement("span");
        impact.className = "importance-impact";
        impact.textContent =
            `${positive ? "+" : ""}${item.impact.toFixed(4)}`;

        row.append(name, value, track, impact);
        chart.appendChild(row);
    });

    return chart;
}


function buildReviewActions(item, rows) {
    const actions = document.createElement("div");
    actions.className = "review-actions";

    const label = document.createElement("span");
    label.className = "review-actions-label";
    label.textContent = "Move to:";

    const toggle = document.createElement("div");
    toggle.className = "toggle";
    toggle.setAttribute("role", "group");

    const save = document.createElement("button");
    save.type = "button";
    save.className = "primary-button";
    save.textContent = "Confirm";
    save.disabled = true;

    let chosen = null;

    ["APPROVE", "DECLINE"].forEach((decision) => {
        const option = document.createElement("button");

        option.type = "button";
        option.className =
            `toggle-option toggle-${decision.toLowerCase()}`;
        option.textContent = decision;

        option.addEventListener("click", () => {
            chosen = decision;
            save.disabled = false;

            toggle.querySelectorAll(".toggle-option").forEach((el) => {
                el.classList.toggle("active", el === option);
            });
        });

        toggle.appendChild(option);
    });

    save.addEventListener("click", async () => {
        if (!chosen) {
            return;
        }

        hideError("review-error");
        hideError("review-success");

        save.disabled = true;

        try {
            await request(
                API.reviewDecision(item.application_id),
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ decision: chosen })
                },
                () => mockResolveReview(item.application_id, chosen)
            );

            rows.forEach((row) => row.remove());

            reviewItems = reviewItems.filter(
                (entry) => entry.application_id !== item.application_id
            );

            updateReviewCount();

            if (!reviewItems.length) {
                showEmptyReview();
            }

            showSuccess(
                "review-success",
                `Application ${item.application_id} moved to ${chosen}.`
            );

            // Решение изменилось, обновляем таблицу predictions
            loadPredictions();

        } catch (error) {
            save.disabled = false;
            showError("review-error", error.message);
        }
    });

    actions.append(label, toggle, save);

    return actions;
}


function buildReviewRows(item) {
    const row = document.createElement("tr");

    row.className = "review-row";
    row.tabIndex = 0;
    row.setAttribute("aria-expanded", "false");

    const chevron = makeCell("▸", "chevron");
    row.appendChild(chevron);

    REVIEW_COLUMNS.forEach((column) => {
        const value = item[column];

        if (value === null || value === undefined) {
            row.appendChild(makeCell("—", "null"));
        } else {
            row.appendChild(
                makeCell(
                    formatPredictionValue(column, value),
                    predictionCellClass(column, value)
                )
            );
        }
    });

    const detail = document.createElement("tr");
    detail.className = "detail-row hidden";

    const cell = document.createElement("td");
    cell.colSpan = REVIEW_COLUMNS.length + 1;

    const panel = document.createElement("div");
    panel.className = "detail-panel";

    const title = document.createElement("h3");
    title.textContent = "Main reasons";

    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent =
        "Red bars increase P(default), green bars decrease it. "
        + "Impact is measured in log-odds of the base model.";

    panel.append(title, hint);

    if (item.feature_importance && item.feature_importance.length) {
        panel.appendChild(buildImportanceChart(item.feature_importance));
    } else {
        const empty = document.createElement("p");
        empty.className = "hint";
        empty.textContent = "No feature importance saved.";
        panel.appendChild(empty);
    }

    panel.appendChild(buildReviewActions(item, [row, detail]));

    cell.appendChild(panel);
    detail.appendChild(cell);

    const toggleRow = () => {
        const isHidden = detail.classList.toggle("hidden");

        row.setAttribute("aria-expanded", String(!isHidden));
        chevron.textContent = isHidden ? "▸" : "▾";
    };

    row.addEventListener("click", toggleRow);

    row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            toggleRow();
        }
    });

    return [row, detail];
}


async function loadReview() {
    hideError("review-error");
    setLoading("review", true);

    try {
        const data = await request(
            API.review,
            {},
            mockFetchReview
        );

        reviewItems = data.items;

        renderReviewHead();

        const body = document.getElementById("review-body");
        body.innerHTML = "";

        reviewItems.forEach((item) => {
            buildReviewRows(item).forEach((row) => {
                body.appendChild(row);
            });
        });

        if (!reviewItems.length) {
            showEmptyReview();
        }

        updateReviewCount();

    } catch (error) {
        showError("review-error", error.message);
    } finally {
        setLoading("review", false);
    }
}


document
    .getElementById("review-submit")
    .addEventListener("click", () => {
        hideError("review-success");
        loadReview();
    });


// ---------------------------------------------------------------------------
// Заглушки ручек (используются при USE_MOCK = true)
// ---------------------------------------------------------------------------

const MOCK_TEXT = {
    term: [" 36 months", " 60 months"],
    emp_title: ["teacher", "manager", "driver", "nurse", "engineer"],
    emp_length: ["< 1 year", "3 years", "10+ years"],
    home_ownership: ["RENT", "MORTGAGE", "OWN"],
    verification_status: ["Verified", "Source Verified", "Not Verified"],
    purpose: ["debt_consolidation", "credit_card", "home_improvement"],
    zip_code: ["123xx", "945xx", "606xx"],
    addr_state: ["CA", "NY", "TX", "FL"],
    application_type: ["Individual", "Joint App"]
};

const rnd = (min, max) => min + Math.random() * (max - min);
const rndInt = (min, max) => Math.floor(rnd(min, max + 1));
const pick = (items) => items[rndInt(0, items.length - 1)];

const MOCK_NUMBER = {
    loan_amnt: () => rndInt(10, 350) * 100,
    annual_inc: () => rndInt(25, 150) * 1000,
    dti: () => Number(rnd(2, 35).toFixed(2)),
    fico_range_low: () => rndInt(132, 160) * 5,
    revol_util: () => Number(rnd(0, 100).toFixed(1))
};

// Столбцы, в которых в заглушке иногда встречаются пропуски
const MOCK_EMPTYABLE = /^(mths_since|num_|mo_sin|emp_)/;

const MOCK_REASON_FEATURES = [
    "dti", "fico_range_low", "loan_amnt", "annual_inc",
    "revol_util", "num_tl_op_past_12m", "term", "purpose"
];

const MOCK_T_LOW = 0.12;
const MOCK_T_HIGH = 0.3;
const MOCK_TOP_K = 5;

const mockDb = { unscored: [], predictions: [] };


function mockDecision(score) {
    if (score < MOCK_T_LOW) {
        return "APPROVE";
    }

    return score < MOCK_T_HIGH ? "REVIEW" : "DECLINE";
}


function uuid() {
    if (window.crypto && crypto.randomUUID) {
        return crypto.randomUUID();
    }

    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(
        /[xy]/g,
        (char) => {
            const digit = Math.floor(Math.random() * 16);
            return (char === "x" ? digit : (digit & 0x3) | 0x8)
                .toString(16);
        }
    );
}


function isoMonth(year, month) {
    return `${year}-${String(month).padStart(2, "0")}-01`;
}


function makeMockApplication() {
    const row = { application_id: uuid() };

    APPLICATION_COLUMNS.forEach((column) => {
        if (MOCK_TEXT[column]) {
            row[column] = pick(MOCK_TEXT[column]);
        } else if (column === "issue_d") {
            row[column] = isoMonth(2018, rndInt(1, 12));
        } else if (column === "earliest_cr_line") {
            row[column] = isoMonth(rndInt(1985, 2012), rndInt(1, 12));
        } else if (MOCK_NUMBER[column]) {
            row[column] = MOCK_NUMBER[column]();
        } else {
            row[column] = rndInt(0, 12);
        }
    });

    row.fico_range_high = row.fico_range_low + 4;

    APPLICATION_COLUMNS.forEach((column) => {
        if (MOCK_EMPTYABLE.test(column) && Math.random() < 0.08) {
            row[column] = null;
        }
    });

    return row;
}


function makeMockReasons(application) {
    return MOCK_REASON_FEATURES
        .slice()
        .sort(() => Math.random() - 0.5)
        .slice(0, MOCK_TOP_K)
        .map((feature) => {
            const impact = Number(rnd(-0.6, 0.6).toFixed(4));

            return {
                feature,
                value: application[feature],
                impact,
                direction: impact > 0
                    ? "increases P(default)"
                    : "decreases P(default)"
            };
        })
        .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
}


function makeMockPrediction(application, timestamp) {
    const score = rnd(0.02, 0.45);
    const features = {};

    SUMMARY_COLUMNS.forEach((column) => {
        features[column] = application[column];
    });

    return {
        application_id: application.application_id,
        ts: timestamp,
        model_version: "mock-0.1",
        score,
        decision: mockDecision(score),
        latency_ms: rnd(5, 40),
        response_code: 200,
        features,
        feature_importance: makeMockReasons(application)
    };
}


function seedMock() {
    for (let i = 0; i < 25; i++) {
        mockDb.unscored.push(makeMockApplication());
    }

    for (let i = 0; i < 60; i++) {
        const days = Math.random() < 0.2 ? 0 : rndInt(0, 400);

        const date = new Date(
            Date.now() - days * 86400000 - rndInt(0, 40000) * 1000
        );

        mockDb.predictions.push(
            makeMockPrediction(makeMockApplication(), date.toISOString())
        );
    }

    mockDb.predictions.sort((a, b) => b.ts.localeCompare(a.ts));
}


// Начало периода в UTC, как на бэкенде. all: без ограничения.
function periodStart(period) {
    const now = new Date();
    const day = 86400000;

    if (period === "today") {
        return new Date(Date.UTC(
            now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()
        ));
    }

    const days = { week: 7, month: 30, year: 365 }[period];

    return days ? new Date(now.getTime() - days * day) : null;
}


function filterByPeriod(items, period) {
    const start = periodStart(period);

    return start
        ? items.filter((item) => new Date(item.ts) >= start)
        : items.slice();
}


function mockFetchUnscored() {
    return { items: mockDb.unscored.slice() };
}


async function mockUploadApplications(file) {
    const text = await file.text();

    const rows = Math.max(
        text.split(/\r?\n/).filter((line) => line.trim()).length - 1,
        0
    );

    if (!rows) {
        throw new Error("The file has no rows with applications.");
    }

    for (let i = 0; i < rows; i++) {
        mockDb.unscored.push(makeMockApplication());
    }

    return { added: rows, skipped: 0 };
}


function mockBatchPredict() {
    const now = new Date().toISOString();

    const predictions = mockDb.unscored.map(
        (application) => makeMockPrediction(application, now)
    );

    mockDb.predictions.unshift(...predictions);
    mockDb.unscored = [];

    const count = (decision) => predictions.filter(
        (item) => item.decision === decision
    ).length;

    return {
        total: predictions.length,
        approved: count("APPROVE"),
        review: count("REVIEW"),
        declined: count("DECLINE")
    };
}


function mockFetchPredictions(period) {
    return { items: filterByPeriod(mockDb.predictions, period) };
}


function mockExportPredictions(period) {
    const items = filterByPeriod(mockDb.predictions, period);

    const csv = toCsv(items, [
        "application_id", "ts", "decision", "score", "model_version",
        "latency_ms", "response_code", "features", "feature_importance"
    ]);

    return csvToBlobUrl(csv);
}


function mockFetchReview() {
    // Очередь: самые старые заявки первыми
    const items = mockDb.predictions
        .filter((item) => item.decision === "REVIEW")
        .sort((a, b) => a.ts.localeCompare(b.ts));

    return { items };
}


function mockResolveReview(applicationId, decision) {
    const item = mockDb.predictions.find(
        (entry) => entry.application_id === applicationId
            && entry.decision === "REVIEW"
    );

    if (!item) {
        throw new Error("Application is not in review.");
    }

    item.decision = decision;

    return { application_id: applicationId, decision };
}


// ---------------------------------------------------------------------------
// Старт
// ---------------------------------------------------------------------------

if (USE_MOCK) {
    seedMock();
}

loadUnscored();
loadPredictions();
loadReview();