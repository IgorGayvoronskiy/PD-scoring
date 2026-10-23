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
    predictionsExport: "/v1/predictions/export"
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

const UNSCORED_COLUMNS = ["application_id", ...APPLICATION_COLUMNS];

const PREDICTION_COLUMNS = [
    "application_id", "ts", "model_version",
    "score", "latency_ms", "response_code"
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


function renderTable(headId, bodyId, columns, rows, formatValue) {
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
                    makeCell(formatValue(column, value))
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
// Вкладка Applications: загрузка CSV и необработанные заявки
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


// ---------------------------------------------------------------------------
// Вкладка Batch scoring
// ---------------------------------------------------------------------------

document
    .getElementById("batch-form")
    .addEventListener("submit", async (event) => {

        event.preventDefault();

        hideError("batch-error");

        document
            .getElementById("batch-result")
            .classList.add("hidden");

        setLoading("batch", true);

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

            document.getElementById(
                "batch-total"
            ).textContent =
                data.total;

            document.getElementById(
                "batch-approved"
            ).textContent =
                data.approved;

            document.getElementById(
                "batch-rejected"
            ).textContent =
                data.rejected;

            const downloadLink =
                document.getElementById(
                    "download-link"
                );

            downloadLink.href =
                data.download_url ||
                `/uploads/${encodeURIComponent(
                    data.filename
                )}`;

            downloadLink.download =
                data.filename || "";

            const tbody =
                document.getElementById(
                    "batch-body"
                );

            tbody.innerHTML = "";

            data.preview.forEach((item) => {
                const row =
                    document.createElement("tr");

                row.appendChild(
                    makeCell(item.application_id)
                );
                row.appendChild(
                    makeCell(item.features)
                );
                row.appendChild(
                    makeCell(
                        formatNumber(
                            item["P(default)"]
                        )
                    )
                );
                row.appendChild(
                    makeCell(item.class)
                );
                row.appendChild(
                    makeCell(item.decision)
                );
                row.appendChild(
                    makeCell(item["main reasons"])
                );

                tbody.appendChild(row);
            });

            document
                .getElementById(
                    "batch-result"
                )
                .classList.remove(
                    "hidden"
                );

            // Оценённые заявки уходят из очереди и появляются в predictions
            loadUnscored();
            loadPredictions();

        } catch (error) {
            showError(
                "batch-error",
                error.message
            );
        } finally {
            setLoading("batch", false);
        }
    });


// ---------------------------------------------------------------------------
// Вкладка Predictions: таблица и выгрузка в CSV
// ---------------------------------------------------------------------------

function getDateRange() {
    const dateFrom = document.getElementById("date-from").value;
    const dateTo = document.getElementById("date-to").value;

    if (dateFrom && dateTo && dateFrom > dateTo) {
        throw new Error(
            "The start date must not be later than the end date."
        );
    }

    return { date_from: dateFrom, date_to: dateTo };
}


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


async function loadPredictions() {
    hideError("pred-error");

    let range;

    try {
        range = getDateRange();
    } catch (error) {
        showError("pred-error", error.message);
        return;
    }

    setLoading("pred", true);

    try {
        const query = buildQuery(range);

        const data = await request(
            query
                ? `${API.predictions}?${query}`
                : API.predictions,
            {},
            () => mockFetchPredictions(range)
        );

        document.getElementById(
            "pred-count"
        ).textContent = data.items.length;

        renderTable(
            "pred-head",
            "pred-body",
            PREDICTION_COLUMNS,
            data.items,
            formatPredictionValue
        );

    } catch (error) {
        showError("pred-error", error.message);
    } finally {
        setLoading("pred", false);
    }
}


document
    .getElementById("pred-form")
    .addEventListener("submit", (event) => {
        event.preventDefault();
        loadPredictions();
    });


document
    .getElementById("pred-download")
    .addEventListener("click", async (event) => {
        const button = event.currentTarget;

        hideError("pred-error");

        let range;

        try {
            range = getDateRange();
        } catch (error) {
            showError("pred-error", error.message);
            return;
        }

        const filename =
            `predictions_${range.date_from || "start"}_${range.date_to || "end"}.csv`;

        button.disabled = true;

        try {
            if (USE_MOCK) {
                await sleep(400);
                triggerDownload(
                    mockExportPredictions(range),
                    filename
                );
            } else {
                const query = buildQuery(range);

                triggerDownload(
                    query
                        ? `${API.predictionsExport}?${query}`
                        : API.predictionsExport,
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
    "dti", "fico_range_low", "loan_amnt",
    "annual_inc", "revol_util", "num_tl_op_past_12m"
];

const MOCK_THRESHOLD = 0.2;

const mockDb = { unscored: [], predictions: [] };


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


function makeMockReasons() {
    return MOCK_REASON_FEATURES
        .slice()
        .sort(() => Math.random() - 0.5)
        .slice(0, 3)
        .map((feature) => ({
            feature,
            impact: Number(rnd(-0.05, 0.05).toFixed(4))
        }));
}


function makeMockPrediction(application, timestamp) {
    return {
        application_id: application.application_id,
        ts: timestamp,
        model_version: "mock-0.1",
        score: rnd(0.02, 0.45),
        latency_ms: rnd(5, 40),
        response_code: 200,
        features: {
            loan_amnt: application.loan_amnt,
            purpose: application.purpose,
            fico_range_low: application.fico_range_low
        },
        feature_importance: makeMockReasons()
    };
}


function seedMock() {
    for (let i = 0; i < 25; i++) {
        mockDb.unscored.push(makeMockApplication());
    }

    for (let i = 0; i < 40; i++) {
        const date = new Date(
            Date.now() - rndInt(0, 30) * 86400000 - rndInt(0, 86000) * 1000
        );

        mockDb.predictions.push(
            makeMockPrediction(makeMockApplication(), date.toISOString())
        );
    }

    mockDb.predictions.sort((a, b) => b.ts.localeCompare(a.ts));
}


function filterByDates(items, range) {
    return items.filter((item) => {
        const day = item.ts.slice(0, 10);

        return (!range.date_from || day >= range.date_from)
            && (!range.date_to || day <= range.date_to);
    });
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
    const applications = mockDb.unscored;
    const now = new Date().toISOString();

    const results = applications.map((application) => {
        const prediction = makeMockPrediction(application, now);
        const isDefault = prediction.score > MOCK_THRESHOLD;

        mockDb.predictions.unshift(prediction);

        return {
            application_id: application.application_id,
            features:
                `loan_amnt=${application.loan_amnt}, `
                + `purpose=${application.purpose}, `
                + `fico=${application.fico_range_low}`,
            "P(default)": prediction.score,
            class: isDefault ? "DEFAULT" : "NO DEFAULT",
            decision: isDefault ? "Reject" : "Approve",
            "main reasons": prediction.feature_importance
                .map((item) => {
                    const sign = item.impact >= 0 ? "+" : "";
                    return `${item.feature} (${sign}${item.impact})`;
                })
                .join("; ")
        };
    });

    mockDb.unscored = [];

    const approved = results.filter(
        (item) => item.decision === "Approve"
    ).length;

    const csv = toCsv(results, [
        "application_id", "features", "P(default)",
        "class", "decision", "main reasons"
    ]);

    return {
        total: results.length,
        approved,
        rejected: results.length - approved,
        filename: "batch_results_mock.csv",
        download_url: csvToBlobUrl(csv),
        preview: results.slice(0, 10)
    };
}


function mockFetchPredictions(range) {
    return { items: filterByDates(mockDb.predictions, range) };
}


function mockExportPredictions(range) {
    const items = filterByDates(mockDb.predictions, range);

    const csv = toCsv(items, [
        ...PREDICTION_COLUMNS, "features", "feature_importance"
    ]);

    return csvToBlobUrl(csv);
}


// ---------------------------------------------------------------------------
// Старт
// ---------------------------------------------------------------------------

if (USE_MOCK) {
    seedMock();
}

loadUnscored();
loadPredictions();