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


const featureDefinitions = [
    {
        name: "id",
        type: "number",
        min: 1,
        step: 1
    },
    {
        name: "revenue",
        type: "number",
        min: 0,
        step: "any"
    },
    {
        name: "dti_n",
        type: "number",
        min: 0,
        step: "any"
    },
    {
        name: "loan_amnt",
        type: "number",
        min: 0,
        step: "any"
    },
    {
        name: "fico_n",
        type: "number",
        min: 300,
        max: 850,
        step: "any"
    },
    {
        name: "experience_c",
        type: "number",
        min: 0,
        max: 1,
        step: 1
    },
    {
        name: "emp_length",
        type: "text"
    },
    {
        name: "purpose",
        type: "text"
    },
    {
        name: "home_ownership_n",
        type: "text"
    },
    {
        name: "addr_state",
        type: "text"
    },
    {
        name: "zip_code",
        type: "text"
    }
];


const featuresContainer = document.getElementById(
    "features-container"
);


function createFeatureForm() {
    featuresContainer.innerHTML = "";

    featureDefinitions.forEach((feature) => {
        const wrapper = document.createElement("div");
        wrapper.className = "field";

        const label = document.createElement("label");
        label.textContent = feature.name;
        label.htmlFor = `feature-${feature.name}`;

        const input = document.createElement("input");

        input.id = `feature-${feature.name}`;
        input.name = feature.name;
        input.type = feature.type;
        input.required = true;

        if (feature.min !== undefined) {
            input.min = feature.min;
        }

        if (feature.max !== undefined) {
            input.max = feature.max;
        }

        if (feature.step !== undefined) {
            input.step = feature.step;
        }

        wrapper.appendChild(label);
        wrapper.appendChild(input);

        featuresContainer.appendChild(wrapper);
    });
}


createFeatureForm();


function getFormData() {
    const formData = {};

    featureDefinitions.forEach((feature) => {
        const input = document.getElementById(
            `feature-${feature.name}`
        );

        let value = input.value;

        if (
            feature.type === "number"
        ) {
            value = Number(value);
        }

        formData[feature.name] = value;
    });

    return formData;
}


function showError(elementId, message) {
    const element = document.getElementById(
        elementId
    );

    element.textContent = message;
    element.classList.remove("hidden");
}


function hideError(elementId) {
    document
        .getElementById(elementId)
        .classList.add("hidden");
}


function formatNumber(value) {
    return Number(value).toFixed(4);
}


function renderReasons(reasons) {
    const tbody = document.getElementById(
        "reasons-body"
    );

    tbody.innerHTML = "";

    reasons.forEach((reason) => {
        const row = document.createElement("tr");

        const feature = document.createElement("td");
        feature.textContent = reason.feature;

        const value = document.createElement("td");
        value.textContent = reason.value;

        const impact = document.createElement("td");

        const sign =
            reason.impact >= 0
                ? "+"
                : "";

        impact.textContent =
            `${sign}${Number(reason.impact).toFixed(4)}`;

        impact.className =
            reason.impact >= 0
                ? "positive"
                : "negative";

        const direction = document.createElement(
            "td"
        );

        direction.textContent =
            reason.direction;

        row.appendChild(feature);
        row.appendChild(value);
        row.appendChild(impact);
        row.appendChild(direction);

        tbody.appendChild(row);
    });
}


document
    .getElementById("single-form")
    .addEventListener("submit", async (event) => {

        event.preventDefault();

        hideError("single-error");

        const resultBlock =
            document.getElementById(
                "single-result"
            );

        try {
            const response = await fetch(
                "/v1/predict",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify(
                        getFormData()
                    )
                }
            );

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail
                        ? JSON.stringify(
                            data.detail
                        )
                        : "Prediction failed"
                );
            }

            document.getElementById(
                "result-score"
            ).textContent =
                formatNumber(data.score);

            document.getElementById(
                "result-threshold"
            ).textContent =
                formatNumber(data.threshold);

            document.getElementById(
                "result-class"
            ).textContent =
                data.class;

            document.getElementById(
                "result-decision"
            ).textContent =
                data.decision;

            renderReasons(
                data.explanations
            );

            resultBlock.classList.remove(
                "hidden"
            );

        } catch (error) {
            showError(
                "single-error",
                error.message
            );
        }
    });


document
    .getElementById("batch-form")
    .addEventListener("submit", async (event) => {

        event.preventDefault();

        hideError("batch-error");

        const fileInput =
            document.getElementById(
                "csv-file"
            );

        if (!fileInput.files.length) {
            showError(
                "batch-error",
                "Please select a CSV file."
            );

            return;
        }

        const formData = new FormData();

        formData.append(
            "file",
            fileInput.files[0]
        );

        try {
            const response = await fetch(
                "/v1/predict/batch",
                {
                    method: "POST",
                    body: formData
                }
            );

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail
                        ? JSON.stringify(
                            data.detail
                        )
                        : "Batch prediction failed"
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
                `/uploads/${encodeURIComponent(
                    data.filename
                )}`;

            const tbody =
                document.getElementById(
                    "batch-body"
                );

            tbody.innerHTML = "";

            data.preview.forEach((item) => {
                const row =
                    document.createElement("tr");

                const features =
                    document.createElement("td");

                features.textContent =
                    item.features;

                const score =
                    document.createElement("td");

                score.textContent =
                    formatNumber(
                        item["P(default)"]
                    );

                const classCell =
                    document.createElement("td");

                classCell.textContent =
                    item.class;

                const decision =
                    document.createElement("td");

                decision.textContent =
                    item.decision;

                const reasons =
                    document.createElement("td");

                reasons.textContent =
                    item.main_reasons;

                row.appendChild(features);
                row.appendChild(score);
                row.appendChild(classCell);
                row.appendChild(decision);
                row.appendChild(reasons);

                tbody.appendChild(row);
            });

            document
                .getElementById(
                    "batch-result"
                )
                .classList.remove("hidden");

        } catch (error) {
            showError(
                "batch-error",
                error.message
            );
        }
    });