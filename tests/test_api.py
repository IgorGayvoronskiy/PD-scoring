import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_version" in r.json()


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_bad_dti_is_422(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "dti_n": -1})
    assert r.status_code == 422


def test_missing_field_is_422(client, good_row):
    row = dict(good_row)
    del row["revenue"]
    assert client.post("/v1/predict", json=row).status_code == 422


def test_extra_field_is_422(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "hacker_field": 1})
    assert r.status_code == 422


def test_zero_loan_amount_is_422(client, good_row):
    r = client.post(
        "/v1/predict",
        json={**good_row, "loan_amnt": 0}
    )
    assert r.status_code == 422

def test_zero_revenue_is_valid(client, good_row):
    r = client.post(
        "/v1/predict",
        json={**good_row, "revenue": 0}
    )
    assert r.status_code == 200


def test_zero_dti_is_valid(client, good_row):
    r = client.post(
        "/v1/predict",
        json={**good_row, "dti_n": 0}
    )
    assert r.status_code == 200


def test_experience_zero_is_valid(client, good_row):
    r = client.post(
        "/v1/predict",
        json={**good_row, "experience_c": 0}
    )
    assert r.status_code == 200


def test_predict_response(client, good_row):
    r = client.post("/v1/predict", json=good_row)

    assert r.status_code == 200

    data = r.json()

    assert isinstance(data["request_id"], str)
    assert isinstance(data["score"], float)
    assert isinstance(data["threshold"], float)
    assert isinstance(data["default"], bool)
    assert data["decision"] in {"APPROVE", "REJECT"}
    assert isinstance(data["explanations"], list)
    assert len(data["explanations"]) <= 5
    assert isinstance(data["model_version"], str)
    assert isinstance(data["latency_ms"], float)

    if data["explanations"]:
        explanation = data["explanations"][0]

        assert "feature" in explanation
        assert "value" in explanation
        assert "impact" in explanation
        assert "direction" in explanation


@pytest.mark.parametrize(
    "field",
    [
        "issue_d",
        "revenue",
        "dti_n",
        "loan_amnt",
        "fico_n",
        "experience_c",
        "emp_length",
        "purpose",
        "home_ownership_n",
        "addr_state",
        "zip_code",
    ],
)
def test_missing_required_field_is_422(client, good_row, field):
    row = dict(good_row)
    del row[field]

    r = client.post("/v1/predict", json=row)

    assert r.status_code == 422
