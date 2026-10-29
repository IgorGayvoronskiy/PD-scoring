import os
import uuid

import pytest

from pd_scoring import db

REQUIRED_FIELDS = [c for c in db.APPLICATION_COLUMNS if c not in db.NULLABLE_COLUMNS]
OPTIONAL_FIELDS = sorted(db.NULLABLE_COLUMNS)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_version" in r.json()


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "PD Scoring" in r.text


def test_index_page_has_tabs_without_batch(client):
    text = client.get("/").text

    assert 'data-tab="applications"' in text
    assert 'data-tab="predictions"' in text
    assert 'data-tab="review"' in text
    assert 'data-tab="batch"' not in text


def test_bad_dti_is_422(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "dti": -1})
    assert r.status_code == 422


def test_bad_fico_is_422(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "fico_range_low": 900})
    assert r.status_code == 422


def test_bad_date_is_422(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "issue_d": "not-a-date"})
    assert r.status_code == 422


def test_extra_field_is_422(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "hacker_field": 1})
    assert r.status_code == 422


def test_zero_loan_amount_is_422(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "loan_amnt": 0})
    assert r.status_code == 422


def test_zero_income_is_valid(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "annual_inc": 0})
    assert r.status_code == 200


def test_zero_dti_is_valid(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "dti": 0})
    assert r.status_code == 200


@pytest.mark.parametrize("date_value", ["2018-01-01", "Jan-2018", 1514764800000])
def test_date_formats_are_accepted(client, good_row, date_value):
    r = client.post("/v1/predict", json={**good_row, "issue_d": date_value})
    assert r.status_code == 200


def test_predict_response(client, good_row):
    r = client.post("/v1/predict", json=good_row)

    assert r.status_code == 200

    data = r.json()

    assert isinstance(data["request_id"], str)
    assert isinstance(data["score"], float)
    assert isinstance(data["t_low"], float)
    assert isinstance(data["t_high"], float)
    assert data["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
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


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_required_field_is_422(client, good_row, field):
    row = dict(good_row)
    row.pop(field)

    r = client.post("/v1/predict", json=row)

    assert r.status_code == 422


@pytest.mark.parametrize("field", OPTIONAL_FIELDS)
def test_missing_optional_field_is_valid(client, good_row, field):
    row = {k: v for k, v in good_row.items() if k != field}

    r = client.post("/v1/predict", json=row)

    assert r.status_code == 200


def test_explicit_nulls_in_optional_fields_are_valid(client, good_row):
    row = {**good_row, **{field: None for field in OPTIONAL_FIELDS}}

    r = client.post("/v1/predict", json=row)

    assert r.status_code == 200
    assert 0.0 <= r.json()["score"] <= 1.0


# Эти проверки срабатывают до обращения к БД, поэтому БД не нужна

@pytest.mark.parametrize("url", ["/v1/predictions", "/v1/predictions/export"])
@pytest.mark.parametrize("period", ["tomorrow", "", "2026-01-01"])
def test_bad_period_is_422(client, url, period):
    assert client.get(url, params={"period": period}).status_code == 422


@pytest.mark.parametrize("decision", ["REVIEW", "approve", "", "REJECT"])
def test_review_decision_must_be_approve_or_decline(client, decision):
    r = client.post(
        f"/v1/review/{uuid.uuid4()}/decision", json={"decision": decision}
    )

    assert r.status_code == 422


def test_review_decision_requires_uuid(client):
    r = client.post("/v1/review/not-a-uuid/decision", json={"decision": "APPROVE"})

    assert r.status_code == 422


@pytest.mark.skipif(bool(os.getenv("DATABASE_URL")), reason="база настроена")
@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("get", "/v1/applications/unscored"),
        ("post", "/v1/predict/batch"),
        ("get", "/v1/predictions"),
        ("get", "/v1/predictions?period=week"),
        ("get", "/v1/predictions/export"),
        ("get", "/v1/review"),
    ],
)
def test_db_endpoints_without_database_are_503(client, method, url):
    assert getattr(client, method)(url).status_code == 503


@pytest.mark.skipif(bool(os.getenv("DATABASE_URL")), reason="база настроена")
def test_review_decision_without_database_is_503(client):
    r = client.post(
        f"/v1/review/{uuid.uuid4()}/decision", json={"decision": "APPROVE"}
    )

    assert r.status_code == 503