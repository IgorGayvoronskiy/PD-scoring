import os

import psycopg
import pytest

DATABASE_URL = os.getenv("DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="нужен Postgres: задайте DATABASE_URL"),
]


def test_success_prediction_is_logged(client, good_row):
    body = client.post("/v1/predict", json=good_row).json()

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, score, features->>'home_ownership_n', response_code "
            "FROM predictions WHERE request_id = %s",
            (body["request_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == body["model_version"]
    assert row[1] == pytest.approx(body["score"])
    assert row[2] == good_row["home_ownership_n"]
    assert row[3] == 200


def test_422_is_logged(client, good_row):
    resp = client.post("/v1/predict", json={**good_row, "loan_amnt": -1})
    body = resp.json()

    assert resp.status_code == 422

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, score, features->>'home_ownership_n', response_code "
            "FROM predictions WHERE request_id = %s",
            (body["request_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == client.app.state.version
    assert row[1] is None
    assert row[2] == good_row["home_ownership_n"]
    assert row[3] == resp.status_code


def test_batch_prediction_is_logged(client, good_row):
    good_rows = [good_row, {**good_row, "home_ownership_n": 'OWN'}]
    body = client.post("/v1/predict/batch", json={"rows": good_rows}).json()

    requests_id = [resp_row['request_id'] for resp_row in body]

    with psycopg.connect(DATABASE_URL) as conn:
        db_rows = conn.execute(
            "SELECT request_id, features->>'home_ownership_n', score, response_code "
            "FROM predictions WHERE request_id = ANY(%s) "
            "ORDER BY features->>'home_ownership_n'",
            (requests_id,),
        ).fetchall()

    assert len(db_rows) == len(good_rows)
    assert len(db_rows) == len(body)

    by_id = {resp_row["request_id"]: resp_row for resp_row in body}
    assert all(str(db_row[0]) in by_id for db_row in db_rows)

    assert [db_row[1] for db_row in db_rows] == sorted(row["home_ownership_n"] for row in good_rows)
    assert all(db_row[3] == 200 for db_row in db_rows)
    for db_row, resp_row in zip(
        sorted(db_rows, key=lambda r: r[0]),
        sorted(body, key=lambda r: r["request_id"]),
        strict=True
    ):
        assert db_row[2] == pytest.approx(resp_row["score"])


def test_bad_row_in_batch_is_logged_once(client, good_row):
    bad_rows = [good_row, {**good_row, "loan_amnt": -1}]
    resp = client.post("/v1/predict/batch", json={"rows": bad_rows})
    body = resp.json()

    assert resp.status_code == 422

    with psycopg.connect(DATABASE_URL) as conn:
        db_rows = conn.execute(
            "SELECT response_code FROM predictions WHERE request_id = %s",
            (body['request_id'],),
        ).fetchall()

    assert len(db_rows) == 1
    assert db_rows[0][0] == resp.status_code
