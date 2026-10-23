import io
import os
import uuid
from datetime import UTC, datetime

import pandas as pd
import psycopg
import pytest

from pd_scoring import db

DATABASE_URL = os.getenv("DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="нужен Postgres: задайте DATABASE_URL"),
]


def _csv(rows):
    buffer = io.StringIO()
    pd.DataFrame(rows).reindex(columns=db.APPLICATION_COLUMNS).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def _upload(client, rows, name="applications.csv"):
    return client.post(
        "/v1/applications/upload",
        files={"file": (name, _csv(rows), "text/csv")},
    )


def _marked_rows(good_row):
    """Две заявки с уникальными emp_title, чтобы найти их в общей таблице."""
    marker = uuid.uuid4().hex
    markers = [f"{marker}-a", f"{marker}-b"]
    rows = [
        {**good_row, "emp_title": markers[0]},
        {**good_row, "emp_title": markers[1], "home_ownership": "OWN", "dti": None},
    ]
    return markers, rows


def _unscored_titles(client):
    items = client.get("/v1/applications/unscored").json()["items"]
    return {item["emp_title"] for item in items}


def test_success_prediction_is_logged(client, good_row):
    body = client.post("/v1/predict", json=good_row).json()

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT p.model_version, p.score, p.features->>'home_ownership', "
            "p.response_code, p.feature_importance, a.scored "
            "FROM predictions p JOIN applications a USING (application_id) "
            "WHERE p.application_id = %s",
            (body["request_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == body["model_version"]
    assert row[1] == pytest.approx(body["score"])
    assert row[2] == good_row["home_ownership"]
    assert row[3] == 200
    assert len(row[4]) == len(body["explanations"])
    assert row[5] is True


def test_invalid_request_is_not_stored(client, good_row):
    marker = uuid.uuid4().hex
    resp = client.post(
        "/v1/predict", json={**good_row, "emp_title": marker, "loan_amnt": -1}
    )

    assert resp.status_code == 422

    with psycopg.connect(DATABASE_URL) as conn:
        count = conn.execute(
            "SELECT count(*) FROM applications WHERE emp_title = %s", (marker,)
        ).fetchone()[0]

    assert count == 0


def test_upload_then_batch_scores_and_marks_applications(client, good_row):
    markers, rows = _marked_rows(good_row)

    uploaded = _upload(client, rows)
    assert uploaded.status_code == 200
    assert uploaded.json() == {"added": 2}

    # Заявки видны в списке необработанных
    assert set(markers) <= _unscored_titles(client)

    batch = client.post("/v1/predict/batch")
    assert batch.status_code == 200

    result = batch.json()
    assert result["total"] >= 2
    assert result["approved"] + result["rejected"] == result["total"]
    assert result["filename"]
    assert len(result["preview"]) == min(10, result["total"])
    assert {
        "application_id", "features", "P(default)", "class", "decision", "main reasons",
    } <= set(result["preview"][0])
    assert client.get(f"/uploads/{result['filename']}").status_code == 200

    # Оценённые заявки ушли из очереди
    assert not set(markers) & _unscored_titles(client)

    with psycopg.connect(DATABASE_URL) as conn:
        db_rows = conn.execute(
            "SELECT a.emp_title, a.scored, p.score, p.response_code, p.feature_importance "
            "FROM applications a JOIN predictions p USING (application_id) "
            "WHERE a.emp_title = ANY(%s) ORDER BY a.emp_title",
            (markers,),
        ).fetchall()

    assert len(db_rows) == 2
    assert all(row[1] is True and row[3] == 200 for row in db_rows)
    assert all(row[4] for row in db_rows)

    # Оценка из batch совпадает с оценкой /v1/predict для той же заявки
    single = client.post("/v1/predict", json=rows[0]).json()
    assert db_rows[0][2] == pytest.approx(single["score"])


def test_upload_with_missing_column_is_422(client, good_row):
    rows = [good_row]
    df = pd.read_csv(io.BytesIO(_csv(rows))).drop(columns=["revol_bal"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    resp = client.post(
        "/v1/applications/upload",
        files={"file": ("a.csv", buffer.getvalue().encode(), "text/csv")},
    )

    assert resp.status_code == 422
    assert resp.json()["detail"]["missing"] == ["revol_bal"]


def test_upload_with_invalid_row_adds_nothing(client, good_row):
    markers, rows = _marked_rows(good_row)
    rows[1]["loan_amnt"] = -1

    resp = _upload(client, rows)

    assert resp.status_code == 422
    # строка 1 это заголовок, плохая заявка стоит в строке 3
    assert resp.json()["detail"]["rows"][0]["row"] == 3

    with psycopg.connect(DATABASE_URL) as conn:
        count = conn.execute(
            "SELECT count(*) FROM applications WHERE emp_title = ANY(%s)", (markers,)
        ).fetchone()[0]

    assert count == 0


def test_upload_rejects_non_csv(client):
    resp = client.post(
        "/v1/applications/upload",
        files={"file": ("a.txt", b"x", "text/plain")},
    )

    assert resp.status_code == 400


def test_predictions_list_and_export(client, good_row):
    body = client.post("/v1/predict", json=good_row).json()
    today = datetime.now(UTC).date().isoformat()

    listed = client.get(
        "/v1/predictions", params={"date_from": today, "date_to": today}
    )
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert body["request_id"] in {item["application_id"] for item in items}
    assert set(items[0]) == {
        "application_id", "ts", "model_version", "score", "latency_ms", "response_code",
    }

    exported = client.get(
        "/v1/predictions/export", params={"date_from": today, "date_to": today}
    )
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("text/csv")
    assert "attachment" in exported.headers["content-disposition"]

    frame = pd.read_csv(io.StringIO(exported.text))
    assert list(frame.columns) == [
        "application_id", "ts", "model_version", "score", "latency_ms",
        "response_code", "features", "feature_importance",
    ]
    assert body["request_id"] in set(frame["application_id"])


def test_predictions_future_range_is_empty(client):
    resp = client.get("/v1/predictions", params={"date_from": "2999-01-01"})

    assert resp.status_code == 200
    assert resp.json()["items"] == []


def test_predictions_reversed_range_is_422(client):
    resp = client.get(
        "/v1/predictions",
        params={"date_from": "2026-02-01", "date_to": "2026-01-01"},
    )

    assert resp.status_code == 422