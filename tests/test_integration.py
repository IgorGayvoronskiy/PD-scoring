import io
import os
import uuid

import pandas as pd
import psycopg
import pytest

from pd_scoring import db
from pd_scoring.service.app import SUMMARY_COLUMNS, decide

DATABASE_URL = os.getenv("DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="нужен Postgres: задайте DATABASE_URL"),
]

PREDICTION_KEYS = {
    "application_id", "ts", "model_version", "score", "decision",
    "latency_ms", "response_code", "features",
}


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


def _make_prediction(good_row, decision, score=0.2):
    """Кладёт в БД заявку и готовое предсказание с нужным decision,
    не завися от того, какой score выдаст модель."""
    payload = {**good_row, "emp_title": uuid.uuid4().hex}
    application_id = db.save_applications([payload])[0]

    db.save_predictions([
        {
            "application_id": application_id,
            "features": payload,
            "score": score,
            "decision": decision,
            "model_version": "test",
            "latency_ms": 1.0,
            "response_code": 200,
            "feature_importance": [
                {
                    "feature": "dti",
                    "value": payload["dti"],
                    "impact": 0.31,
                    "direction": "increases P(default)",
                },
                {
                    "feature": "fico_range_low",
                    "value": payload["fico_range_low"],
                    "impact": -0.12,
                    "direction": "decreases P(default)",
                },
            ],
        }
    ])

    return application_id


def _review_ids(client):
    items = client.get("/v1/review").json()["items"]
    return {item["application_id"] for item in items}


def _prediction_decisions(client, period="today"):
    items = client.get("/v1/predictions", params={"period": period}).json()["items"]
    return {item["application_id"]: item["decision"] for item in items}


def test_success_prediction_is_logged(client, good_row):
    body = client.post("/v1/predict", json=good_row).json()

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT p.model_version, p.score, p.features->>'home_ownership', "
            "p.response_code, p.feature_importance, a.scored, p.decision "
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
    assert row[6] == body["decision"]


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
    assert set(result) == {"total", "approved", "review", "declined"}
    assert result["total"] >= 2
    assert result["approved"] + result["review"] + result["declined"] == result["total"]

    # Оценённые заявки ушли из очереди
    assert not set(markers) & _unscored_titles(client)

    with psycopg.connect(DATABASE_URL) as conn:
        db_rows = conn.execute(
            "SELECT a.emp_title, a.scored, p.score, p.response_code, "
            "p.feature_importance, p.decision "
            "FROM applications a JOIN predictions p USING (application_id) "
            "WHERE a.emp_title = ANY(%s) ORDER BY a.emp_title",
            (markers,),
        ).fetchall()

    assert len(db_rows) == 2
    assert all(row[1] is True and row[3] == 200 for row in db_rows)
    assert all(row[4] for row in db_rows)

    # Решение в БД соответствует порогам из метаданных модели
    meta = client.app.state.meta
    for row in db_rows:
        assert row[5] == decide(row[2], meta["t_low"], meta["t_high"])

    # Оценка из batch совпадает с оценкой /v1/predict для той же заявки
    single = client.post("/v1/predict", json=rows[0]).json()
    assert db_rows[0][2] == pytest.approx(single["score"])


def test_batch_without_unscored_returns_zeros(client):
    # Первый вызов разбирает очередь, второй видит её пустой
    client.post("/v1/predict/batch")
    resp = client.post("/v1/predict/batch")

    assert resp.status_code == 200
    assert resp.json() == {"total": 0, "approved": 0, "review": 0, "declined": 0}


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

    listed = client.get("/v1/predictions", params={"period": "today"})
    assert listed.status_code == 200
    items = listed.json()["items"]

    mine = next(item for item in items if item["application_id"] == body["request_id"])

    assert set(mine) == PREDICTION_KEYS
    assert mine["decision"] == body["decision"]
    # В таблицу попадают только признаки из SUMMARY_COLUMNS
    assert set(mine["features"]) == set(SUMMARY_COLUMNS)
    assert mine["features"]["loan_amnt"] == good_row["loan_amnt"]

    exported = client.get("/v1/predictions/export", params={"period": "today"})
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("text/csv")
    assert "attachment" in exported.headers["content-disposition"]
    assert "predictions_today.csv" in exported.headers["content-disposition"]

    frame = pd.read_csv(io.StringIO(exported.text))
    assert list(frame.columns) == [
        "application_id", "ts", "model_version", "score", "decision",
        "latency_ms", "response_code", "features", "feature_importance",
    ]
    assert body["request_id"] in set(frame["application_id"])


def test_predictions_default_period_is_all(client, good_row):
    body = client.post("/v1/predict", json=good_row).json()

    resp = client.get("/v1/predictions")

    assert resp.status_code == 200
    assert body["request_id"] in {
        item["application_id"] for item in resp.json()["items"]
    }


def test_predictions_periods(client, good_row):
    application_id = _make_prediction(good_row, "APPROVE", score=0.01)

    # Сдвигаем предсказание на 10 суток назад: вне today и week, внутри month
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "UPDATE predictions SET ts = now() - interval '10 days' "
            "WHERE application_id = %s",
            (application_id,),
        )

    def present(period):
        return application_id in _prediction_decisions(client, period)

    assert not present("today")
    assert not present("week")
    assert present("month")
    assert present("year")

    # Сдвигаем за пределы года: остаётся только в all
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "UPDATE predictions SET ts = now() - interval '400 days' "
            "WHERE application_id = %s",
            (application_id,),
        )

    assert not present("year")
    assert present("all")


def test_review_lists_only_review_applications(client, good_row):
    review_id = _make_prediction(good_row, "REVIEW")
    approve_id = _make_prediction(good_row, "APPROVE", score=0.01)
    decline_id = _make_prediction(good_row, "DECLINE", score=0.9)

    resp = client.get("/v1/review")
    assert resp.status_code == 200

    items = {item["application_id"]: item for item in resp.json()["items"]}

    assert review_id in items
    assert approve_id not in items
    assert decline_id not in items

    item = items[review_id]
    assert set(item) == {
        "application_id", "ts", "model_version", "score", "decision",
        "feature_importance",
    }
    assert item["decision"] == "REVIEW"
    assert [x["feature"] for x in item["feature_importance"]] == [
        "dti", "fico_range_low",
    ]
    assert set(item["feature_importance"][0]) == {
        "feature", "value", "impact", "direction",
    }


@pytest.mark.parametrize("decision", ["APPROVE", "DECLINE"])
def test_review_resolve_moves_application(client, good_row, decision):
    application_id = _make_prediction(good_row, "REVIEW")

    assert application_id in _review_ids(client)

    resp = client.post(
        f"/v1/review/{application_id}/decision", json={"decision": decision}
    )

    assert resp.status_code == 200
    assert resp.json() == {"application_id": application_id, "decision": decision}

    # Заявка исчезла из очереди, а в predictions у неё новое решение
    assert application_id not in _review_ids(client)
    assert _prediction_decisions(client)[application_id] == decision

    with psycopg.connect(DATABASE_URL) as conn:
        reviewed_at = conn.execute(
            "SELECT reviewed_at FROM predictions WHERE application_id = %s",
            (application_id,),
        ).fetchone()[0]

    assert reviewed_at is not None


def test_review_resolve_twice_is_409(client, good_row):
    application_id = _make_prediction(good_row, "REVIEW")
    url = f"/v1/review/{application_id}/decision"

    assert client.post(url, json={"decision": "APPROVE"}).status_code == 200

    again = client.post(url, json={"decision": "DECLINE"})

    assert again.status_code == 409
    # Первое решение не перезаписано
    assert _prediction_decisions(client)[application_id] == "APPROVE"


def test_review_resolve_not_in_review_is_409(client, good_row):
    application_id = _make_prediction(good_row, "DECLINE", score=0.9)

    resp = client.post(
        f"/v1/review/{application_id}/decision", json={"decision": "APPROVE"}
    )

    assert resp.status_code == 409
    assert _prediction_decisions(client)[application_id] == "DECLINE"


def test_review_resolve_unknown_application_is_409(client):
    resp = client.post(
        f"/v1/review/{uuid.uuid4()}/decision", json={"decision": "APPROVE"}
    )

    assert resp.status_code == 409