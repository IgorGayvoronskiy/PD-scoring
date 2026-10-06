def test_predict_smoke(client, good_row):
    r = client.post("/v1/predict", json=good_row)

    threshold = client.app.state.meta["threshold"]

    assert r.status_code == 200

    body = r.json()

    assert 0.0 <= body["score"] <= 1.0
    assert body["threshold"] == threshold
    assert isinstance(body["default"], bool)
    assert body["decision"] in {"APPROVE", "REJECT"}
    assert body["decision"] == (
        "REJECT" if body["default"] else "APPROVE"
    )
    assert isinstance(body["explanations"], list)
    assert len(body["explanations"]) <= 5
    assert body["latency_ms"] >= 0
    assert body["model_version"]
    assert (body["score"] >= threshold) == body["default"]
    assert body["model_version"] == client.app.state.version


def test_batch_and_single_agree(client, good_row):
    from pathlib import Path

    import pandas as pd

    batch_file = Path("input.csv")
    pd.DataFrame([good_row]).to_csv(batch_file, index=False)

    single = client.post(
        "/v1/predict",
        json=good_row,
    )

    with batch_file.open("rb") as f:
        batch = client.post(
            "/v1/predict/batch",
            files={
                "file": (
                    "input.csv",
                    f,
                    "text/csv",
                )
            },
        )

    assert single.status_code == 200
    assert batch.status_code == 200

    single_body = single.json()
    batch_body = batch.json()

    assert batch_body["total"] == 1
    assert batch_body["approved"] + batch_body["rejected"] == 1

    row = batch_body["preview"][0]

    assert abs(
        single_body["score"] - row["P(default)"]
    ) < 1e-12

    assert row["class"] == (
        "DEFAULT"
        if single_body["default"]
        else "NO DEFAULT"
    )

    assert row["decision"] == single_body["decision"]