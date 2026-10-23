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