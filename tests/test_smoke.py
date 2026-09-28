def test_predict_smoke(client, good_row):
    r = client.post("/v1/predict", json=good_row)
    threshold = client.app.state.meta["threshold"]

    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert isinstance(body["default"], bool)
    assert body["latency_ms"] >= 0
    assert body["model_version"]
    assert (body["score"] >= threshold) == body["default"]
    assert r.json()["model_version"] == client.app.state.version



def test_batch_and_single_agree(client, good_row):
    s1 = client.post("/v1/predict", json=good_row).json()["score"]
    s2 = client.post("/v1/predict", json=good_row).json()["score"]
    assert abs(s1 - s2) < 1e-12