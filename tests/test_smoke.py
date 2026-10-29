import pytest

from pd_scoring.service.app import decide


def test_predict_smoke(client, good_row):
    r = client.post("/v1/predict", json=good_row)

    meta = client.app.state.meta

    assert r.status_code == 200

    body = r.json()

    assert 0.0 <= body["score"] <= 1.0
    assert body["t_low"] == meta["t_low"]
    assert body["t_high"] == meta["t_high"]
    assert body["t_low"] <= body["t_high"]
    assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert body["decision"] == decide(
        body["score"], meta["t_low"], meta["t_high"]
    )
    assert isinstance(body["explanations"], list)
    assert len(body["explanations"]) <= 5
    assert body["latency_ms"] >= 0
    assert body["model_version"]
    assert body["model_version"] == client.app.state.version


def test_thresholds_are_loaded_from_metadata(client):
    meta = client.app.state.meta

    assert client.app.state.t_low == float(meta["t_low"])
    assert client.app.state.t_high == float(meta["t_high"])


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "APPROVE"),
        (0.0999, "APPROVE"),
        (0.1, "REVIEW"),      # score == t_low: уже REVIEW
        (0.2, "REVIEW"),
        (0.2999, "REVIEW"),
        (0.3, "DECLINE"),     # score == t_high: уже DECLINE
        (1.0, "DECLINE"),
    ],
)
def test_decide_boundaries(score, expected):
    assert decide(score, 0.1, 0.3) == expected


def test_decide_without_review_zone():
    # t_low == t_high: зоны REVIEW нет
    assert decide(0.19, 0.2, 0.2) == "APPROVE"
    assert decide(0.2, 0.2, 0.2) == "DECLINE"