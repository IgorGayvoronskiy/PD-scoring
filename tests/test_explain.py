import json

from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV

from pd_scoring import db
from pd_scoring.service.app import build_model_frame
from pd_scoring.service.explain import explain_predictions, get_base_estimator


def test_base_estimator_is_extracted_from_calibrated_model(client):
    model = client.app.state.model

    assert isinstance(model, CalibratedClassifierCV)
    assert isinstance(get_base_estimator(model), CatBoostClassifier)
    assert client.app.state.explain_model is get_base_estimator(model)


def test_uncalibrated_model_is_returned_as_is(client):
    base = client.app.state.explain_model

    assert get_base_estimator(base) is base


def test_explanations_are_json_compliant_with_missing_values(client, good_row):
    meta = client.app.state.meta
    row = {**good_row, **{field: None for field in db.NULLABLE_COLUMNS}}

    frame = build_model_frame([row], meta["features"], meta["categorical_cols"])
    explanations = explain_predictions(
        client.app.state.model, frame, top_k=len(meta["features"])
    )[0]

    # NaN в JSON недопустим: все значения должны сериализоваться строго
    json.dumps(explanations, allow_nan=False)

    assert any(item["value"] is None for item in explanations)