import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # scikit-learn < 1.6
    FrozenEstimator = None


def get_base_estimator(model):
    """Возвращает исходную (некалиброванную) модель.

    В артефакте лежит одна калиброванная модель
    CalibratedClassifierCV(FrozenEstimator(CatBoostClassifier)). SHAP считается
    по вложенному CatBoost: калибровка (isotonic/sigmoid) монотонна и не
    меняет ранжирования, но SHAP-значения не умеет считать по обёртке.
    Если передана уже некалиброванная модель, она возвращается как есть.
    """
    if isinstance(model, CalibratedClassifierCV):
        base = model.calibrated_classifiers_[0].estimator

        if FrozenEstimator is not None and isinstance(base, FrozenEstimator):
            base = base.estimator

        return base

    return model


def _to_python(value):
    """Приводит значение признака к JSON-совместимому виду.

    NaN и NA превращаются в None (NaN в JSON-ответе недопустим),
    numpy-скаляры в обычные числа Python.
    """
    if value is None or value is pd.NA:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and np.isnan(value):
        return None

    return value


def _get_shap_values(
    model,
    features: pd.DataFrame,
):
    explainer = shap.Explainer(get_base_estimator(model))
    explanation = explainer(features)

    values = explanation.values

    if values.ndim == 2:
        return values

    if values.ndim == 3:
        return values[:, :, 1]

    raise ValueError(
        f"Unsupported SHAP values shape: {values.shape}"
    )


def explain_predictions(
    model,
    features: pd.DataFrame,
    top_k: int = 5,
) -> list[list[dict]]:
    """
    Explain multiple predictions using one SHAP explainer.

    `model` может быть как калиброванной моделью (из неё берётся базовая),
    так и самой базовой моделью. Impact измеряется в лог-шансах базовой
    модели: знак и порядок признаков информативны, но это не вклад
    в калиброванную вероятность.
    """
    shap_values = _get_shap_values(
        model,
        features,
    )

    results = []

    for row_idx in range(len(features)):
        row = features.iloc[row_idx]

        explanations = []

        for feature, value, shap_value in zip(
            features.columns,
            row,
            shap_values[row_idx],
            strict=True,
        ):
            shap_value = float(shap_value)

            explanations.append(
                {
                    "feature": feature,
                    "value": _to_python(value),
                    "impact": shap_value,
                    "direction": (
                        "increases P(default)"
                        if shap_value > 0
                        else "decreases P(default)"
                    ),
                }
            )

        explanations.sort(
            key=lambda item: abs(item["impact"]),
            reverse=True,
        )

        results.append(
            explanations[:top_k]
        )

    return results


def explain_prediction(
    model,
    features: pd.DataFrame,
    top_k: int = 5,
) -> list[dict]:
    results = explain_predictions(
        model,
        features,
        top_k=top_k,
    )

    if len(results) != 1:
        raise ValueError(
            "explain_prediction expects exactly one row"
        )

    return results[0]