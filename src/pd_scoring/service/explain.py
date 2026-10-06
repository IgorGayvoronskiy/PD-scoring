import pandas as pd
import shap


def _get_shap_values(
    model,
    features: pd.DataFrame,
):
    explainer = shap.Explainer(model)
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

    Returns one list of explanations for each input row.
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
                    "value": value,
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
    """
    Explain a single prediction.

    This is a convenience wrapper around explain_predictions().
    """
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