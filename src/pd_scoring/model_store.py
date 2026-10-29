import joblib

from pd_scoring.config import settings


def load_model() -> tuple[object, dict, str]:
    """Возвращает (модель, метаданные, версия).

    Без MODEL_NAME модель читается из joblib-артефакта (model_path).
    С MODEL_NAME берётся версия реестра MLflow по алиасу (по умолчанию champion).
    """
    if not settings.model_name:
        bundle = joblib.load(settings.model_path)
        return bundle["model"], bundle["metadata"], bundle["metadata"]["model_version"]

    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mv = MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_alias)

    model = mlflow.sklearn.load_model(f"models:/{settings.model_name}/{mv.version}")
    meta = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/metadata.json")

    return model, meta, f"{settings.model_name}-v{mv.version}"