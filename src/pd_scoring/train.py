"""Обучение PD-модели: данные, CatBoost, калибровка, запись в MLflow, регистрация и гейт.

  MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python -m pd_scoring.train

Новая версия всегда получает алиас challenger. Алиас champion она получает, только если
на одной и той же тестовой выборке (champion пересчитывается на ней заново):
  - Gini лучше, чем у текущего champion, больше чем на GATE_MIN_GAIN_GINI;
  - PR-AUC не упал больше чем на GATE_MAX_PR_AUC_DROP;
  - Brier Score не вырос больше чем на GATE_MAX_BRIER_INCREASE
(или champion ещё нет).

В реестр кладётся калиброванная модель целиком:
CalibratedClassifierCV(FrozenEstimator(CatBoostClassifier)), как её ждёт сервис.
"""
import hashlib
import json
import os
from pathlib import Path

import catboost
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from catboost import CatBoostClassifier, Pool  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pd_scoring.service.explain import get_explainer  # noqa: E402

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # scikit-learn < 1.6
    FrozenEstimator = None


# --------------------------------------------------------------------------- #
# Настройки (переопределяются переменными окружения)
# --------------------------------------------------------------------------- #

DATA_PATH = Path(os.getenv("DATA_PATH", "data/accepted_2007_to_2018Q4.csv.gz"))
MODEL_NAME = os.getenv("MODEL_NAME", "pd_scoring")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "pd_scoring")

# Временное разбиение: train / valid / test
TRAIN_END = os.getenv("TRAIN_END", "2015-12-31")
VALID_END = os.getenv("VALID_END", "2016-12-31")
TEST_END = os.getenv("TEST_END", "2018-12-31")
# Доля valid (по времени, последняя часть) под калибровку; остальное под early stopping
CALIB_FRAC = float(os.getenv("CALIB_FRAC", "0.5"))

TOTAL_ITERATIONS = int(os.getenv("TOTAL_ITERATIONS", "2000"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.03"))
DEPTH = int(os.getenv("DEPTH", "6"))
EVAL_METRIC = os.getenv("EVAL_METRIC", "NormalizedGini")
EARLY_STOPPING_ROUNDS = int(os.getenv("EARLY_STOPPING_ROUNDS", "100"))
CALIBRATION_METHOD = os.getenv("CALIBRATION_METHOD", "isotonic")

# Пороги решения: score < T_LOW -> APPROVE, T_LOW <= score < T_HIGH -> REVIEW,
# score >= T_HIGH -> DECLINE. После переобучения их стоит перепроверять на valid_calib.
T_LOW = float(os.getenv("T_LOW", "0.2027"))
T_HIGH = float(os.getenv("T_HIGH", "0.22"))
LGD = float(os.getenv("LGD", "0.6"))

# Гейт champion. Gini основной, PR-AUC и Brier не должны заметно ухудшаться:
# Brier следит за качеством калибровки, от которой зависят пороги.
MIN_GAIN_GINI = float(os.getenv("GATE_MIN_GAIN_GINI", "5e-5"))
MAX_PR_AUC_DROP = float(os.getenv("GATE_MAX_PR_AUC_DROP", "5e-4"))
MAX_BRIER_INCREASE = float(os.getenv("GATE_MAX_BRIER_INCREASE", "1e-4"))

SHAP_SAMPLE_SIZE = int(os.getenv("SHAP_SAMPLE_SIZE", "5000"))
TOP_N_FEATURES = 20
SEED = 42

TARGET = "Default"
DATE_COL = "issue_d"

NUMERIC = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low",
    "fico_range_high", "inq_last_6mths", "mths_since_last_delinq", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc",
    "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal",
    "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "delinq_amnt",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
    "credit_history_months",  # производный из earliest_cr_line
]
CATEGORICAL = [
    "term", "emp_length", "home_ownership", "verification_status",
    "purpose", "addr_state", "application_type",
]
# Высокая кардинальность: для CatBoost остаются нативными категориями
HIGH_CARDINALITY = ["emp_title", "zip_code"]

FEATURES = NUMERIC + CATEGORICAL + HIGH_CARDINALITY
CAT_FEATURES = CATEGORICAL + HIGH_CARDINALITY


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df[TARGET] = (df["loan_status"] == "Charged Off").astype(int)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%b-%Y", errors="coerce")

    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_months"] = (
        (df[DATE_COL].dt.year - earliest.dt.year) * 12
        + (df[DATE_COL].dt.month - earliest.dt.month)
    )

    # Та же нормализация, что в сервисе (build_model_frame)
    for col in HIGH_CARDINALITY:
        df[col] = df[col].fillna("missing").astype(str).str.strip().str.lower()

    return (
        df[FEATURES + [TARGET, DATE_COL]]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


CHUNK_ROWS = int(os.getenv("CHUNK_ROWS", "200000"))


def load_data(path: Path) -> pd.DataFrame:
    usecols = FEATURES + [DATE_COL, "earliest_cr_line", "loan_status"]
    usecols = [c for c in dict.fromkeys(usecols) if c != "credit_history_months"]

    dtypes = {c: "float32" for c in NUMERIC if c != "credit_history_months"}
    dtypes.update({c: str for c in CATEGORICAL + HIGH_CARDINALITY})

    parts = []
    reader = pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=CHUNK_ROWS)

    for chunk in reader:
        chunk = chunk[chunk["loan_status"].isin(["Fully Paid", "Charged Off"])]
        parts.append(add_derived_features(chunk))

    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(DATE_COL).reset_index(drop=True)


def time_based_split(df: pd.DataFrame):
    train = df[df[DATE_COL] <= TRAIN_END]
    valid = df[(df[DATE_COL] > TRAIN_END) & (df[DATE_COL] <= VALID_END)]
    test = df[(df[DATE_COL] > VALID_END) & (df[DATE_COL] <= TEST_END)]
    return train, valid, test


def split_valid_for_calibration(valid: pd.DataFrame):
    """valid_es для early stopping, valid_calib только для калибровки:
    калибровка на данных, повлиявших на выбор модели, была бы оптимистичной.
    test остаётся нетронутым."""
    valid = valid.sort_values(DATE_COL)
    split_idx = int(len(valid) * (1 - CALIB_FRAC))
    return valid.iloc[:split_idx], valid.iloc[split_idx:]


def validate_splits(**splits: pd.DataFrame) -> None:
    for name, part in splits.items():
        if len(part) < 1000:
            raise ValueError(f"Слишком мало строк в {name}: {len(part)}")
        if part[TARGET].nunique() < 2:
            raise ValueError(f"В {name} только один класс таргета")


def prepare_frame(df: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    """Вход модели: пропуск в категориях -> "missing" ДО приведения к str
    (иначе NaN превратится в "nan"); числовые пропуски остаются NaN."""
    frame = df[features].copy()
    for col in categorical:
        frame[col] = frame[col].fillna("missing").astype(str)
    return frame


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Модель
# --------------------------------------------------------------------------- #

def train_catboost(X_train, y_train, X_val, y_val) -> CatBoostClassifier:
    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(X_val, y_val, cat_features=CAT_FEATURES)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric=EVAL_METRIC,
        auto_class_weights="Balanced",
        iterations=TOTAL_ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        random_seed=SEED,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=100,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def calibrate_model(base_model, X_calib, y_calib) -> CalibratedClassifierCV:
    """Калибрует уже обученную модель: балансировка весов классов улучшает
    ранжирование, но искажает масштаб вероятностей."""
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model), method=CALIBRATION_METHOD
        )
    else:
        calibrated = CalibratedClassifierCV(
            estimator=base_model, method=CALIBRATION_METHOD, cv="prefit"
        )
    calibrated.fit(X_calib, y_calib)
    return calibrated


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #

def compute_ml_metrics(
        y_true: np.ndarray,
        proba: np.ndarray,
        y_pred: np.ndarray | None = None
) -> dict:
    roc_auc = roc_auc_score(y_true, proba)

    metrics = {
        "PR-AUC": average_precision_score(y_true, proba),
        "ROC-AUC": roc_auc,
        "Gini": 2 * roc_auc - 1,
        "Brier Score": brier_score_loss(y_true, proba),
    }

    if y_pred is not None:
        metrics["Precision"] = precision_score(y_true, y_pred)
        metrics["Recall"] = recall_score(y_true, y_pred)

    return {k: float(v) for k, v in metrics.items()}


def compute_calibration(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> dict:
    """Точки reliability-диаграммы: предсказанный PD и доля дефолтов по бинам."""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return {
        "predicted_pd": [float(x) for x in mean_pred],
        "observed_default_rate": [float(x) for x in frac_pos],
    }


def compute_business_metrics(y_true, loan_amnt, proba, threshold: float, lgd: float) -> dict:
    """Одобряются заявки с PD ниже порога (как APPROVE в сервисе).
    Expected loss считается по одобренным: PD * LGD * сумма займа."""
    approved = proba < threshold

    exposure = loan_amnt[approved].sum()
    expected_loss = float((proba[approved] * lgd * loan_amnt[approved]).sum())

    return {
        "approval_rate": float(approved.mean()),
        "default_rate_approved": float(y_true[approved].mean()) if approved.any() else float("nan"),
        "expected_loss_total": expected_loss,
        "expected_loss_rate": expected_loss / float(exposure) if exposure else float("nan"),
    }


def decision_breakdown(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Доля заявок и фактическая доля дефолтов в каждой зоне решения."""
    decision = np.where(proba < T_LOW, "approve", np.where(proba < T_HIGH, "review", "decline"))

    result = {}
    for name in ("approve", "review", "decline"):
        mask = decision == name
        result[f"share_{name}"] = float(mask.mean())
        if mask.any():
            result[f"default_rate_{name}"] = float(y_true[mask].mean())

    return result


def shap_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP-важность. Передаётся калиброванная модель: базовый CatBoost
    извлекается внутри get_explainer, как и в сервисе, поэтому цифры
    совпадают с объяснениями в интерфейсе (лог-шансы базовой модели)."""
    if len(X) > SHAP_SAMPLE_SIZE:
        X = X.sample(SHAP_SAMPLE_SIZE, random_state=SEED)

    values = get_explainer(model)(X).values

    if values.ndim == 3:
        values = values[:, :, 1]

    return (
        pd.DataFrame({
            "feature": list(X.columns),
            "mean_abs_shap": np.abs(values).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def plot_importance(importance: pd.DataFrame, top_n: int = TOP_N_FEATURES):
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 1.5))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("mean |SHAP| (лог-шансы базовой модели)")
    ax.set_title("SHAP-важность признаков (калиброванная модель)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Гейт champion
# --------------------------------------------------------------------------- #

GATE_METRICS = ["Gini", "PR-AUC", "Brier Score"]


def champion_metrics(client: MlflowClient, test: pd.DataFrame):
    """Возвращает (версия, метрики, источник). Champion пересчитывается на
    текущем test: если данные или разбиение изменились, метрики из старого
    запуска были бы несравнимы. Если пересчитать не вышло, берутся метрики
    запуска; если и их нет, метрики None."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None, None, "none"

    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
        meta = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/metadata.json")
        frame = prepare_frame(test, meta["features"], meta["categorical_cols"])
        proba = model.predict_proba(frame)[:, 1]
        return mv.version, compute_ml_metrics(test[TARGET].to_numpy(), proba), "rescored"
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось пересчитать champion v{mv.version}: {exc!r}")

    stored = client.get_run(mv.run_id).data.metrics
    if all(name in stored for name in GATE_METRICS):
        return mv.version, {name: stored[name] for name in GATE_METRICS}, "stored"

    return mv.version, None, "unavailable"


def gate(new: dict, old: dict) -> tuple[bool, list[str]]:
    """Возвращает (повышать ли, причины отказа)."""
    reasons = []

    gain = new["Gini"] - old["Gini"]
    if gain <= MIN_GAIN_GINI:
        reasons.append(f"Gini gain {gain:.6f} <= {MIN_GAIN_GINI}")

    pr_auc_change = new["PR-AUC"] - old["PR-AUC"]
    if pr_auc_change < -MAX_PR_AUC_DROP:
        reasons.append(f"PR-AUC dropped by {-pr_auc_change:.6f} (limit {MAX_PR_AUC_DROP})")

    brier_change = new["Brier Score"] - old["Brier Score"]
    if brier_change > MAX_BRIER_INCREASE:
        reasons.append(f"Brier Score grew by {brier_change:.6f} (limit {MAX_BRIER_INCREASE})")

    return not reasons, reasons


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #

def main() -> dict:
    if not T_LOW <= T_HIGH:
        raise ValueError(f"T_LOW не должен превышать T_HIGH: {T_LOW} и {T_HIGH}")

    data_md5 = file_md5(DATA_PATH)
    df = load_data(DATA_PATH)

    train, valid, test = time_based_split(df)
    valid_es, valid_calib = split_valid_for_calibration(valid)
    validate_splits(train=train, valid_es=valid_es, valid_calib=valid_calib, test=test)
    print(
        f"train: {len(train)}, valid_es: {len(valid_es)}, "
        f"valid_calib: {len(valid_calib)}, test: {len(test)}"
    )

    X_train, y_train = prepare_frame(train, FEATURES, CAT_FEATURES), train[TARGET]
    X_es, y_es = prepare_frame(valid_es, FEATURES, CAT_FEATURES), valid_es[TARGET]
    X_calib, y_calib = prepare_frame(valid_calib, FEATURES, CAT_FEATURES), valid_calib[TARGET]
    X_test, y_test = prepare_frame(test, FEATURES, CAT_FEATURES), test[TARGET].to_numpy()

    base_model = train_catboost(X_train, y_train, X_es, y_es)
    raw_proba = base_model.predict_proba(X_test)[:, 1]

    calibrated = calibrate_model(base_model, X_calib, y_calib)
    proba = calibrated.predict_proba(X_test)[:, 1]

    raw_metrics = compute_ml_metrics(y_test, raw_proba)
    metrics = compute_ml_metrics(y_test, proba, (proba >= T_LOW).astype(int))
    business = compute_business_metrics(
        y_test, test["loan_amnt"].to_numpy(), proba, threshold=T_LOW, lgd=LGD
    )
    breakdown = decision_breakdown(y_test, proba)

    importance = shap_importance(calibrated, X_calib)

    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    with mlflow.start_run() as run:
        metadata = {
            "model_name": MODEL_NAME,
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "n_train": int(len(train)),
            "features": FEATURES,
            "categorical_cols": CAT_FEATURES,
            "t_low": round(T_LOW, 4),
            "t_high": round(T_HIGH, 4),
            "calibration": CALIBRATION_METHOD,
            "ml_metrics": {k: round(v, 4) for k, v in metrics.items()},
            "business_metrics": {k: round(v, 4) for k, v in business.items()},
            "libs": {
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
                "catboost": catboost.__version__,
            },
        }
        mlflow.log_params({
            "TOTAL_ITERATIONS": TOTAL_ITERATIONS,
            "LEARNING_RATE": LEARNING_RATE,
            "DEPTH": DEPTH,
            "EVAL_METRIC": EVAL_METRIC,
            "EARLY_STOPPING_ROUNDS": EARLY_STOPPING_ROUNDS,
            "best_iteration": base_model.get_best_iteration(),
            "calibration": CALIBRATION_METHOD,
            "calib_frac": CALIB_FRAC,
            "t_low": T_LOW,
            "t_high": T_HIGH,
            "lgd": LGD,
            "train_end": TRAIN_END,
            "valid_end": VALID_END,
            "test_end": TEST_END,
            "model": "CatBoostClassifier+CalibratedClassifierCV",
            "seed": SEED,
            "data": str(DATA_PATH),
            "data_md5": data_md5,
        })
        mlflow.log_metrics({
            **metrics,
            **{f"raw_{k}": v for k, v in raw_metrics.items()},
            **{k: v for k, v in business.items() if np.isfinite(v)},
            **breakdown,
        })
        mlflow.log_dict(metadata, "metadata.json")
        mlflow.log_dict(compute_calibration(y_test, raw_proba), "calibration_raw.json")
        mlflow.log_dict(compute_calibration(y_test, proba), "calibration_calibrated.json")

        fig = plot_importance(importance)
        mlflow.log_figure(fig, "shap_importance.png")
        plt.close(fig)
        mlflow.log_dict(
            {row.feature: float(row.mean_abs_shap) for row in importance.itertuples()},
            "shap_importance.json",
        )

        info = mlflow.sklearn.log_model(
            calibrated,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle",
        )
        version = info.registered_model_version

    old_version, old_metrics, source = champion_metrics(client, test)

    if old_version is None:
        promoted, reasons = True, ["no champion yet"]
    elif old_metrics is None:
        promoted, reasons = False, [f"champion v{old_version} could not be evaluated"]
    else:
        promoted, reasons = gate(metrics, old_metrics)

    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)

    client.set_tag(run.info.run_id, "gate", "promoted" if promoted else "rejected")
    client.set_tag(run.info.run_id, "gate_reasons", "; ".join(reasons))

    result = {
        "run_id": run.info.run_id,
        "version": version,
        "gini": round(metrics["Gini"], 4),
        "champion_before": old_version,
        "champion_metrics_before": old_metrics,
        "champion_metrics_source": source,
        "promoted": promoted,
        "gate_reasons": reasons,
    }
    print(json.dumps(result, ensure_ascii=False))

    xcom = Path("/airflow/xcom")
    if xcom.is_dir():
        (xcom / "return.json").write_text(json.dumps(result))

    return result


if __name__ == "__main__":
    main()