import math
import uuid
from datetime import UTC, date, datetime

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from pd_scoring.config import settings

# Столбцы заявки: сырые признаки модели в порядке обучения
APPLICATION_COLUMNS = [
    "loan_amnt", "term", "emp_title", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "purpose", "zip_code",
    "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "collections_12_mths_ex_med",
    "application_type", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
    "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal",
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
]

TEXT_COLUMNS = {
    "term", "emp_title", "emp_length", "home_ownership",
    "verification_status", "purpose", "zip_code", "addr_state",
    "application_type",
}
DATE_COLUMNS = {"issue_d", "earliest_cr_line"}  # остальные столбцы: double precision

# Столбцы, которые могут быть пустыми (по доле пропусков в обучающих данных);
# все остальные столбцы NOT NULL
NULLABLE_COLUMNS = {
    "mths_since_last_delinq", "mths_since_recent_inq", "num_tl_120dpd_2m",
    "mo_sin_old_il_acct", "emp_title", "emp_length", "pct_tl_nvr_dlq",
    "avg_cur_bal", "num_rev_accts", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "num_actv_rev_tl", "total_rev_hi_lim",
    "tot_cur_bal", "tot_coll_amt", "num_il_tl", "mo_sin_rcnt_tl",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_rev_tl_bal_gt_0",
    "num_tl_90g_dpd_24m", "num_tl_30dpd", "num_bc_tl",
    "total_il_high_credit_limit", "num_op_rev_tl", "num_tl_op_past_12m",
    "tot_hi_cred_lim", "bc_util", "percent_bc_gt_75", "bc_open_to_buy",
    "mths_since_recent_bc", "num_bc_sats", "num_sats", "total_bal_ex_mort",
    "mort_acc", "total_bc_limit", "acc_open_past_24mths", "revol_util",
    "pub_rec_bankruptcies", "dti", "collections_12_mths_ex_med",
    "chargeoff_within_12_mths", "tax_liens", "inq_last_6mths", "zip_code",
}
assert NULLABLE_COLUMNS <= set(APPLICATION_COLUMNS)


def _column_ddl(col: str) -> str:
    if col in DATE_COLUMNS:
        sql_type = "date"
    elif col in TEXT_COLUMNS:
        sql_type = "text"
    else:
        sql_type = "double precision"
    not_null = "" if col in NULLABLE_COLUMNS else " NOT NULL"
    return f"    {col} {sql_type}{not_null}"


_APPLICATION_FEATURES_DDL = ",\n".join(_column_ddl(c) for c in APPLICATION_COLUMNS)

DDL = f"""
CREATE TABLE IF NOT EXISTS applications (
    application_id  uuid PRIMARY KEY,
    created_at      timestamptz NOT NULL DEFAULT now(),
    scored          boolean NOT NULL DEFAULT false,
{_APPLICATION_FEATURES_DDL}
);

CREATE INDEX IF NOT EXISTS applications_unscored_idx
    ON applications (created_at) WHERE NOT scored;

CREATE TABLE IF NOT EXISTS predictions (
    application_id      uuid PRIMARY KEY REFERENCES applications (application_id),
    ts                  timestamptz NOT NULL DEFAULT now(),
    model_version       text NOT NULL,
    features            jsonb NOT NULL,
    score               double precision,
    latency_ms          real,
    response_code       integer NOT NULL,
    feature_importance  jsonb
);
"""

def init() -> None:
    if not settings.database_url:
        return
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(7001)")
        conn.execute(DDL)


def _clean(value):
    """NaN -> None, чтобы в БД попал NULL, а не NaN."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _to_date(value) -> date:
    """Принимает date/datetime, epoch в миллисекундах, 'YYYY-MM-DD' или 'Aug-2003'."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=UTC).date()
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return datetime.strptime(text, "%b-%Y").date()


def _db_value(col: str, value):
    value = _clean(value)
    if value is None:
        return None
    return _to_date(value) if col in DATE_COLUMNS else value


def save_applications(rows: list[dict]) -> list[str]:
    """Сохраняет заявки со scored = false и возвращает их application_id
    (в том же порядке). Если в row есть 'application_id', он сохраняется,
    иначе генерируется. Отсутствующие ключи и NaN записываются как NULL;
    для обязательных столбцов БД вернёт ошибку. Идентификаторы возвращаются
    и при выключенной БД, чтобы их можно было использовать дальше."""
    ids = [str(row.get("application_id") or uuid.uuid4()) for row in rows]
    if not settings.database_url or not rows:
        return ids

    columns = ", ".join(APPLICATION_COLUMNS)
    placeholders = ", ".join(["%s"] * (len(APPLICATION_COLUMNS) + 1))
    query = f"INSERT INTO applications (application_id, {columns}) VALUES ({placeholders})"

    params = [
        (application_id, *(_db_value(col, row.get(col)) for col in APPLICATION_COLUMNS))
        for application_id, row in zip(ids, rows, strict=True)
    ]
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, params)
    return ids


def fetch_unscored(limit: int | None = None) -> list[dict]:
    """Возвращает необработанные заявки (scored = false), самые старые первыми.
    application_id приводится к строке; issue_d и earliest_cr_line приходят
    как datetime.date, пропуски как None."""
    if not settings.database_url:
        return []
    columns = ", ".join(APPLICATION_COLUMNS)
    query = (
        f"SELECT application_id, {columns} FROM applications "
        "WHERE NOT scored ORDER BY created_at, application_id"
    )
    params: tuple = ()
    if limit is not None:
        query += " LIMIT %s"
        params = (limit,)

    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        rows = conn.execute(query, params).fetchall()
    for row in rows:
        row["application_id"] = str(row["application_id"])
    return rows


def mark_scored(application_ids: list[str]) -> int:
    """Ставит scored = true по списку application_id.
    Возвращает число обновлённых строк."""
    if not settings.database_url or not application_ids:
        return 0
    ids = [uuid.UUID(str(i)) for i in application_ids]
    with psycopg.connect(settings.database_url) as conn:
        cur = conn.execute(
            "UPDATE applications SET scored = true WHERE application_id = ANY(%s)",
            (ids,),
        )
        return cur.rowcount


def save_prediction(
    application_id: str,
    features: dict,
    score: float | None,
    model_version: str,
    latency_ms: float,
    response_code: int,
    feature_importance: list[dict] | None = None,
) -> None:
    """application_id должен совпадать с application_id существующей заявки."""
    if not settings.database_url:
        return
    query = """
        INSERT INTO predictions (
            application_id,
            model_version,
            features,
            score,
            latency_ms,
            response_code,
            feature_importance
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            query,
            (
                application_id,
                model_version,
                Json(features),
                score,
                latency_ms,
                response_code,
                Json(feature_importance) if feature_importance is not None else None,
            ),
        )


def save_predictions(records: list[dict]) -> None:
    """Сохраняет результаты скоринга пачкой и в той же транзакции ставит
    scored = true у соответствующих заявок: либо записываются и предсказания,
    и флаги, либо ничего.

    Каждая запись: application_id, features, score, model_version,
    latency_ms, response_code, feature_importance (необязательно)."""
    if not settings.database_url or not records:
        return
    query = """
        INSERT INTO predictions (
            application_id,
            model_version,
            features,
            score,
            latency_ms,
            response_code,
            feature_importance
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    params = []
    for record in records:
        importance = record.get("feature_importance")
        params.append(
            (
                record["application_id"],
                record["model_version"],
                Json(record["features"]),
                record["score"],
                record["latency_ms"],
                record["response_code"],
                Json(importance) if importance is not None else None,
            )
        )
    ids = [uuid.UUID(str(record["application_id"])) for record in records]

    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, params)
            cur.execute(
                "UPDATE applications SET scored = true WHERE application_id = ANY(%s)",
                (ids,),
            )


def fetch_predictions(
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int | None = None,
    details: bool = False,
) -> list[dict]:
    """Возвращает предсказания, самые новые первыми. Границы date_from и
    date_to включаются и считаются по UTC. details=True добавляет столбцы
    features и feature_importance."""
    if not settings.database_url:
        return []
    columns = "application_id, ts, model_version, score, latency_ms, response_code"
    if details:
        columns += ", features, feature_importance"

    conditions = []
    params: list = []
    if date_from is not None:
        conditions.append("ts >= (%s::date)::timestamp AT TIME ZONE 'UTC'")
        params.append(date_from)
    if date_to is not None:
        conditions.append("ts < ((%s::date) + 1)::timestamp AT TIME ZONE 'UTC'")
        params.append(date_to)

    query = f"SELECT {columns} FROM predictions"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY ts DESC, application_id"
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        rows = conn.execute(query, params).fetchall()
    for row in rows:
        row["application_id"] = str(row["application_id"])
    return rows