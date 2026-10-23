import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
)

from pd_scoring import db
from pd_scoring.config import settings
from pd_scoring.service.explain import (
    explain_prediction,
    explain_predictions,
    get_base_estimator,
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = Path("uploads")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

TOP_K = 5
BATCH_LIMIT = 1000          # заявок за один запуск batch-скоринга
UNSCORED_LIMIT = 1000       # строк в таблице необработанных заявок
PREDICTIONS_LIMIT = 1000    # строк в таблице predictions (экспорт без лимита)
MAX_UPLOAD_ROWS = 10_000
MAX_REPORTED_ERRORS = 20

# Признаки с высокой кардинальностью: при обучении строки нормализовались
# (пропуск -> "missing", обрезка пробелов, нижний регистр)
HIGH_CARDINALITY = ["emp_title", "zip_code"]

# Признаки для краткого описания заявки в таблице и CSV с результатами
SUMMARY_COLUMNS = [
    "loan_amnt", "term", "purpose", "annual_inc", "dti", "fico_range_low",
]


# --------------------------------------------------------------------------- #
# Схема заявки
# --------------------------------------------------------------------------- #

def _parse_date(value):
    """Принимает date/datetime, 'YYYY-MM-DD', 'Aug-2003' и epoch в миллисекундах."""
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(value / 1000, tz=UTC).date()

    if isinstance(value, str):
        text = value.strip()

        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            pass

        try:
            return datetime.strptime(text, "%b-%Y").date()
        except ValueError:
            pass

    raise ValueError(
        "expected a date: YYYY-MM-DD, Mon-YYYY or epoch milliseconds"
    )


# Ограничения значений (остальные числовые поля без ограничений)
FIELD_CONSTRAINTS = {
    "loan_amnt": {"gt": 0},
    "annual_inc": {"ge": 0},
    "dti": {"ge": 0},
    "revol_util": {"ge": 0},
    "fico_range_low": {"ge": 300, "le": 850},
    "fico_range_high": {"ge": 300, "le": 850},
}


def _field_definitions() -> dict:
    """Поля строятся из списков столбцов в db.py, чтобы схема API
    и таблица applications не расходились."""
    definitions = {}

    for column in db.APPLICATION_COLUMNS:
        if column in db.DATE_COLUMNS:
            annotation = Annotated[date, BeforeValidator(_parse_date)]
        elif column in db.TEXT_COLUMNS:
            annotation = str
        else:
            annotation = float

        constraints = FIELD_CONSTRAINTS.get(column, {})

        if column in db.NULLABLE_COLUMNS:
            definitions[column] = (
                annotation | None,
                Field(None, **constraints),
            )
        else:
            definitions[column] = (annotation, Field(..., **constraints))

    return definitions


Features = create_model(
    "Features",
    __config__=ConfigDict(extra="forbid"),
    **_field_definitions(),
)


class Explanation(BaseModel):
    feature: str
    value: object
    impact: float
    direction: str


class Prediction(BaseModel):
    request_id: str
    score: float
    threshold: float
    default: bool
    decision: str
    explanations: list[Explanation]
    model_version: str
    latency_ms: float


class BatchResult(BaseModel):
    filename: str
    total: int
    approved: int
    rejected: int
    preview: list[dict]


class UploadResult(BaseModel):
    added: int


# --------------------------------------------------------------------------- #
# Приложение
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(settings.model_path)

    app.state.model = bundle["model"]
    # В артефакте одна калиброванная модель: для SHAP из неё берётся базовая
    app.state.explain_model = get_base_estimator(bundle["model"])
    app.state.meta = bundle["metadata"]
    app.state.version = bundle["metadata"]["model_version"]

    db.init()

    yield

    app.state.model = None
    app.state.explain_model = None


app = FastAPI(
    title="pd-scoring-service",
    version="1.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOADS_DIR),
    name="uploads",
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Batch-скоринг берёт заявки из БД и помечает их обработанными:
# параллельные запуски в одном процессе выполняются по очереди
_batch_lock = threading.Lock()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": getattr(app.state, "version", "unknown"),
        "model_path": settings.model_path,
    }


@app.get("/ready")
def ready():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {"status": "ready"}


def _require_db() -> None:
    if not settings.database_url:
        raise HTTPException(
            status_code=503,
            detail="Database is not configured",
        )


# --------------------------------------------------------------------------- #
# Обработчики ошибок
# --------------------------------------------------------------------------- #

# Ошибки не пишутся в predictions: у predictions внешний ключ на applications,
# а заявки с неверным телом в applications не попадают.

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    request_id = str(uuid.uuid4())

    logger.warning(
        "Validation error on %s (request_id=%s)",
        request.url.path,
        request_id,
    )

    return JSONResponse(
        status_code=422,
        content={
            "request_id": request_id,
            "detail": jsonable_encoder(exc.errors()),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    request_id = str(uuid.uuid4())

    logger.exception(
        "Unhandled error on %s (request_id=%s)",
        request.url.path,
        request_id,
    )

    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "detail": "Internal server error",
        },
    )


# --------------------------------------------------------------------------- #
# Подготовка признаков и скоринг
# --------------------------------------------------------------------------- #

def build_model_frame(
    payloads: list[dict],
    features: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    """Превращает заявки в вход модели так же, как при обучении:
    - credit_history_months из earliest_cr_line на дату issue_d;
    - emp_title и zip_code: пропуск -> "missing", strip, нижний регистр;
    - остальные категории: пропуск -> "missing", приведение к str;
    - числовые пропуски остаются NaN (CatBoost обрабатывает их сам).
    """
    frame = pd.DataFrame(payloads)

    issue = pd.to_datetime(frame["issue_d"], errors="coerce")
    earliest = pd.to_datetime(frame["earliest_cr_line"], errors="coerce")

    frame["credit_history_months"] = (
        (issue.dt.year - earliest.dt.year) * 12
        + (issue.dt.month - earliest.dt.month)
    )

    for column in HIGH_CARDINALITY:
        frame[column] = (
            frame[column]
            .fillna("missing")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    for column in categorical:
        frame[column] = frame[column].fillna("missing").astype(str)

    numeric = [column for column in features if column not in categorical]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")

    return frame.reindex(columns=features)


def _prepare_frame(payloads: list[dict]) -> pd.DataFrame:
    meta = app.state.meta
    features = meta["features"]

    categorical = meta.get("categorical_cols") or [
        column for column in features if column in db.TEXT_COLUMNS
    ]

    return build_model_frame(payloads, features, categorical)


def _build_prediction(
    model,
    frame: pd.DataFrame,
    score: float,
    request_id: str,
    latency_ms: float,
) -> Prediction:
    threshold = float(app.state.meta["threshold"])
    default = score >= threshold

    explanations = explain_prediction(
        model,
        frame,
        top_k=TOP_K,
    )

    return Prediction(
        request_id=request_id,
        score=score,
        threshold=threshold,
        default=default,
        decision="REJECT" if default else "APPROVE",
        explanations=[
            Explanation(**item)
            for item in explanations
        ],
        model_version=app.state.version,
        latency_ms=latency_ms,
    )


def _format_batch_row(
    application_id: str,
    payload: dict,
    score: float,
    default: bool,
    explanations: list[dict],
) -> dict:
    features = ", ".join(
        f"{key}={payload.get(key)}"
        for key in SUMMARY_COLUMNS
    )

    reasons = "; ".join(
        (
            f"{item['feature']} "
            f"({'+' if item['impact'] > 0 else ''}{item['impact']:.4f})"
        )
        for item in explanations
    )

    return {
        "application_id": application_id,
        "features": features,
        "P(default)": score,
        "class": "DEFAULT" if default else "NO DEFAULT",
        "decision": "REJECT" if default else "APPROVE",
        "main reasons": reasons,
    }


def _save_batch_csv(rows: list[dict]) -> str:
    filename = (
        f"batch_result_"
        f"{uuid.uuid4().hex}.csv"
    )

    pd.DataFrame(rows).to_csv(
        UPLOADS_DIR / filename,
        index=False,
    )

    return filename


def _persist_single(
    application_id: str,
    payload: dict,
    prediction: Prediction,
) -> None:
    """Сохраняет заявку из /v1/predict и результат скоринга.
    Если запись предсказания не удалась, заявка остаётся необработанной
    и будет оценена при следующем batch-скоринге."""
    if not settings.database_url:
        return

    db.save_applications([{**payload, "application_id": application_id}])
    db.save_predictions([
        {
            "application_id": application_id,
            "features": payload,
            "score": prediction.score,
            "model_version": prediction.model_version,
            "latency_ms": prediction.latency_ms,
            "response_code": 200,
            "feature_importance": [
                item.model_dump() for item in prediction.explanations
            ],
        }
    ])


# --------------------------------------------------------------------------- #
# Одиночный скоринг (для тестов и ручных проверок)
# --------------------------------------------------------------------------- #

@app.post(
    "/v1/predict",
    response_model=Prediction,
)
def predict(
    x: "Features",
    bg: BackgroundTasks,
) -> Prediction:
    t0 = time.perf_counter()

    application_id = str(uuid.uuid4())
    payload = x.model_dump(mode="json")

    frame = _prepare_frame([payload])

    score = float(
        app.state.model.predict_proba(frame)[0, 1]
    )

    latency_ms = round(
        (time.perf_counter() - t0) * 1000,
        2,
    )

    prediction = _build_prediction(
        app.state.explain_model,
        frame,
        score,
        application_id,
        latency_ms,
    )

    bg.add_task(
        _persist_single,
        application_id,
        payload,
        prediction,
    )

    return prediction


# --------------------------------------------------------------------------- #
# Заявки: загрузка из CSV и список необработанных
# --------------------------------------------------------------------------- #

@app.post(
    "/v1/applications/upload",
    response_model=UploadResult,
)
def upload_applications(file: UploadFile) -> UploadResult:
    _require_db()

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="CSV file is required",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported",
        )

    try:
        df = pd.read_csv(
            BytesIO(file.file.read()),
            dtype=str,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid CSV file: {exc}",
        ) from exc

    columns = set(df.columns)
    expected = set(db.APPLICATION_COLUMNS)

    missing = [c for c in db.APPLICATION_COLUMNS if c not in columns]
    unexpected = sorted(columns - expected)

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Wrong CSV columns",
                "missing": missing,
                "unexpected": unexpected,
            },
        )

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail="The file has no rows with applications",
        )

    if len(df) > MAX_UPLOAD_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"At most {MAX_UPLOAD_ROWS} rows per file",
        )

    records = df.astype(object).where(df.notna(), None).to_dict(
        orient="records"
    )

    rows = []
    errors = []

    for number, record in enumerate(records, start=2):
        try:
            rows.append(Features(**record).model_dump(mode="json"))
        except ValidationError as exc:
            errors.append(
                {
                    "row": number,
                    "errors": jsonable_encoder(
                        exc.errors(
                            include_url=False,
                            include_context=False,
                            include_input=False,
                        )
                    ),
                }
            )

            if len(errors) >= MAX_REPORTED_ERRORS:
                break

    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid rows, nothing was added",
                "rows": errors,
            },
        )

    ids = db.save_applications(rows)

    return UploadResult(added=len(ids))


@app.get("/v1/applications/unscored")
def unscored_applications():
    _require_db()

    return {
        "items": jsonable_encoder(
            db.fetch_unscored(limit=UNSCORED_LIMIT)
        )
    }


# --------------------------------------------------------------------------- #
# Batch-скоринг необработанных заявок из БД
# --------------------------------------------------------------------------- #

@app.post(
    "/v1/predict/batch",
    response_model=BatchResult,
)
def predict_batch() -> BatchResult:
    _require_db()

    with _batch_lock:
        applications = db.fetch_unscored(limit=BATCH_LIMIT)

        if not applications:
            return BatchResult(
                filename="",
                total=0,
                approved=0,
                rejected=0,
                preview=[],
            )

        ids = [row.pop("application_id") for row in applications]
        payloads = [jsonable_encoder(row) for row in applications]

        t0 = time.perf_counter()

        frame = _prepare_frame(payloads)
        scores = app.state.model.predict_proba(frame)[:, 1]

        latency_ms = round(
            (time.perf_counter() - t0) * 1000,
            2,
        )

        explanations = explain_predictions(
            app.state.explain_model,
            frame,
            top_k=TOP_K,
        )

        threshold = float(app.state.meta["threshold"])

        records = []
        rows = []

        for application_id, payload, score, row_explanations in zip(
            ids,
            payloads,
            scores,
            explanations,
            strict=True,
        ):
            score = float(score)
            default = score >= threshold

            records.append(
                {
                    "application_id": application_id,
                    "features": payload,
                    "score": score,
                    "model_version": app.state.version,
                    "latency_ms": latency_ms,
                    "response_code": 200,
                    "feature_importance": row_explanations,
                }
            )

            rows.append(
                _format_batch_row(
                    application_id,
                    payload,
                    score,
                    default,
                    row_explanations,
                )
            )

        db.save_predictions(records)

    filename = _save_batch_csv(rows)

    rejected = sum(row["class"] == "DEFAULT" for row in rows)

    return BatchResult(
        filename=filename,
        total=len(rows),
        approved=len(rows) - rejected,
        rejected=rejected,
        preview=rows[:10],
    )


# --------------------------------------------------------------------------- #
# Predictions: таблица и выгрузка в CSV
# --------------------------------------------------------------------------- #

def _check_dates(date_from: date | None, date_to: date | None) -> None:
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=422,
            detail="date_from must not be later than date_to",
        )


@app.get("/v1/predictions")
def list_predictions(
    date_from: date | None = None,
    date_to: date | None = None,
):
    _require_db()
    _check_dates(date_from, date_to)

    return {
        "items": jsonable_encoder(
            db.fetch_predictions(
                date_from,
                date_to,
                limit=PREDICTIONS_LIMIT,
            )
        )
    }


@app.get("/v1/predictions/export")
def export_predictions(
    date_from: date | None = None,
    date_to: date | None = None,
):
    _require_db()
    _check_dates(date_from, date_to)

    columns = [
        "application_id", "ts", "model_version", "score",
        "latency_ms", "response_code", "features", "feature_importance",
    ]

    rows = db.fetch_predictions(date_from, date_to, details=True)

    frame = pd.DataFrame(jsonable_encoder(rows), columns=columns)

    for column in ("features", "feature_importance"):
        frame[column] = frame[column].map(
            lambda value: None
            if value is None
            else json.dumps(value, ensure_ascii=False)
        )

    filename = (
        f"predictions_{date_from or 'start'}_{date_to or 'end'}.csv"
    )

    return Response(
        content=frame.to_csv(index=False),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )