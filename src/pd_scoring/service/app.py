import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from pd_scoring import db
from pd_scoring.config import settings
from pd_scoring.service.explain import (
    explain_prediction,
    explain_predictions,
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = Path("uploads")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class Features(BaseModel):
    model_config = {"extra": "forbid"}

    id: int = Field(gt=0)
    revenue: float = Field(ge=0)
    dti_n: float = Field(ge=0)
    loan_amnt: float = Field(gt=0)
    fico_n: float = Field(ge=300, le=850)
    experience_c: int = Field(ge=0, le=1)
    emp_length: str
    purpose: str
    home_ownership_n: str
    addr_state: str
    zip_code: str


class BatchFeatures(BaseModel):
    model_config = {"extra": "forbid"}

    rows: list[Features] = Field(min_length=1, max_length=1000)


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


class BatchPrediction(BaseModel):
    request_id: str
    features: dict
    score: float
    threshold: float
    default: bool
    decision: str
    explanations: list[Explanation]


class BatchResult(BaseModel):
    filename: str
    total: int
    approved: int
    rejected: int
    preview: list[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(settings.model_path)

    app.state.model = bundle["model"]
    app.state.shap_available_model = bundle["shap_available_model"]
    app.state.meta = bundle["metadata"]
    app.state.version = bundle["metadata"]["model_version"]

    db.init()

    yield

    app.state.model = None
    app.state.shap_available_model = None


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


logger = logging.getLogger(__name__)


async def _safe_body(request: Request):
    try:
        return await request.json()
    except Exception:
        raw = await request.body()

        return {
            "raw": raw.decode(
                "utf-8",
                errors="replace",
            )
        }


PREDICT_PATHS = {
    "/v1/predict",
    "/v1/predict/batch"
}


def _log_failure_task(
    request_id: str,
    body,
    status_code: int,
    request: Request,
):
    if request.url.path not in PREDICT_PATHS:
        return None

    return db.save_prediction(
        request_id,
        body,
        None,
        getattr(app.state, "version", "unknown"),
        0.0,
        status_code,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    request_id = str(uuid.uuid4())
    body = await _safe_body(request)

    return JSONResponse(
        status_code=422,
        content={
            "request_id": request_id,
            "detail": jsonable_encoder(exc.errors()),
        },
        background=_log_failure_task(
            request_id,
            body,
            422,
            request,
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled error on %s",
        request.url.path,
    )

    request_id = str(uuid.uuid4())
    body = await _safe_body(request)

    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "detail": "Internal server error",
        },
        background=_log_failure_task(
            request_id,
            body,
            500,
            request,
        ),
    )


def _prepare_frame(payloads: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(payloads).reindex(
        columns=app.state.meta["features"]
    )


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
        top_k=5,
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


def _score_batch(
    payloads: list[dict],
    bg: BackgroundTasks,
) -> list[BatchPrediction]:
    t0 = time.perf_counter()

    frame = _prepare_frame(payloads)

    scores = app.state.model.predict_proba(frame)[:, 1]

    explanations = explain_predictions(
        app.state.shap_available_model,
        frame,
        top_k=5,
    )

    latency_ms = round(
        (time.perf_counter() - t0) * 1000,
        2,
    )

    threshold = float(app.state.meta["threshold"])

    results = []

    for payload, score, row_explanations in zip(
        payloads,
        scores,
        explanations,
        strict=True,
    ):
        score = float(score)
        request_id = str(uuid.uuid4())
        default = score >= threshold

        bg.add_task(
            db.save_prediction,
            request_id,
            payload,
            score,
            app.state.version,
            latency_ms,
            200,
        )

        results.append(
            BatchPrediction(
                request_id=request_id,
                features=payload,
                score=score,
                threshold=threshold,
                default=default,
                decision=(
                    "REJECT"
                    if default
                    else "APPROVE"
                ),
                explanations=[
                    Explanation(**item)
                    for item in row_explanations
                ],
            )
        )

    return results


def _format_batch_row(
    result: BatchPrediction,
) -> dict:
    features = ", ".join(
        f"{key}={value}"
        for key, value in result.features.items()
    )

    reasons = "; ".join(
        (
            f"{item.feature} "
            f"({'+' if item.impact > 0 else ''}{item.impact:.4f})"
        )
        for item in result.explanations
    )

    return {
        "features": features,
        "P(default)": result.score,
        "class": "DEFAULT" if result.default else "NO DEFAULT",
        "decision": result.decision,
        "main reasons": reasons,
    }


def _save_batch_csv(
    results: list[BatchPrediction],
) -> str:
    rows = [
        _format_batch_row(result)
        for result in results
    ]

    result_df = pd.DataFrame(rows)

    filename = (
        f"batch_result_"
        f"{uuid.uuid4().hex}.csv"
    )

    result_df.to_csv(
        UPLOADS_DIR / filename,
        index=False,
    )

    return filename


@app.post(
    "/v1/predict",
    response_model=Prediction,
)
def predict(
    x: Features,
    bg: BackgroundTasks,
) -> Prediction:
    t0 = time.perf_counter()

    request_id = str(uuid.uuid4())
    payload = x.model_dump()

    frame = _prepare_frame([payload])

    score = float(
        app.state.model.predict_proba(frame)[0, 1]
    )

    latency_ms = round(
        (time.perf_counter() - t0) * 1000,
        2,
    )

    bg.add_task(
        db.save_prediction,
        request_id,
        payload,
        score,
        app.state.version,
        latency_ms,
        200,
    )

    return _build_prediction(
        app.state.shap_available_model,
        frame,
        score,
        request_id,
        latency_ms,
    )


@app.post(
    "/v1/predict/batch",
    response_model=BatchResult,
)
async def predict_batch(
    bg: BackgroundTasks,
    file: UploadFile,
) -> BatchResult:
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

    content = await file.read()

    try:
        from io import BytesIO

        df = pd.read_csv(
            BytesIO(content),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid CSV file: {exc}",
        ) from exc

    feature_names = app.state.meta["features"]

    missing = [
        column
        for column in feature_names
        if column not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required columns",
                "columns": missing,
            },
        )

    if df.isnull().any().any():
        raise HTTPException(
            status_code=422,
            detail="CSV contains missing values",
        )

    string_features = [
        "emp_length",
        "purpose",
        "home_ownership_n",
        "addr_state",
        "zip_code",
    ]

    for column in string_features:
        if column in df.columns:
            df[column] = df[column].astype(str)

    try:
        rows = df.to_dict(
            orient="records"
        )

        batch = BatchFeatures(
            rows=rows,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=jsonable_encoder(exc.errors())
            if hasattr(exc, "errors")
            else str(exc),
        ) from exc

    results = _score_batch(
        [row.model_dump() for row in batch.rows],
        bg,
    )

    filename = _save_batch_csv(results)

    approved = sum(
        not result.default
        for result in results
    )

    rejected = sum(
        result.default
        for result in results
    )

    preview = [
        _format_batch_row(result)
        for result in results[:10]
    ]

    return BatchResult(
        filename=filename,
        total=len(results),
        approved=approved,
        rejected=rejected,
        preview=preview,
    )