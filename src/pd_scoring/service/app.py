import logging
import time
import uuid
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from pd_scoring import db
from pd_scoring.config import settings


class Features(BaseModel):
    model_config = {"extra": "forbid"}

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


class Prediction(BaseModel):
    request_id: str
    score: float
    default: bool
    model_version: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(settings.model_path)

    app.state.model = bundle['model']
    app.state.meta = bundle['metadata']
    app.state.version = bundle['metadata']['model_version']

    db.init()
    yield
    app.state.model = None


app = FastAPI(title="pd-scoring-service", version="1.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": getattr(app.state, "version", "unknown"),
        "model_path": settings.model_path,
    }


@app.get("/ready")
def ready():
    if getattr(app.state, "model", "None") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"status": "ready"}


logger = logging.getLogger(__name__)

async def _safe_body(request: Request):
    try:
        return await request.json()
    except Exception:
        raw = await request.body()
        return {"raw": raw.decode("utf-8", errors="replace")}


PREDICT_PATHS = {"/v1/predict", "/v1/predict/batch"}

def _log_failure_task(
        request_id: str,
        body,
        status_code: int,
        request: Request
)-> BackgroundTask | None:
    if request.url.path not in PREDICT_PATHS:
        return None
    
    return BackgroundTask(
        db.save_prediction,
        request_id,
        body,
        None,
        getattr(app.state, "version", "unknown"),
        None,
        status_code,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = str(uuid.uuid4())
    body = await _safe_body(request)

    return JSONResponse(
        status_code=422,
        content={"request_id": request_id, "detail": jsonable_encoder(exc.errors())},
        background=_log_failure_task(request_id, body, 422, request),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    request_id = str(uuid.uuid4())
    body = await _safe_body(request)

    return JSONResponse(
        status_code=500,
        content={"request_id": request_id, "detail": "Internal server error"},
        background=_log_failure_task(request_id, body, 500, request),
    )


@app.post("/v1/predict")
def predict(x: Features, bg: BackgroundTasks) -> Prediction:
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    payload = x.model_dump()
    frame = pd.DataFrame([payload]).reindex(columns=app.state.meta["features"])

    score = float(app.state.model.predict_proba(frame)[0, 1])

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    bg.add_task(db.save_prediction, request_id, payload, score, app.state.version, latency_ms, 200)

    default = score >= app.state.meta["threshold"]

    return Prediction(
        score=score,
        default=default,
        model_version=app.state.version,
        request_id=request_id,
        latency_ms=latency_ms,
    )


@app.post("/v1/predict/batch")
def predict_batch(x: BatchFeatures, bg: BackgroundTasks) -> list[Prediction]:
    t0 = time.perf_counter()

    payloads = [row.model_dump() for row in x.rows]

    frame = pd.DataFrame(payloads).reindex(columns=app.state.meta["features"])

    scores = app.state.model.predict_proba(frame)[:, 1]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    results = []
    for payload, score in zip(payloads, scores, strict=True):
        score = float(score)
        request_id = str(uuid.uuid4())
        
        bg.add_task(
            db.save_prediction,
            request_id,
            payload,
            score,
            app.state.version,
            latency_ms,
            200,
        )

        default = score >= app.state.meta["threshold"]
        
        results.append(
            Prediction(
                score=score,
                default=default,
                model_version=app.state.version,
                request_id=request_id,
                latency_ms=latency_ms
            )
        )

    return results
