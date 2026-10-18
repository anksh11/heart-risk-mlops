"""FastAPI service exposing the heart-disease risk model.

Endpoints
    GET  /health         liveness + model status
    GET  /model-info     metadata of the loaded model
    POST /predict        single patient -> prediction + confidence
    POST /predict/batch  many patients
    GET  /metrics        Prometheus exposition format
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from api.schemas import BatchRequest, BatchResponse, PatientFeatures, PredictionResponse
from heart_risk import config
from heart_risk.predict import load_meta, load_model, predict_records

MODEL_PATH = Path(os.environ.get("MODEL_PATH", config.MODEL_FILE))
META_PATH = Path(os.environ.get("MODEL_META_PATH", config.MODEL_META_FILE))
THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))


class JsonFormatter(logging.Formatter):
    """One JSON object per line – easy to ship to Loki/ELK/CloudWatch."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
log = logging.getLogger("heart_api")
log.handlers = [handler]
log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
log.propagate = False

REQUESTS = Counter("heart_api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram(
    "heart_api_request_latency_seconds",
    "Request latency",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions served", ["label", "risk_band"])
PROBABILITY = Histogram(
    "heart_api_predicted_probability",
    "Distribution of predicted disease probability",
    buckets=tuple(i / 10 for i in range(1, 11)),
)
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if the model is loaded")
MODEL_INFO = Gauge("heart_api_model_info", "Loaded model metadata", ["model_name", "model_version"])

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state["model"] = load_model(MODEL_PATH)
        state["meta"] = load_meta(META_PATH)
        MODEL_LOADED.set(1)
        MODEL_INFO.labels(state["meta"].get("model_name", "unknown"), state["meta"].get("model_version", "?")).set(1)
        log.info("model loaded", extra={"extra_fields": {"path": str(MODEL_PATH), **_model_labels()}})
    except FileNotFoundError:
        MODEL_LOADED.set(0)
        log.error("model file not found", extra={"extra_fields": {"path": str(MODEL_PATH)}})
    yield
    state.clear()


app = FastAPI(
    title="Heart Disease Risk API",
    version="1.0.0",
    description="Predicts presence of heart disease from 13 UCI clinical attributes.",
    lifespan=lifespan,
)


def _model_labels() -> dict:
    meta = state.get("meta", {})
    return {"model_name": meta.get("model_name", "unknown"), "model_version": str(meta.get("model_version", "?"))}


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.scope.get("route").path if request.scope.get("route") else request.url.path
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(elapsed)
            log.info(
                "request",
                extra={
                    "extra_fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": path,
                        "status": status,
                        "latency_ms": round(elapsed * 1000, 2),
                        "client": request.client.host if request.client else None,
                    }
                },
            )


def _require_model():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return state["model"]


def _score(records: list[dict], request_id: str) -> list[dict]:
    preds = predict_records(_require_model(), records, threshold=THRESHOLD)
    for rec, p in zip(records, preds, strict=True):
        PREDICTIONS.labels(p["label"], p["risk_band"]).inc()
        PROBABILITY.observe(p["probability_disease"])
        log.info("prediction", extra={"extra_fields": {"request_id": request_id, "input": rec, **p}})
    return preds


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "degraded", "model_loaded": "model" in state, **_model_labels()}


@app.get("/model-info")
def model_info():
    _require_model()
    return state["meta"] | {"decision_threshold": THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    pred = _score([patient.model_dump()], request.state.request_id)[0]
    return {**pred, **_model_labels(), "request_id": request.state.request_id}


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    preds = _score([p.model_dump() for p in batch.patients], request.state.request_id)
    return {"predictions": preds, **_model_labels(), "request_id": request.state.request_id}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
