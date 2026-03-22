import os
import time
import json
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.lightgbm
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Any, Union
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from mlflow.tracking import MlflowClient
from src.inference.schemas import PredictionResponse, Item, BatchRequest
from src.inference.helpers import inference_preprocessing

# MLflow config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "porto-training")
DEFAULT_ALIAS = os.getenv("DEFAULT_ALIAS", "production")
PREP_PATH = os.getenv("PREP_PATH", "/data/training/processed/preprocessors.pkl")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# FastAPI
app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Prometheus metrics definitions
REQUEST_COUNTER = Counter("inference_requests_total", "Total inference requests", ["endpoint", "model", "status"])
ERROR_COUNTER = Counter("inference_errors_total", "Total errors", ["endpoint", "model"])
INVALID_PAYLOAD = Counter("invalid_payload_total", "Invalid payloads", ["endpoint", "model"])
INFERENCE_TIME = Histogram("inference_latency_seconds", "Inference latency", ["endpoint", "model"], buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
MODEL_LOAD_TIME = Histogram("model_load_latency_seconds", "Model load latency", ["endpoint", "model", "version"], buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
PREPROCESSING_TIME = Histogram("preprocessing_latency_seconds", "Preprocessing latency", ["endpoint", "model"], buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
CACHE_HIT = Counter("model_cache_hit_total", "Model cache hits", ["model", "version"])
CACHE_MISS = Counter("model_cache_miss_total", "Model cache misses", ["model", "version"])
INFERENCE_INFLIGHT = Gauge("inference_inflight", "In-flight inference requests", ["endpoint"])
MISSING_FEATURES = Counter("input_feature_missing_total", "Missing or null feature values", ["endpoint", "model"])
FEATURE_PREP_ERRORS = Counter("feature_preprocessing_errors_total", "Feature preprocessing errors", ["endpoint", "model"])
MODEL_LOAD_ERRORS = Counter("model_load_errors_total", "Model load errors", ["endpoint", "model"])
PREDICT_ERRORS = Counter("model_prediction_errors_total", "Model prediction errors", ["model"])
PREDICTION_VALUE = Histogram("prediction_distribution", "Prediction value distribution", ["endpoint", "model", "version"], buckets=[i/20 for i in range(21)])
BATCH_SIZE = Histogram("batch_size", "Batch size")
REQUEST_PAYLOAD_BYTES = Histogram("inference_request_bytes", "Inference request payload size in bytes", ["endpoint"])
RESPONSE_PAYLOAD_BYTES = Histogram("inference_response_bytes", "Inference response payload size in bytes", ["endpoint"])
PAGE_VIEWS = Counter("page_views_total", "Total page hits", ["endpoint"])

# Model cache
MODEL_CACHE = {}
CACHE_LIMIT = 5


def _cache_key(model_ref: str, version: str) -> str:
    return f"{MODEL_NAME}:{model_ref}:{version}"


def cache_model(key, model, version: str):
    """Cache a model instance with a simple size limit."""
    if len(MODEL_CACHE) >= CACHE_LIMIT:
        MODEL_CACHE.clear()
    MODEL_CACHE[key] = {"model": model, "version": version}


def _normalize_model_ref(model_ref: str) -> str:
    """Normalize a model reference into a concrete alias or version token."""
    if not model_ref or model_ref == "default":
        return DEFAULT_ALIAS
    return model_ref


def _candidate_model_uris(model_ref: str) -> List[str]:
    """Build a list of MLflow model URIs to try for a given reference."""
    if str(model_ref).isdigit():
        return [f"models:/{MODEL_NAME}/{model_ref}"]
    return [
        f"models:/{MODEL_NAME}@{model_ref}",
        f"models:/{MODEL_NAME}/{model_ref}",
        f"models:/{MODEL_NAME}/{str(model_ref).capitalize()}",
    ]


def _latest_version():
    """Return the latest registered model version as a string."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))


def _resolve_model_version(model_ref: str) -> str:
    """Resolve a model reference (alias/stage/version) to a concrete version string."""
    if not model_ref:
        return "unknown"
    if str(model_ref).isdigit():
        return str(model_ref)
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, model_ref)
        return str(mv.version)
    except Exception:
        pass
    try:
        latest = client.get_latest_versions(MODEL_NAME, stages=[str(model_ref).capitalize()])
        if latest:
            return str(latest[0].version)
    except Exception:
        pass
    return "unknown"


def _artifact_source_exists(source: str) -> bool:
    """Check whether a model artifact source is locally reachable."""
    if not source:
        return False
    if source.startswith("file://"):
        return Path(source.replace("file://", "", 1)).exists()
    if source.startswith("/"):
        return Path(source).exists()
    return True


def _sorted_versions(only_accessible: bool = False) -> List[int]:
    """Return model versions sorted by newest, optionally filtering to accessible artifacts."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    out = []
    for v in versions:
        if not only_accessible or _artifact_source_exists(getattr(v, "source", "")):
            out.append(int(v.version))
    return sorted(out, reverse=True)


def _extract_aliases(registered_model):
    """Extract and normalize alias names from an MLflow registered model."""
    aliases = getattr(registered_model, "aliases", {}) or {}
    if isinstance(aliases, dict):
        return sorted(aliases.keys())
    try:
        return sorted([a.alias for a in aliases])
    except Exception:
        return []


def _single_payload(item: Union[Item, Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize a single prediction payload into a flat feature dict."""
    if isinstance(item, dict):
        payload = item.get("root") or item.get("features") or item
        item_id = item.get("id")
    else:
        payload = item.root or item.features
        item_id = item.id

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Body must contain a feature dictionary in `root` or `features`.")

    if item_id is not None and "id" not in payload:
        payload = dict(payload)
        payload["id"] = item_id
    return payload


def _batch_payload(items: Union[BatchRequest, List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a batch payload into a list of feature dicts."""
    if isinstance(items, list):
        records = items
    elif isinstance(items, dict):
        records = items.get("root") or items.get("records") or items.get("items")
    else:
        records = items.root or items.records or items.items

    if not isinstance(records, list):
        raise HTTPException(status_code=422, detail="Body must contain a list of records in `root`, `records`, or `items`.")
    return records


def _load_model_any(model_uri: str):
    """Try multiple MLflow loaders to resolve a model URI."""
    loaders = (
        mlflow.sklearn.load_model,
        mlflow.lightgbm.load_model,
        mlflow.pyfunc.load_model,
    )
    errors = []
    for loader in loaders:
        try:
            return loader(model_uri)
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError(" | ".join(errors))


def _predict_scores(model_obj, X: pd.DataFrame):
    """Return score-like outputs, preferring class-1 probabilities when available."""
    if hasattr(model_obj, "predict_proba"):
        try:
            proba = model_obj.predict_proba(X)
            if hasattr(proba, "shape") and len(getattr(proba, "shape", [])) == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
        except Exception:
            pass

    preds = model_obj.predict(X)
    if hasattr(preds, "shape") and len(getattr(preds, "shape", [])) == 2 and preds.shape[1] >= 2:
        return preds[:, 1]
    return preds


def _count_missing_values(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> int:
    """Count missing or null values in a payload dict or list of dicts."""
    if isinstance(payload, dict):
        values = payload.values()
        return sum(1 for v in values if v is None or (isinstance(v, float) and pd.isna(v)))
    missing = 0
    for record in payload:
        missing += sum(1 for v in record.values() if v is None or (isinstance(v, float) and pd.isna(v)))
    return missing


def load_model_from_registry(model_ref: str):
    """Load a model from MLflow registry with caching and fallbacks."""
    model_ref = _normalize_model_ref(model_ref)
    version_hint = _resolve_model_version(model_ref)
    cache_key = _cache_key(model_ref, version_hint)
    if cache_key in MODEL_CACHE:
        cached = MODEL_CACHE[cache_key]
        version = cached.get("version", version_hint)
        CACHE_HIT.labels(model_ref, version).inc()
        return cached["model"], version
    CACHE_MISS.labels(model_ref, version_hint).inc()

    errors = []
    load_start = time.time()
    for model_uri in _candidate_model_uris(model_ref):
        try:
            model = _load_model_any(model_uri)
            resolved_version = _resolve_model_version(model_ref)
            cache_key = _cache_key(model_ref, resolved_version)
            cache_model(cache_key, model, resolved_version)
            MODEL_LOAD_TIME.labels("load_model", model_ref, resolved_version).observe(time.time() - load_start)
            return model, resolved_version
        except Exception as e:
            errors.append(f"{model_uri} -> {e}")

    if model_ref == DEFAULT_ALIAS:
        for ver in _sorted_versions(only_accessible=True):
            fallback_uri = f"models:/{MODEL_NAME}/{ver}"
            try:
                model = _load_model_any(fallback_uri)
                resolved_version = str(ver)
                cache_key = _cache_key(model_ref, resolved_version)
                cache_model(cache_key, model, resolved_version)
                MODEL_LOAD_TIME.labels("load_model", model_ref, resolved_version).observe(time.time() - load_start)
                return model, resolved_version
            except Exception as e:
                errors.append(f"{fallback_uri} -> {e}")

    MODEL_LOAD_ERRORS.labels("load_model", model_ref).inc()
    raise HTTPException(status_code=404, detail=f"Model '{model_ref}' not found. Tried: {' | '.join(errors)}")


@app.on_event("startup")
def load_default_model():
    """Warm the cache by loading the default model at startup."""
    try:
        _, version = load_model_from_registry(DEFAULT_ALIAS)
        print(f"Loaded default model alias: {DEFAULT_ALIAS}")
    except Exception as e:
        print(f"Failed to load default model: {e}")


@app.get("/models")
def list_available_models():
    """Return available model aliases and versions from the registry."""
    try:
        version_list = _sorted_versions(only_accessible=True)
        registered_model = client.get_registered_model(MODEL_NAME)
        aliases = _extract_aliases(registered_model)
        return {"aliases": aliases, "versions": version_list, "default": DEFAULT_ALIAS}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    """Return a basic service health response."""
    return {"status": "ok"}


@app.post("/ui/page_view")
def ui_page_view(page: str = Query(..., min_length=1)):
    """Record a UI page view for dashboards."""
    PAGE_VIEWS.labels(f"/ui/{page}").inc()
    return {"status": "ok", "page": page}

@app.post("/predict", response_model=PredictionResponse)
def predict(item: Union[Item, Dict[str, Any]], model: str = Query(DEFAULT_ALIAS)):
    """Run inference for a single payload and record metrics."""
    start = time.time()
    model = _normalize_model_ref(model)
    INFERENCE_INFLIGHT.labels("/predict").inc()
    try:
        payload = _single_payload(item)
        REQUEST_PAYLOAD_BYTES.labels("/predict").observe(len(json.dumps(payload).encode("utf-8")))
        missing = _count_missing_values(payload)
        if missing:
            MISSING_FEATURES.labels("/predict", model).inc(missing)

        model_obj, version = load_model_from_registry(model)
        df = pd.DataFrame([payload])
        prep_start = time.time()
        try:
            X = inference_preprocessing(df, PREP_PATH)
        except Exception:
            FEATURE_PREP_ERRORS.labels("/predict", model).inc()
            raise
        PREPROCESSING_TIME.labels("/predict", model).observe(time.time() - prep_start)

        try:
            pred = _predict_scores(model_obj, X)[0]
        except Exception:
            PREDICT_ERRORS.labels(model).inc()
            raise

        REQUEST_COUNTER.labels("/predict", model, "ok").inc()
        PREDICTION_VALUE.labels("/predict", model, version).observe(pred)

        response = {"prediction": float(pred)}
        RESPONSE_PAYLOAD_BYTES.labels("/predict").observe(len(json.dumps(response).encode("utf-8")))
        return response
    except Exception as e:
        if isinstance(e, HTTPException) and e.status_code == 422:
            INVALID_PAYLOAD.labels("/predict", model).inc()
        REQUEST_COUNTER.labels("/predict", model, "error").inc()
        ERROR_COUNTER.labels("/predict", model).inc()
        raise
    finally:
        INFERENCE_TIME.labels("/predict", model).observe(time.time() - start)
        INFERENCE_INFLIGHT.labels("/predict").dec()

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(items: Union[BatchRequest, List[Dict[str, Any]], Dict[str, Any]], model: str = Query(DEFAULT_ALIAS)):
    """Run inference for a batch payload and record metrics."""
    start = time.time()
    model = _normalize_model_ref(model)
    INFERENCE_INFLIGHT.labels("/predict/batch").inc()
    try:
        payload = _batch_payload(items)
        REQUEST_PAYLOAD_BYTES.labels("/predict/batch").observe(len(json.dumps(payload).encode("utf-8")))
        missing = _count_missing_values(payload)
        if missing:
            MISSING_FEATURES.labels("/predict/batch", model).inc(missing)

        model_obj, version = load_model_from_registry(model)
        df = pd.DataFrame(payload)
        prep_start = time.time()
        try:
            X = inference_preprocessing(df, PREP_PATH)
        except Exception:
            FEATURE_PREP_ERRORS.labels("/predict/batch", model).inc()
            raise
        PREPROCESSING_TIME.labels("/predict/batch", model).observe(time.time() - prep_start)

        try:
            preds = _predict_scores(model_obj, X)
        except Exception:
            PREDICT_ERRORS.labels(model).inc()
            raise

        REQUEST_COUNTER.labels("/predict/batch", model, "ok").inc()
        BATCH_SIZE.observe(len(payload))
        for p in preds:
            PREDICTION_VALUE.labels("/predict/batch", model, version).observe(p)

        response = [{"prediction": float(p)} for p in preds]
        RESPONSE_PAYLOAD_BYTES.labels("/predict/batch").observe(len(json.dumps(response).encode("utf-8")))
        return response
    except Exception as e:
        if isinstance(e, HTTPException) and e.status_code == 422:
            INVALID_PAYLOAD.labels("/predict/batch", model).inc()
        REQUEST_COUNTER.labels("/predict/batch", model, "error").inc()
        ERROR_COUNTER.labels("/predict/batch", model).inc()
        raise
    finally:
        INFERENCE_TIME.labels("/predict/batch", model).observe(time.time() - start)
        INFERENCE_INFLIGHT.labels("/predict/batch").dec()
