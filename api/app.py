"""
FastAPI inference server.

Startup: loads the best model (by roc_auc) from MLflow local store.

POST /predict  — accepts feature key/values, returns prediction + probability
GET  /health   — liveness check
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

TRACKING_URI = "file:///app/mlruns"
EXPERIMENT_NAME = "german_fintech_classifier"

_model = None


def load_best_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found. Run training first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished runs found. Run training first.")

    run_id = runs[0].info.run_id
    roc_auc = runs[0].data.metrics.get("roc_auc", 0)
    print(f"[serve] Loading model from run {run_id} (roc_auc={roc_auc:.4f})")

    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_best_model()
    print("[serve] Model ready.")
    yield
    _model = None


app = FastAPI(
    title="German FinTech Classifier",
    description="Predicts whether a company has been rebranded (has a former name).",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    features: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def accept_flat(cls, data: Any) -> Any:
        # Allow flat JSON: {"Local court": "Berlin", ...}
        if isinstance(data, dict) and "features" not in data:
            return {"features": data}
        return data


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    input_df = pd.DataFrame([request.features])

    # Align to the exact columns the model was trained on
    try:
        trained_cols = _model.named_steps["clf"].feature_names_in_.tolist()
        for col in trained_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[trained_cols]
    except AttributeError:
        pass

    prediction = int(_model.predict(input_df)[0])
    probability = round(float(_model.predict_proba(input_df)[0][1]), 4)

    return PredictResponse(prediction=prediction, probability=probability)
