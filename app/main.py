"""FastAPI application for Blinkit order delivery-time predictions."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from app.features import FEATURES, normalize_payload

logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/delivery_time_model.joblib"))
FRONTEND_PATH = Path(__file__).resolve().parent.parent / "templates" / "index.html"


class DeliveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    order_date: date | None = Field(default=None, description="Order date in YYYY-MM-DD format.")
    order_hour: int | None = Field(default=None, ge=0, le=23)
    city: str = Field(min_length=1, max_length=80)
    dark_store_id: str | None = Field(default=None, max_length=80)
    category: str = Field(min_length=1, max_length=100)
    quantity: float | None = Field(default=None, ge=0, le=10000)
    unit_price: float | None = Field(default=None, ge=0, le=1_000_000)
    order_value: float | None = Field(default=None, ge=0, le=10_000_000)
    distance_km: float | None = Field(default=None, ge=0, le=500)
    payment_mode: str | None = Field(default=None, max_length=80)


class DeliveryPrediction(BaseModel):
    predicted_delivery_mins: float
    model: str
    unit: Literal["minutes"] = "minutes"


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.bundle = None
    if MODEL_PATH.is_file():
        try:
            bundle = joblib.load(MODEL_PATH)
            if not isinstance(bundle, dict) or "model" not in bundle:
                raise ValueError("The model artifact is missing its 'model' pipeline.")
            application.state.bundle = bundle
            logger.info("Loaded delivery-time model '%s'.", bundle.get("model_name", "unknown"))
        except (OSError, ValueError, EOFError, KeyError, ImportError) as exc:
            logger.exception("Could not load model artifact at %s.", MODEL_PATH)
            raise RuntimeError(f"Invalid model artifact at {MODEL_PATH}: {exc}") from exc
    else:
        logger.warning(
            "Model artifact %s does not exist. Train the model before using /predict.",
            MODEL_PATH,
        )
    yield


app = FastAPI(
    title="Blinkit Delivery Time Prediction API",
    description="Predict completed-order delivery duration from checkout-time features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    try:
        return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.exception("Could not read the prediction UI at %s.", FRONTEND_PATH)
        raise HTTPException(status_code=500, detail="The prediction page could not be loaded.") from exc


@app.get("/health", tags=["health"])
def health(request: Request) -> dict[str, str | bool]:
    bundle = getattr(request.app.state, "bundle", None)
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model": bundle.get("model_name", "unavailable") if bundle else "unavailable",
    }


@app.post("/predict", response_model=DeliveryPrediction, tags=["prediction"])
def predict(payload: DeliveryRequest, request: Request) -> DeliveryPrediction:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is not trained yet. Run `python train.py` first.",
        )
    try:
        features = normalize_payload(payload.model_dump(exclude_none=False))
        prediction = float(bundle["model"].predict(features.loc[:, FEATURES])[0])
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        logger.exception("Could not produce a delivery-time prediction.")
        raise HTTPException(status_code=422, detail=f"Could not process prediction: {exc}") from exc
    if not np.isfinite(prediction):
        logger.error("The loaded model returned a non-finite delivery-time prediction.")
        raise HTTPException(status_code=500, detail="The model returned an invalid prediction.")
    return DeliveryPrediction(
        predicted_delivery_mins=round(max(0.0, prediction), 2),
        model=str(bundle.get("model_name", "unknown")),
    )
