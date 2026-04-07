from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import (
    DEFAULT_FX_RATE,
    LoadedModelBundle,
    load_model_bundle,
    predict_proxy_valuation_from_bundle,
    score_proxy_valuations_from_bundle,
)
from utils import ARTIFACTS_DIR


DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", ARTIFACTS_DIR / "best_model.joblib"))
API_DEFAULT_FX_RATE = float(os.getenv("DEFAULT_FX_RATE", str(DEFAULT_FX_RATE)))


class CurrencyPriceOutput(BaseModel):
    expected_price_proxy: float
    listing_price: float | None = None
    delta_abs: float | None = None
    delta_pct: float | None = None


class PredictionResponse(BaseModel):
    base_currency: str
    output_currency: str
    fx_rate_used: float | None = None
    price_outputs: dict[str, CurrencyPriceOutput]
    top_factors: list[str] = Field(default_factory=list)
    explanation_summary: str | None = None
    valuation_note: str


class SinglePredictionRequest(BaseModel):
    object_features: dict[str, Any] = Field(
        ...,
        description="Listing attributes used for MVP proxy valuation.",
    )
    output_currency: Literal["USD", "RUB", "BOTH"] = Field(
        default="USD",
        description="Return price outputs in USD, RUB, or both currencies.",
    )
    fx_rate: float | None = Field(
        default=None,
        gt=0,
        description="USD/RUB conversion rate. If omitted for RUB/BOTH, API fallback is used.",
    )
    include_explanation: bool = Field(
        default=True,
        description="If true, include lightweight local explainability in the response.",
    )


class BatchPredictionRequest(BaseModel):
    objects: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of listing objects for batch scoring.",
    )
    rank_by_undervaluation: bool = Field(
        default=True,
        description="If true, sort output by delta_pct descending and add undervaluation_rank.",
    )
    output_currency: Literal["USD", "RUB", "BOTH"] = Field(
        default="USD",
        description="Return price outputs in USD, RUB, or both currencies.",
    )
    fx_rate: float | None = Field(
        default=None,
        gt=0,
        description="USD/RUB conversion rate. If omitted for RUB/BOTH, API fallback is used.",
    )
    include_explanations: bool = Field(
        default=False,
        description="If true, compute lightweight explanation blocks for each object.",
    )


class BatchPredictionItem(BaseModel):
    input_index: int | None = None
    listing_id: str | None = None
    base_currency: str
    output_currency: str
    fx_rate_used: float | None = None
    price_outputs: dict[str, CurrencyPriceOutput]
    top_factors: list[str] = Field(default_factory=list)
    explanation_summary: str | None = None
    valuation_note: str
    undervaluation_rank: int | None = None


class BatchPredictionResponse(BaseModel):
    count: int
    ranked: bool
    results: list[BatchPredictionItem]


app = FastAPI(
    title="Bishkek Real Estate MVP Proxy-Valuation API",
    version="0.1.0",
    description=(
        "MVP API for proxy valuation on listing data. "
        "Predictions represent expected_price_proxy, not a transaction-based market price."
    ),
)


@lru_cache(maxsize=1)
def get_model_bundle() -> LoadedModelBundle:
    model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    return load_model_bundle(model_path)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


@app.get("/health")
def health() -> dict[str, str]:
    get_model_bundle()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: SinglePredictionRequest) -> PredictionResponse:
    try:
        bundle = get_model_bundle()
        result = predict_proxy_valuation_from_bundle(
            object_features=request.object_features,
            bundle=bundle,
            output_currency=request.output_currency,
            fx_rate=request.fx_rate,
            default_fx_rate=API_DEFAULT_FX_RATE,
            include_explanation=request.include_explanation,
        )
        return PredictionResponse(**_sanitize_for_json(result))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        bundle = get_model_bundle()
        objects_with_index = []
        for index, item in enumerate(request.objects):
            enriched_item = dict(item)
            enriched_item["input_index"] = index
            objects_with_index.append(enriched_item)

        results = score_proxy_valuations_from_bundle(
            objects=objects_with_index,
            bundle=bundle,
            output_currency=request.output_currency,
            fx_rate=request.fx_rate,
            default_fx_rate=API_DEFAULT_FX_RATE,
            rank_results=request.rank_by_undervaluation,
            include_explanations=request.include_explanations,
        )
        return BatchPredictionResponse(
            count=len(results),
            ranked=request.rank_by_undervaluation,
            results=_sanitize_for_json(results),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {exc}") from exc


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "bishkek-real-estate-mvp-api",
        "mode": "proxy-valuation",
        "base_currency": "USD",
        "default_fx_rate": API_DEFAULT_FX_RATE,
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
        "batch_predict_url": "/predict/batch",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
