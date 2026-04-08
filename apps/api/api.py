from __future__ import annotations

import math
import os
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from sqlalchemy.orm import aliased
from starlette.middleware.sessions import SessionMiddleware

from apps.api.deps import get_current_user, get_db_session
from ml.model.inference import (
    DEFAULT_FX_RATE,
    LoadedModelBundle,
    load_model_bundle,
    predict_proxy_valuation_from_bundle,
    score_proxy_valuations_from_bundle,
)
from ml.model.utils import ARTIFACTS_DIR
from shared.auth import verify_password
from shared.db.models import Listing, ShortlistItem, User, Valuation


DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", ARTIFACTS_DIR / "best_model.joblib"))
API_DEFAULT_FX_RATE = float(os.getenv("DEFAULT_FX_RATE", str(DEFAULT_FX_RATE)))
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "real_estate_session")
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-session-secret-change-me")
SESSION_MAX_AGE_SECONDS = int(os.getenv("SESSION_MAX_AGE_SECONDS", str(60 * 60 * 12)))
DEFAULT_UI_ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def _parse_allowed_origins() -> list[str]:
    raw_value = os.getenv("UI_ALLOWED_ORIGINS", DEFAULT_UI_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


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


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthUserResponse(BaseModel):
    id: int
    name: str
    email: str


class OpportunityItem(BaseModel):
    listing_id: int
    title: str
    city: str | None = None
    district: str | None = None
    area: float | None = None
    rooms: int | None = None
    floor: int | None = None
    total_floors: int | None = None
    listing_price: float | None = None
    predicted_price: float
    undervaluation_delta: float
    undervaluation_percent: float
    score: float
    rank_position: int | None = None
    is_saved: bool = False


class OpportunityListResponse(BaseModel):
    items: list[OpportunityItem]


class SaveShortlistRequest(BaseModel):
    listing_id: int
    rank_position: int | None = Field(default=None, ge=1)


class ShortlistMutationResponse(BaseModel):
    listing_id: int
    saved: bool


app = FastAPI(
    title="Bishkek Real Estate MVP Proxy-Valuation API",
    version="0.2.0",
    description=(
        "MVP API for proxy valuation on listing data. "
        "Predictions represent expected_price_proxy, not a transaction-based market price."
    ),
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie=SESSION_COOKIE_NAME,
    max_age=SESSION_MAX_AGE_SECONDS,
    same_site="lax",
    https_only=False,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def _to_float(value: Decimal | float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _serialize_user(user: User) -> AuthUserResponse:
    return AuthUserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
    )


def _serialize_opportunity(row: dict[str, Any]) -> OpportunityItem:
    return OpportunityItem(
        listing_id=row["listing_id"],
        title=row["title"],
        city=row["city"],
        district=row["district"],
        area=_to_float(row["area"]),
        rooms=row["rooms"],
        floor=row["floor"],
        total_floors=row["total_floors"],
        listing_price=_to_float(row["listing_price"]),
        predicted_price=float(row["predicted_price"]),
        undervaluation_delta=float(row["undervaluation_delta"]),
        undervaluation_percent=float(row["undervaluation_percent"]),
        score=float(row["score"]),
        rank_position=row.get("rank_position"),
        is_saved=bool(row["is_saved"]),
    )


def _to_quantized_decimal(value: float, precision: str) -> Decimal:
    return Decimal(str(value)).quantize(Decimal(precision))


def _build_scoring_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "listing_id": row["listing_id"],
        "district": row["district"],
        "rooms": row["rooms"],
        "floor": row["floor"],
        "total_floors": row["total_floors"],
        "listing_price": _to_float(row["listing_price"]),
        "price_usd": _to_float(row["listing_price"]),
        "area": _to_float(row["area"]),
        "total_area_m2": _to_float(row["area"]),
    }


def _build_display_score(rank_position: int, total_items: int) -> float:
    if total_items <= 1:
        return 1.0
    return 1.0 - ((rank_position - 1) / (total_items - 1))


def _ensure_valuations(session: Session) -> None:
    missing_valuations_count = session.execute(
        select(func.count(Listing.id))
        .select_from(Listing)
        .outerjoin(Valuation, Valuation.listing_id == Listing.id)
        .where(
            Listing.listing_price.is_not(None),
            Valuation.id.is_(None),
        )
    ).scalar_one()

    if missing_valuations_count == 0:
        return

    listing_rows = session.execute(
        select(
            Listing.id.label("listing_id"),
            Listing.district.label("district"),
            Listing.area.label("area"),
            Listing.rooms.label("rooms"),
            Listing.floor.label("floor"),
            Listing.total_floors.label("total_floors"),
            Listing.listing_price.label("listing_price"),
        )
        .where(Listing.listing_price.is_not(None))
        .order_by(Listing.id.asc())
    ).mappings().all()

    if not listing_rows:
        return

    bundle = get_model_bundle()
    scored_results = score_proxy_valuations_from_bundle(
        objects=[_build_scoring_payload(row) for row in listing_rows],
        bundle=bundle,
        rank_results=True,
        include_explanations=False,
    )

    total_results = len(scored_results)
    valuation_rows: list[dict[str, Any]] = []
    for result in scored_results:
        listing_id = result.get("listing_id")
        price_outputs = result.get("price_outputs", {})
        usd_output = price_outputs.get("USD", {})
        if listing_id is None:
            continue

        rank_position = int(result.get("undervaluation_rank") or total_results)
        predicted_price = float(usd_output["expected_price_proxy"])
        undervaluation_delta = float(usd_output["delta_abs"] or 0.0)
        undervaluation_percent = float(usd_output["delta_pct"] or 0.0)
        score = _build_display_score(rank_position, total_results)

        valuation_rows.append(
            {
                "listing_id": int(listing_id),
                "predicted_price": _to_quantized_decimal(predicted_price, "0.01"),
                "undervaluation_delta": _to_quantized_decimal(undervaluation_delta, "0.01"),
                "undervaluation_percent": _to_quantized_decimal(undervaluation_percent, "0.0001"),
                "score": _to_quantized_decimal(score, "0.0001"),
            }
        )

    if not valuation_rows:
        return

    statement = insert(Valuation).values(valuation_rows)
    statement = statement.on_conflict_do_update(
        index_elements=[Valuation.listing_id],
        set_={
            "predicted_price": statement.excluded.predicted_price,
            "undervaluation_delta": statement.excluded.undervaluation_delta,
            "undervaluation_percent": statement.excluded.undervaluation_percent,
            "score": statement.excluded.score,
        },
    )
    session.execute(statement)
    session.commit()


def _build_opportunity_base_query(user_id: int):
    saved_shortlist_item = aliased(ShortlistItem)
    is_saved_expression = (
        select(saved_shortlist_item.id)
        .where(
            saved_shortlist_item.user_id == user_id,
            saved_shortlist_item.listing_id == Listing.id,
        )
        .exists()
    )

    return select(
        Listing.id.label("listing_id"),
        Listing.title.label("title"),
        Listing.city.label("city"),
        Listing.district.label("district"),
        Listing.area.label("area"),
        Listing.rooms.label("rooms"),
        Listing.floor.label("floor"),
        Listing.total_floors.label("total_floors"),
        Listing.listing_price.label("listing_price"),
        Valuation.predicted_price.label("predicted_price"),
        Valuation.undervaluation_delta.label("undervaluation_delta"),
        Valuation.undervaluation_percent.label("undervaluation_percent"),
        Valuation.score.label("score"),
        is_saved_expression.label("is_saved"),
    ).join(Valuation, Valuation.listing_id == Listing.id)


def _apply_sorting(statement, sort_by: Literal["score", "undervaluation_percent"]):
    if sort_by == "undervaluation_percent":
        return statement.order_by(
            Valuation.undervaluation_percent.desc(),
            Valuation.score.desc(),
            Listing.id.asc(),
        )
    return statement.order_by(
        Valuation.score.desc(),
        Valuation.undervaluation_percent.desc(),
        Listing.id.asc(),
    )


@app.get("/health")
def health() -> dict[str, str]:
    get_model_bundle()
    return {"status": "ok"}


@app.post("/auth/login", response_model=AuthUserResponse)
def login(
    payload: LoginRequest,
    request: Request,
    session: Session = Depends(get_db_session),
) -> AuthUserResponse:
    user = session.execute(
        select(User).where(User.email == payload.email.strip().lower())
    ).scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    request.session.clear()
    request.session["user_id"] = user.id
    return _serialize_user(user)


@app.post("/auth/logout")
def logout(request: Request) -> dict[str, str]:
    request.session.clear()
    return {"status": "ok"}


@app.get("/auth/me", response_model=AuthUserResponse)
def auth_me(current_user: User = Depends(get_current_user)) -> AuthUserResponse:
    return _serialize_user(current_user)


@app.get("/opportunities", response_model=OpportunityListResponse)
def get_opportunities(
    sort_by: Literal["score", "undervaluation_percent"] = Query(default="score"),
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> OpportunityListResponse:
    _ensure_valuations(session)
    statement = _build_opportunity_base_query(current_user.id)
    statement = _apply_sorting(statement, sort_by).limit(limit)
    rows = session.execute(statement).mappings().all()
    return OpportunityListResponse(items=[_serialize_opportunity(row) for row in rows])


@app.get("/shortlist", response_model=OpportunityListResponse)
def get_shortlist(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> OpportunityListResponse:
    _ensure_valuations(session)
    statement = (
        _build_opportunity_base_query(current_user.id)
        .join(
            ShortlistItem,
            (ShortlistItem.listing_id == Listing.id) & (ShortlistItem.user_id == current_user.id),
        )
        .add_columns(ShortlistItem.rank_position.label("rank_position"))
        .order_by(
            ShortlistItem.rank_position.asc(),
            Valuation.score.desc(),
            Listing.id.asc(),
        )
    )
    rows = session.execute(statement).mappings().all()
    return OpportunityListResponse(items=[_serialize_opportunity(row) for row in rows])


@app.post("/shortlist", response_model=ShortlistMutationResponse)
def save_shortlist_item(
    payload: SaveShortlistRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> ShortlistMutationResponse:
    _ensure_valuations(session)
    listing_exists = session.execute(
        select(Listing.id).where(Listing.id == payload.listing_id)
    ).scalar_one_or_none()
    if listing_exists is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found.")

    existing_item = session.execute(
        select(ShortlistItem).where(
            ShortlistItem.user_id == current_user.id,
            ShortlistItem.listing_id == payload.listing_id,
        )
    ).scalar_one_or_none()

    if existing_item is None:
        max_rank = session.execute(
            select(func.max(ShortlistItem.rank_position)).where(ShortlistItem.user_id == current_user.id)
        ).scalar_one()
        next_rank = int(max_rank or 0) + 1
        existing_item = ShortlistItem(
            user_id=current_user.id,
            listing_id=payload.listing_id,
            rank_position=payload.rank_position or next_rank,
        )
        session.add(existing_item)
    elif payload.rank_position is not None:
        existing_item.rank_position = payload.rank_position

    session.commit()
    return ShortlistMutationResponse(listing_id=payload.listing_id, saved=True)


@app.delete("/shortlist/{listing_id}", response_model=ShortlistMutationResponse)
def delete_shortlist_item(
    listing_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> ShortlistMutationResponse:
    existing_item = session.execute(
        select(ShortlistItem).where(
            ShortlistItem.user_id == current_user.id,
            ShortlistItem.listing_id == listing_id,
        )
    ).scalar_one_or_none()

    if existing_item is not None:
        session.delete(existing_item)
        session.commit()

    return ShortlistMutationResponse(listing_id=listing_id, saved=False)


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
        "login_url": "/auth/login",
        "current_user_url": "/auth/me",
        "opportunities_url": "/opportunities",
        "shortlist_url": "/shortlist",
        "predict_url": "/predict",
        "batch_predict_url": "/predict/batch",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
