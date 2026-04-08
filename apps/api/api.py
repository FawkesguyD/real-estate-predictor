from __future__ import annotations

import math
import os
import logging
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
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
DEFAULT_LISTING_CURRENCY = "RUB"
MODEL_BASE_CURRENCY = "USD"
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "real_estate_session")
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-session-secret-change-me")
SESSION_MAX_AGE_SECONDS = int(os.getenv("SESSION_MAX_AGE_SECONDS", str(60 * 60 * 12)))
DEFAULT_UI_ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
LOGGER = logging.getLogger("uvicorn.error")

LISTING_TO_MODEL_FIELD_MAP: dict[str, str] = {
    "district": "district",
    "rooms": "rooms",
    "floor": "floor",
    "total_floors": "total_floors",
    "total_area_m2": "area",
    "living_area_m2": "living_area_m2",
    "kitchen_area_m2": "kitchen_area_m2",
    "ceiling_height": "ceiling_height",
    "building_type": "building_type",
    "building_series": "building_series",
    "year_built": "year_built",
    "condition": "condition",
    "heating": "heating",
    "gas_supply": "gas_supply",
    "bathroom": "bathroom",
    "balcony": "balcony",
    "parking": "parking",
    "furniture": "furniture",
    "flooring": "flooring",
    "door_type": "door_type",
    "has_landline_phone": "has_landline_phone",
    "internet": "internet",
    "mortgage": "mortgage",
    "seller_type": "seller_type",
    "latitude": "latitude",
    "longitude": "longitude",
    "photo_count": "photo_count",
}


def _parse_allowed_origins() -> list[str]:
    raw_value = os.getenv("UI_ALLOWED_ORIGINS", DEFAULT_UI_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


class CurrencyPriceOutput(BaseModel):
    expected_price_proxy: float
    comparison_currency: str
    predicted_price_currency: str
    listing_price_in_comparison_currency: float | None = None
    delta_abs: float | None = None
    delta_pct: float | None = None


class PredictionResponse(BaseModel):
    base_currency: str
    output_currency: str
    listing_price: float | None = None
    listing_currency: str
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
    listing_price: float | None = None
    listing_currency: str
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
    building_type: str | None = None
    condition: str | None = None
    year_built: int | None = None
    seller_type: str | None = None
    listing_price: float | None = None
    listing_currency: str
    listing_price_in_comparison_currency: float | None = None
    predicted_price: float
    predicted_price_currency: str
    comparison_currency: str
    fx_rate_used: float | None = None
    delta_abs: float
    delta_pct: float
    score: float
    explanation_summary: str
    top_factors: list[str] = Field(default_factory=list)
    source_url: str | None = None
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


def _normalize_currency(currency: str | None, default: str = DEFAULT_LISTING_CURRENCY) -> str:
    normalized = default if currency is None else str(currency)
    normalized = normalized.upper()
    if normalized not in {"USD", "RUB"}:
        raise ValueError(f"Unsupported currency '{currency}'.")
    return normalized


def _convert_amount(amount: float | None, source_currency: str, target_currency: str, fx_rate: float | None) -> float | None:
    if amount is None:
        return None

    normalized_source = _normalize_currency(source_currency)
    normalized_target = _normalize_currency(target_currency, default=MODEL_BASE_CURRENCY)
    if normalized_source == normalized_target:
        return amount

    if fx_rate is None or fx_rate <= 0:
        raise ValueError("fx_rate must be positive when currency conversion is required.")

    if normalized_source == "USD" and normalized_target == "RUB":
        return amount * fx_rate
    if normalized_source == "RUB" and normalized_target == "USD":
        return amount / fx_rate

    raise ValueError(f"Unsupported currency conversion {normalized_source}->{normalized_target}.")


def _resolve_rows_fx_rate(
    rows: list[dict[str, Any]],
    output_currency: str,
    fx_rate: float | None,
) -> float | None:
    if output_currency != MODEL_BASE_CURRENCY:
        return float(fx_rate or API_DEFAULT_FX_RATE)

    requires_fx = any(
        _normalize_currency(row.get("listing_currency"), default=DEFAULT_LISTING_CURRENCY)
        != MODEL_BASE_CURRENCY
        for row in rows
    )
    if not requires_fx:
        return None
    return float(fx_rate or API_DEFAULT_FX_RATE)


def _build_comparison_metrics(
    *,
    listing_price: float | None,
    listing_currency: str,
    predicted_price_base: float,
    comparison_currency: str,
    fx_rate: float | None,
) -> dict[str, float | None]:
    predicted_price = _convert_amount(predicted_price_base, MODEL_BASE_CURRENCY, comparison_currency, fx_rate)
    listing_price_in_comparison_currency = _convert_amount(
        listing_price,
        listing_currency,
        comparison_currency,
        fx_rate,
    )
    delta_abs = 0.0
    delta_pct = 0.0
    if predicted_price is not None and listing_price_in_comparison_currency is not None:
        delta_abs = predicted_price - listing_price_in_comparison_currency
        if listing_price_in_comparison_currency != 0:
            delta_pct = delta_abs / listing_price_in_comparison_currency

    return {
        "predicted_price": predicted_price,
        "listing_price_in_comparison_currency": listing_price_in_comparison_currency,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
    }


def _fallback_explanation_summary() -> str:
    return (
        "Model estimate is a proxy valuation based on listing data. Use it as a screening signal, "
        "not as a transaction-based fair market price."
    )


def _serialize_user(user: User) -> AuthUserResponse:
    return AuthUserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
    )


def _serialize_opportunity(
    row: dict[str, Any],
    *,
    comparison_currency: str,
    fx_rate_used: float | None,
) -> OpportunityItem:
    listing_currency = _normalize_currency(row.get("listing_currency"), default=DEFAULT_LISTING_CURRENCY)
    comparison_metrics = _build_comparison_metrics(
        listing_price=_to_float(row["listing_price"]),
        listing_currency=listing_currency,
        predicted_price_base=float(row["predicted_price"]),
        comparison_currency=comparison_currency,
        fx_rate=fx_rate_used,
    )

    return OpportunityItem(
        listing_id=row["listing_id"],
        title=row["title"],
        city=row["city"],
        district=row["district"],
        area=_to_float(row["area"]),
        rooms=row["rooms"],
        floor=row["floor"],
        total_floors=row["total_floors"],
        building_type=row.get("building_type"),
        condition=row.get("condition"),
        year_built=row.get("year_built"),
        seller_type=row.get("seller_type"),
        listing_price=_to_float(row["listing_price"]),
        listing_currency=listing_currency,
        listing_price_in_comparison_currency=comparison_metrics["listing_price_in_comparison_currency"],
        predicted_price=float(comparison_metrics["predicted_price"] or 0.0),
        predicted_price_currency=comparison_currency,
        comparison_currency=comparison_currency,
        fx_rate_used=fx_rate_used,
        delta_abs=float(comparison_metrics["delta_abs"] or 0.0),
        delta_pct=float(comparison_metrics["delta_pct"] or 0.0),
        score=float(row["score"]),
        explanation_summary=row.get("explanation_summary") or _fallback_explanation_summary(),
        top_factors=list(row.get("top_factors") or []),
        source_url=row.get("source_url"),
        rank_position=row.get("rank_position"),
        is_saved=bool(row["is_saved"]),
    )


def _to_quantized_decimal(value: float, precision: str) -> Decimal:
    return Decimal(str(value)).quantize(Decimal(precision))


def _build_scoring_payload(row: dict[str, Any], bundle: LoadedModelBundle) -> dict[str, Any]:
    supported_features = set(bundle.feature_config.feature_columns)
    payload: dict[str, Any] = {
        "listing_id": row["listing_id"],
        "listing_price": _to_float(row["listing_price"]),
        "listing_currency": _normalize_currency(row.get("listing_currency"), default=DEFAULT_LISTING_CURRENCY),
    }

    for model_field, source_field in LISTING_TO_MODEL_FIELD_MAP.items():
        if model_field not in supported_features:
            continue
        raw_value = row.get(source_field)
        payload[model_field] = _to_float(raw_value) if isinstance(raw_value, Decimal) else raw_value

    return payload


def _build_valuation_listing_query(*, only_missing: bool):
    statement = (
        select(
            Listing.id.label("listing_id"),
            Listing.district.label("district"),
            Listing.area.label("area"),
            Listing.living_area_m2.label("living_area_m2"),
            Listing.kitchen_area_m2.label("kitchen_area_m2"),
            Listing.rooms.label("rooms"),
            Listing.floor.label("floor"),
            Listing.total_floors.label("total_floors"),
            Listing.ceiling_height.label("ceiling_height"),
            Listing.building_type.label("building_type"),
            Listing.building_series.label("building_series"),
            Listing.year_built.label("year_built"),
            Listing.condition.label("condition"),
            Listing.heating.label("heating"),
            Listing.gas_supply.label("gas_supply"),
            Listing.bathroom.label("bathroom"),
            Listing.balcony.label("balcony"),
            Listing.parking.label("parking"),
            Listing.furniture.label("furniture"),
            Listing.flooring.label("flooring"),
            Listing.door_type.label("door_type"),
            Listing.has_landline_phone.label("has_landline_phone"),
            Listing.internet.label("internet"),
            Listing.mortgage.label("mortgage"),
            Listing.seller_type.label("seller_type"),
            Listing.latitude.label("latitude"),
            Listing.longitude.label("longitude"),
            Listing.photo_count.label("photo_count"),
            Listing.listing_price.label("listing_price"),
            Listing.listing_currency.label("listing_currency"),
        )
        .where(Listing.listing_price.is_not(None))
        .order_by(Listing.id.asc())
    )

    if only_missing:
        statement = (
            statement.outerjoin(Valuation, Valuation.listing_id == Listing.id)
            .where(Valuation.id.is_(None))
        )

    return statement


def _recalculate_valuation_scores(session: Session) -> None:
    session.execute(
        text(
            """
            WITH ranked AS (
              SELECT
                id,
                row_number() OVER (
                  ORDER BY undervaluation_percent DESC, listing_id ASC
                ) AS rank_position,
                count(*) OVER () AS total_items
              FROM valuations
            )
            UPDATE valuations AS valuation
            SET score = CASE
              WHEN ranked.total_items <= 1 THEN 1.0000
              ELSE ROUND(
                1 - ((ranked.rank_position - 1)::numeric / (ranked.total_items - 1)),
                4
              )
            END
            FROM ranked
            WHERE valuation.id = ranked.id
            """
        )
    )


def ensure_listing_valuations(
    session: Session,
    *,
    only_missing: bool = True,
    include_explanations: bool = False,
) -> int:
    listing_rows = session.execute(
        _build_valuation_listing_query(only_missing=only_missing)
    ).mappings().all()

    if not listing_rows:
        LOGGER.debug(
            "valuation_backfill_skipped only_missing=%s include_explanations=%s",
            only_missing,
            include_explanations,
        )
        return 0

    started_at = perf_counter()
    bundle = get_model_bundle()
    scored_results = score_proxy_valuations_from_bundle(
        objects=[_build_scoring_payload(row, bundle) for row in listing_rows],
        bundle=bundle,
        rank_results=False,
        include_explanations=include_explanations,
    )

    valuation_rows: list[dict[str, Any]] = []
    for result in scored_results:
        listing_id = result.get("listing_id")
        price_outputs = result.get("price_outputs", {})
        usd_output = price_outputs.get("USD", {})
        if listing_id is None:
            continue

        predicted_price = float(usd_output["expected_price_proxy"])
        undervaluation_delta = float(usd_output["delta_abs"] or 0.0)
        undervaluation_percent = float(usd_output["delta_pct"] or 0.0)

        row_payload = {
            "listing_id": int(listing_id),
            "predicted_price": _to_quantized_decimal(predicted_price, "0.01"),
            "undervaluation_delta": _to_quantized_decimal(undervaluation_delta, "0.01"),
            "undervaluation_percent": _to_quantized_decimal(undervaluation_percent, "0.0001"),
            "score": _to_quantized_decimal(0.0, "0.0001"),
        }
        if include_explanations:
            row_payload["explanation_summary"] = result.get("explanation_summary")
            row_payload["top_factors"] = list(result.get("top_factors") or [])

        valuation_rows.append(row_payload)

    if not valuation_rows:
        return 0

    statement = insert(Valuation).values(valuation_rows)
    update_mapping = {
        "predicted_price": statement.excluded.predicted_price,
        "undervaluation_delta": statement.excluded.undervaluation_delta,
        "undervaluation_percent": statement.excluded.undervaluation_percent,
        "score": statement.excluded.score,
    }
    if include_explanations:
        update_mapping["explanation_summary"] = statement.excluded.explanation_summary
        update_mapping["top_factors"] = statement.excluded.top_factors

    statement = statement.on_conflict_do_update(
        index_elements=[Valuation.listing_id],
        set_=update_mapping,
    )
    session.execute(statement)
    _recalculate_valuation_scores(session)
    session.commit()
    elapsed_ms = round((perf_counter() - started_at) * 1000, 1)
    LOGGER.info(
        "valuation_backfill_done rows=%s only_missing=%s include_explanations=%s elapsed_ms=%s",
        len(valuation_rows),
        only_missing,
        include_explanations,
        elapsed_ms,
    )
    return len(valuation_rows)


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
        Listing.building_type.label("building_type"),
        Listing.condition.label("condition"),
        Listing.year_built.label("year_built"),
        Listing.seller_type.label("seller_type"),
        Listing.listing_price.label("listing_price"),
        Listing.listing_currency.label("listing_currency"),
        Listing.source_url.label("source_url"),
        Valuation.predicted_price.label("predicted_price"),
        Valuation.score.label("score"),
        Valuation.explanation_summary.label("explanation_summary"),
        Valuation.top_factors.label("top_factors"),
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
    output_currency: Literal["USD", "RUB"] = Query(default="RUB"),
    fx_rate: float | None = Query(default=None, gt=0),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> OpportunityListResponse:
    started_at = perf_counter()
    backfilled_rows = ensure_listing_valuations(session, only_missing=True, include_explanations=False)
    statement = _build_opportunity_base_query(current_user.id)
    statement = _apply_sorting(statement, sort_by).limit(limit)
    rows = session.execute(statement).mappings().all()
    normalized_output_currency = _normalize_currency(output_currency, default=DEFAULT_LISTING_CURRENCY)
    fx_rate_used = _resolve_rows_fx_rate(rows, normalized_output_currency, fx_rate)
    response = OpportunityListResponse(
        items=[
            _serialize_opportunity(
                row,
                comparison_currency=normalized_output_currency,
                fx_rate_used=fx_rate_used,
            )
            for row in rows
        ]
    )
    LOGGER.info(
        "opportunities_list_ready user_id=%s sort_by=%s limit=%s rows=%s backfilled_rows=%s elapsed_ms=%s",
        current_user.id,
        sort_by,
        limit,
        len(response.items),
        backfilled_rows,
        round((perf_counter() - started_at) * 1000, 1),
    )
    return response


@app.get("/shortlist", response_model=OpportunityListResponse)
def get_shortlist(
    output_currency: Literal["USD", "RUB"] = Query(default="RUB"),
    fx_rate: float | None = Query(default=None, gt=0),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> OpportunityListResponse:
    started_at = perf_counter()
    backfilled_rows = ensure_listing_valuations(session, only_missing=True, include_explanations=False)
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
    normalized_output_currency = _normalize_currency(output_currency, default=DEFAULT_LISTING_CURRENCY)
    fx_rate_used = _resolve_rows_fx_rate(rows, normalized_output_currency, fx_rate)
    response = OpportunityListResponse(
        items=[
            _serialize_opportunity(
                row,
                comparison_currency=normalized_output_currency,
                fx_rate_used=fx_rate_used,
            )
            for row in rows
        ]
    )
    LOGGER.info(
        "shortlist_ready user_id=%s rows=%s backfilled_rows=%s elapsed_ms=%s",
        current_user.id,
        len(response.items),
        backfilled_rows,
        round((perf_counter() - started_at) * 1000, 1),
    )
    return response


@app.post("/shortlist", response_model=ShortlistMutationResponse)
def save_shortlist_item(
    payload: SaveShortlistRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> ShortlistMutationResponse:
    ensure_listing_valuations(session, only_missing=True, include_explanations=False)
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
        "base_currency": MODEL_BASE_CURRENCY,
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
