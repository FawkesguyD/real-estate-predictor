from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from ml.model.preprocessing import FeatureConfig, prepare_inference_frame


DEFAULT_BASE_CURRENCY = "USD"
DEFAULT_OUTPUT_CURRENCY = "USD"
DEFAULT_LISTING_CURRENCY = "USD"
SUPPORTED_OUTPUT_CURRENCIES = {"USD", "RUB", "BOTH"}
SUPPORTED_PRICE_CURRENCIES = {"USD", "RUB"}
DEFAULT_FX_RATE = 90.0
VALUATION_NOTE = (
    "MVP proxy valuation based on listing prices; not a transaction-based fair market valuation."
)


@dataclass
class LoadedModelBundle:
    model_name: str
    model: Any
    feature_config: FeatureConfig
    metrics: dict[str, Any]
    target_column: str
    log_target: bool


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _normalize_price_currency(
    currency: str | None,
    default: str = DEFAULT_LISTING_CURRENCY,
) -> Literal["USD", "RUB"]:
    if currency is None or pd.isna(currency):
        normalized = default
    else:
        normalized = str(currency)
    normalized = normalized.upper()
    if normalized not in SUPPORTED_PRICE_CURRENCIES:
        supported = ", ".join(sorted(SUPPORTED_PRICE_CURRENCIES))
        raise ValueError(f"Unsupported currency '{currency}'. Expected one of: {supported}.")
    return normalized  # type: ignore[return-value]


def _normalize_output_currency(output_currency: str | None) -> Literal["USD", "RUB", "BOTH"]:
    normalized = (output_currency or DEFAULT_OUTPUT_CURRENCY).upper()
    if normalized not in SUPPORTED_OUTPUT_CURRENCIES:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_CURRENCIES))
        raise ValueError(f"Unsupported output_currency '{output_currency}'. Expected one of: {supported}.")
    return normalized  # type: ignore[return-value]


def _resolve_fx_rate(
    conversion_required: bool,
    fx_rate: float | None,
    default_fx_rate: float,
) -> float | None:
    if not conversion_required:
        return None

    resolved_rate = default_fx_rate if fx_rate is None else fx_rate
    if resolved_rate is None or resolved_rate <= 0:
        raise ValueError("fx_rate must be positive when USD/RUB conversion is required.")
    return float(resolved_rate)


def _convert_amount(
    amount: float | None,
    source_currency: str,
    target_currency: str,
    fx_rate_used: float | None,
) -> float | None:
    if amount is None:
        return None

    normalized_source = _normalize_price_currency(source_currency)
    normalized_target = _normalize_price_currency(target_currency, default=DEFAULT_BASE_CURRENCY)
    if normalized_source == normalized_target:
        return float(amount)

    if fx_rate_used is None:
        raise ValueError("fx_rate is required for USD/RUB conversion.")

    if normalized_source == "USD" and normalized_target == "RUB":
        return float(amount) * fx_rate_used
    if normalized_source == "RUB" and normalized_target == "USD":
        return float(amount) / fx_rate_used

    raise ValueError(f"Unsupported conversion from {normalized_source} to {normalized_target}.")


def _normalize_object_features(
    object_features: dict[str, Any],
    bundle: LoadedModelBundle,
) -> dict[str, Any]:
    return dict(object_features)


def _normalize_objects_collection(
    objects: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    bundle: LoadedModelBundle,
) -> pd.DataFrame:
    return (
        pd.DataFrame([objects])
        if isinstance(objects, dict)
        else pd.DataFrame(objects)
        if isinstance(objects, list)
        else objects.copy()
    )


def _needs_fx_conversion_for_object(
    object_features: dict[str, Any],
    output_currency: Literal["USD", "RUB", "BOTH"],
) -> bool:
    listing_price = _coerce_optional_float(object_features.get("listing_price"))
    listing_currency = _normalize_price_currency(object_features.get("listing_currency"))
    return output_currency != DEFAULT_BASE_CURRENCY or (
        listing_price is not None and listing_currency != DEFAULT_BASE_CURRENCY
    )


def _needs_fx_conversion_for_collection(
    raw_frame: pd.DataFrame,
    output_currency: Literal["USD", "RUB", "BOTH"],
) -> bool:
    if output_currency != DEFAULT_BASE_CURRENCY:
        return True

    if "listing_price" not in raw_frame.columns:
        return False

    if "listing_currency" not in raw_frame.columns:
        return False

    listing_price = pd.to_numeric(raw_frame["listing_price"], errors="coerce")
    listing_currencies = raw_frame["listing_currency"].map(
        lambda value: _normalize_price_currency(value)
    )
    return bool((listing_price.notna() & (listing_currencies != DEFAULT_BASE_CURRENCY)).any())


def _prepare_object_features_for_valuation(
    object_features: dict[str, Any],
    bundle: LoadedModelBundle,
    fx_rate_used: float | None,
) -> dict[str, Any]:
    normalized = dict(object_features)
    normalized["listing_currency"] = _normalize_price_currency(normalized.get("listing_currency"))

    has_base_listing_price = bundle.target_column in normalized and not pd.isna(normalized[bundle.target_column])
    if not has_base_listing_price:
        listing_price = _coerce_optional_float(normalized.get("listing_price"))
        if listing_price is not None:
            normalized[bundle.target_column] = _convert_amount(
                listing_price,
                normalized["listing_currency"],
                DEFAULT_BASE_CURRENCY,
                fx_rate_used,
            )

    return normalized


def _prepare_objects_collection_for_valuation(
    objects: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    bundle: LoadedModelBundle,
    fx_rate_used: float | None,
) -> pd.DataFrame:
    raw_frame = _normalize_objects_collection(objects, bundle)
    prepared = raw_frame.copy()

    if "listing_currency" in prepared.columns:
        prepared["listing_currency"] = prepared["listing_currency"].map(
            lambda value: _normalize_price_currency(value)
        )
    else:
        prepared["listing_currency"] = DEFAULT_LISTING_CURRENCY

    if bundle.target_column not in prepared.columns:
        prepared[bundle.target_column] = np.nan

    if "listing_price" not in prepared.columns:
        return prepared

    missing_base_price = prepared[bundle.target_column].isna() & prepared["listing_price"].notna()
    if not missing_base_price.any():
        return prepared

    prepared.loc[missing_base_price, bundle.target_column] = prepared.loc[missing_base_price].apply(
        lambda row: _convert_amount(
            _coerce_optional_float(row.get("listing_price")),
            row.get("listing_currency") or DEFAULT_LISTING_CURRENCY,
            DEFAULT_BASE_CURRENCY,
            fx_rate_used,
        ),
        axis=1,
    )

    return prepared


def _build_currency_block(
    expected_price_proxy_usd: float,
    listing_price: float | None,
    listing_currency: str,
    currency: Literal["USD", "RUB"],
    fx_rate_used: float | None,
) -> dict[str, float | None]:
    expected_price_proxy = _convert_amount(
        expected_price_proxy_usd,
        DEFAULT_BASE_CURRENCY,
        currency,
        fx_rate_used,
    )
    listing_price_in_comparison_currency = _convert_amount(
        listing_price,
        listing_currency,
        currency,
        fx_rate_used,
    )
    delta_abs = None
    delta_pct = None
    if expected_price_proxy is not None and listing_price_in_comparison_currency is not None:
        delta_abs = expected_price_proxy - listing_price_in_comparison_currency
        if listing_price_in_comparison_currency != 0:
            delta_pct = delta_abs / listing_price_in_comparison_currency

    return {
        "expected_price_proxy": expected_price_proxy,
        "comparison_currency": currency,
        "predicted_price_currency": currency,
        "listing_price_in_comparison_currency": listing_price_in_comparison_currency,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
    }


def _build_price_outputs(
    expected_price_proxy_usd: float,
    listing_price: float | None,
    listing_currency: str,
    output_currency: Literal["USD", "RUB", "BOTH"],
    fx_rate_used: float | None,
) -> dict[str, dict[str, float | None]]:
    price_outputs: dict[str, dict[str, float | None]] = {}

    if output_currency in {"USD", "BOTH"}:
        price_outputs["USD"] = _build_currency_block(
            expected_price_proxy_usd=expected_price_proxy_usd,
            listing_price=listing_price,
            listing_currency=listing_currency,
            currency="USD",
            fx_rate_used=fx_rate_used,
        )

    if output_currency in {"RUB", "BOTH"}:
        price_outputs["RUB"] = _build_currency_block(
            expected_price_proxy_usd=expected_price_proxy_usd,
            listing_price=listing_price,
            listing_currency=listing_currency,
            currency="RUB",
            fx_rate_used=fx_rate_used,
        )

    return price_outputs


def _feature_group(feature_name: str) -> str:
    feature_groups = {
        "total_area_m2": "area",
        "living_area_m2": "area",
        "kitchen_area_m2": "area",
        "area_per_room": "area_layout",
        "rooms": "rooms",
        "district": "district",
        "latitude": "geo",
        "longitude": "geo",
        "has_coordinates": "geo",
        "building_type": "building",
        "building_series": "building",
        "condition": "condition",
        "year_built": "age",
        "building_age": "age",
        "floor": "floor",
        "total_floors": "floor",
        "floor_ratio": "floor",
        "is_top_floor": "floor",
        "is_first_floor": "floor",
    }
    return feature_groups.get(feature_name, feature_name)


def _format_feature_topic(feature_name: str, raw_value: Any) -> str:
    default_topics = {
        "total_area_m2": "площадь объекта",
        "living_area_m2": "жилая площадь",
        "kitchen_area_m2": "площадь кухни",
        "rooms": "комнатность",
        "floor": "этаж",
        "total_floors": "этажность дома",
        "ceiling_height": "высота потолков",
        "year_built": "год постройки",
        "building_age": "возраст здания",
        "district": "район",
        "building_type": "тип дома",
        "building_series": "серия дома",
        "condition": "состояние объекта",
        "heating": "отопление",
        "gas_supply": "газоснабжение",
        "bathroom": "санузел",
        "balcony": "балкон",
        "parking": "парковка",
        "furniture": "меблировка",
        "flooring": "покрытие пола",
        "door_type": "тип двери",
        "has_landline_phone": "стационарный телефон",
        "internet": "интернет",
        "mortgage": "ипотека",
        "seller_type": "тип продавца",
        "floor_ratio": "позиция по этажу",
        "is_top_floor": "расположение на последнем этаже",
        "is_first_floor": "расположение на первом этаже",
        "area_per_room": "соотношение площади к комнатам",
        "latitude": "широта объекта",
        "longitude": "долгота объекта",
        "has_coordinates": "наличие координат объекта",
        "photo_count": "количество фото",
    }

    if raw_value is None or pd.isna(raw_value):
        return default_topics.get(feature_name, feature_name.replace("_", " "))

    topic_builders: dict[str, Any] = {
        "total_area_m2": lambda value: f"площадь {float(value):.1f} м²",
        "living_area_m2": lambda value: f"жилая площадь {float(value):.1f} м²",
        "kitchen_area_m2": lambda value: f"площадь кухни {float(value):.1f} м²",
        "rooms": lambda value: f"комнатность {int(float(value))}",
        "floor": lambda value: f"этаж {int(float(value))}",
        "total_floors": lambda value: f"этажность дома {int(float(value))}",
        "ceiling_height": lambda value: f"высота потолков {float(value):.2f} м",
        "year_built": lambda value: f"год постройки {int(float(value))}",
        "building_age": lambda value: f"возраст здания около {int(float(value))} лет",
        "district": lambda value: f"район {value}",
        "building_type": lambda value: f"тип дома {value}",
        "building_series": lambda value: f"серия дома {value}",
        "condition": lambda value: f"состояние {value}",
        "heating": lambda value: f"отопление {value}",
        "gas_supply": lambda value: f"газоснабжение {value}",
        "bathroom": lambda value: f"санузел {value}",
        "balcony": lambda value: f"балкон {value}",
        "parking": lambda value: f"парковка {value}",
        "furniture": lambda value: f"меблировка {value}",
        "flooring": lambda value: f"покрытие пола {value}",
        "door_type": lambda value: f"тип двери {value}",
        "has_landline_phone": lambda value: f"стационарный телефон {value}",
        "internet": lambda value: f"интернет {value}",
        "mortgage": lambda value: f"ипотека {value}",
        "seller_type": lambda value: f"тип продавца {value}",
        "floor_ratio": lambda value: "позиция по этажу",
        "is_top_floor": lambda value: "расположение на последнем этаже",
        "is_first_floor": lambda value: "расположение на первом этаже",
        "area_per_room": lambda value: "соотношение площади к комнатам",
        "latitude": lambda value: "широта объекта",
        "longitude": lambda value: "долгота объекта",
        "has_coordinates": lambda value: "наличие координат объекта",
        "photo_count": lambda value: f"количество фото {int(float(value))}",
    }

    builder = topic_builders.get(feature_name)
    if builder is None:
        return default_topics.get(feature_name, feature_name.replace("_", " "))

    try:
        return builder(raw_value)
    except (TypeError, ValueError):
        return default_topics.get(feature_name, feature_name.replace("_", " "))


def _build_explanation_summary(
    factor_details: list[dict[str, Any]],
) -> str:
    positive_topics = [item["topic"] for item in factor_details if item["direction"] == "positive"]
    negative_topics = [item["topic"] for item in factor_details if item["direction"] == "negative"]

    if positive_topics and negative_topics:
        return (
            f"Proxy-оценка в основном поддержана факторами: {', '.join(positive_topics[:2])}. "
            f"При этом признаки вроде {negative_topics[0]} частично снизили оценку. "
            "Это модельная оценка по данным объявлений, а не transaction-based fair market valuation."
        )
    if positive_topics:
        return (
            f"Proxy-оценка в основном поддержана факторами: {', '.join(positive_topics[:3])}. "
            "Это модельная оценка по данным объявлений, а не transaction-based fair market valuation."
        )
    if negative_topics:
        return (
            f"На proxy-оценку сильнее всего повлияли признаки вроде {', '.join(negative_topics[:3])}. "
            "Это модельная оценка по данным объявлений, а не transaction-based fair market valuation."
        )
    return (
        "Proxy-оценка построена по характеристикам объявления и отражает паттерны listing data, "
        "а не transaction-based fair market valuation."
    )


def _fallback_explanation(object_features: dict[str, Any]) -> tuple[list[str], str]:
    factors: list[dict[str, Any]] = []

    total_area = _coerce_optional_float(object_features.get("total_area_m2"))
    if total_area is not None:
        direction = "positive" if total_area >= 80 else "negative" if total_area <= 45 else "positive"
        topic = f"площадь {total_area:.1f} м²"
        factors.append({"topic": topic, "direction": direction})

    rooms = _coerce_optional_float(object_features.get("rooms"))
    if rooms is not None:
        direction = "positive" if rooms >= 3 else "negative" if rooms <= 1 else "positive"
        topic = f"комнатность {int(rooms)}"
        factors.append({"topic": topic, "direction": direction})

    district = object_features.get("district")
    if district:
        factors.append({"topic": f"район {district}", "direction": "positive"})

    condition = object_features.get("condition")
    if isinstance(condition, str) and condition.strip():
        lowered = condition.lower()
        direction = "positive"
        if any(token in lowered for token in ["требует", "плох", "стар", "без ремонта"]):
            direction = "negative"
        factors.append({"topic": f"состояние {condition}", "direction": direction})

    year_built = _coerce_optional_float(object_features.get("year_built"))
    if year_built is not None:
        direction = "positive" if year_built >= 2010 else "negative" if year_built <= 1985 else "positive"
        factors.append({"topic": f"год постройки {int(year_built)}", "direction": direction})

    top_factors = [
        f"Фактор «{item['topic']}» {'повысил' if item['direction'] == 'positive' else 'снизил'} proxy-оценку"
        for item in factors[:5]
    ]
    return top_factors[:5], _build_explanation_summary(factors[:5])


def explain_prediction_from_bundle(
    object_features: dict[str, Any],
    bundle: LoadedModelBundle,
    max_factors: int = 5,
) -> dict[str, Any]:
    normalized_features = _normalize_object_features(object_features, bundle)
    inference_frame = prepare_inference_frame(normalized_features, bundle.feature_config)

    if bundle.model_name == "catboost_regressor":
        try:
            from catboost import Pool

            cat_frame = inference_frame.copy()
            for column in bundle.feature_config.categorical_features:
                cat_frame[column] = cat_frame[column].fillna("missing").astype(str)

            shap_values = bundle.model.get_feature_importance(
                type="ShapValues",
                data=Pool(cat_frame, cat_features=bundle.feature_config.categorical_features),
            )
            feature_names = list(cat_frame.columns)
            contributions = shap_values[0][:-1]

            ranked_indices = np.argsort(np.abs(contributions))[::-1]
            factor_details: list[dict[str, Any]] = []
            seen_groups: set[str] = set()

            for index in ranked_indices:
                contribution = float(contributions[index])
                if np.isclose(contribution, 0.0):
                    continue

                feature_name = feature_names[index]
                group_name = _feature_group(feature_name)
                if group_name in seen_groups:
                    continue

                raw_value = normalized_features.get(feature_name)
                topic = _format_feature_topic(feature_name, raw_value)
                direction = "positive" if contribution > 0 else "negative"
                factor_details.append(
                    {
                        "feature": feature_name,
                        "topic": topic,
                        "direction": direction,
                        "contribution": contribution,
                    }
                )
                seen_groups.add(group_name)

                if len(factor_details) >= max(3, max_factors):
                    break

            if factor_details:
                top_factors = [
                    f"Фактор «{item['topic']}» {'повысил' if item['direction'] == 'positive' else 'снизил'} proxy-оценку"
                    for item in factor_details[:max_factors]
                ]
                return {
                    "top_factors": top_factors,
                    "explanation_summary": _build_explanation_summary(factor_details[:max_factors]),
                }
        except Exception:
            pass

    top_factors, explanation_summary = _fallback_explanation(normalized_features)
    return {
        "top_factors": top_factors[:max_factors],
        "explanation_summary": explanation_summary,
    }


def load_model_bundle(model_path: str | Path) -> LoadedModelBundle:
    payload = joblib.load(model_path)
    feature_config = FeatureConfig(**payload["feature_config"])
    return LoadedModelBundle(
        model_name=payload["model_name"],
        model=payload["model"],
        feature_config=feature_config,
        metrics=payload["metrics"],
        target_column=payload["target_column"],
        log_target=payload["log_target"],
    )


def _predict_frame(bundle: LoadedModelBundle, frame: pd.DataFrame) -> np.ndarray:
    if bundle.model_name == "catboost_regressor":
        cat_frame = frame.copy()
        for column in bundle.feature_config.categorical_features:
            cat_frame[column] = cat_frame[column].fillna("missing").astype(str)
        predictions = bundle.model.predict(cat_frame)
    else:
        predictions = bundle.model.predict(frame)

    if bundle.log_target:
        predictions = np.expm1(predictions)

    return np.clip(np.asarray(predictions, dtype=float), a_min=0.0, a_max=None)


def _build_single_prediction_response(
    expected_price_proxy: float,
    listing_price: Any,
) -> dict[str, float | None]:
    listing_price_value = None
    delta_abs = None
    delta_pct = None

    if listing_price is not None and not pd.isna(listing_price):
        listing_price_value = float(listing_price)
        delta_abs = expected_price_proxy - listing_price_value
        if listing_price_value != 0:
            delta_pct = delta_abs / listing_price_value

    return {
        "expected_price_proxy": expected_price_proxy,
        "listing_price": listing_price_value,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
    }


def _attach_delta_columns(
    scored: pd.DataFrame,
    listing_price_column: str | None,
) -> pd.DataFrame:
    if listing_price_column and listing_price_column in scored.columns:
        scored["listing_price"] = pd.to_numeric(scored[listing_price_column], errors="coerce")
        scored["delta_abs"] = scored["expected_price_proxy"] - scored["listing_price"]
        scored["delta_pct"] = np.where(
            scored["listing_price"].notna() & (scored["listing_price"] != 0),
            scored["delta_abs"] / scored["listing_price"],
            np.nan,
        )
    else:
        scored["listing_price"] = np.nan
        scored["delta_abs"] = np.nan
        scored["delta_pct"] = np.nan

    return scored


def predict_expected_price_from_bundle(
    object_features: dict[str, Any],
    bundle: LoadedModelBundle,
) -> dict[str, float | None]:
    normalized_features = _normalize_object_features(object_features, bundle)
    inference_frame = prepare_inference_frame(normalized_features, bundle.feature_config)
    expected_price_proxy = float(_predict_frame(bundle, inference_frame)[0])
    listing_price = normalized_features.get(bundle.target_column)
    return _build_single_prediction_response(expected_price_proxy, listing_price)


def predict_expected_price(
    object_features: dict[str, Any],
    model_path: str | Path,
) -> dict[str, float | None]:
    bundle = load_model_bundle(model_path)
    return predict_expected_price_from_bundle(object_features, bundle)


def score_objects_from_bundle(
    objects: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    bundle: LoadedModelBundle,
    listing_price_column: str | None = "price_usd",
) -> pd.DataFrame:
    raw_frame = _normalize_objects_collection(objects, bundle)
    inference_frame = prepare_inference_frame(raw_frame, bundle.feature_config)
    predictions = _predict_frame(bundle, inference_frame)

    scored = raw_frame.copy()
    scored["expected_price_proxy"] = predictions
    return _attach_delta_columns(scored, listing_price_column)


def score_objects(
    objects: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    model_path: str | Path,
    listing_price_column: str | None = "price_usd",
) -> pd.DataFrame:
    bundle = load_model_bundle(model_path)
    return score_objects_from_bundle(objects, bundle, listing_price_column)


def rank_by_undervaluation(scored_frame: pd.DataFrame) -> pd.DataFrame:
    ranked = scored_frame.sort_values(
        by=["delta_pct", "delta_abs"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    ranked["undervaluation_rank"] = ranked.index + 1
    return ranked


def predict_proxy_valuation_from_bundle(
    object_features: dict[str, Any],
    bundle: LoadedModelBundle,
    output_currency: str = DEFAULT_OUTPUT_CURRENCY,
    fx_rate: float | None = None,
    default_fx_rate: float = DEFAULT_FX_RATE,
    include_explanation: bool = True,
) -> dict[str, Any]:
    normalized_output_currency = _normalize_output_currency(output_currency)
    fx_rate_used = _resolve_fx_rate(
        _needs_fx_conversion_for_object(object_features, normalized_output_currency),
        fx_rate,
        default_fx_rate,
    )
    prepared_features = _prepare_object_features_for_valuation(object_features, bundle, fx_rate_used)
    prediction = predict_expected_price_from_bundle(prepared_features, bundle)
    listing_price = _coerce_optional_float(prepared_features.get("listing_price"))
    listing_currency = _normalize_price_currency(prepared_features.get("listing_currency"))

    response: dict[str, Any] = {
        "base_currency": DEFAULT_BASE_CURRENCY,
        "output_currency": normalized_output_currency,
        "listing_price": listing_price,
        "listing_currency": listing_currency,
        "fx_rate_used": fx_rate_used,
        "price_outputs": _build_price_outputs(
            expected_price_proxy_usd=float(prediction["expected_price_proxy"]),
            listing_price=listing_price,
            listing_currency=listing_currency,
            output_currency=normalized_output_currency,
            fx_rate_used=fx_rate_used,
        ),
        "valuation_note": VALUATION_NOTE,
    }

    if include_explanation:
        response.update(explain_prediction_from_bundle(prepared_features, bundle))

    return response


def predict_proxy_valuation(
    object_features: dict[str, Any],
    model_path: str | Path,
    output_currency: str = DEFAULT_OUTPUT_CURRENCY,
    fx_rate: float | None = None,
    default_fx_rate: float = DEFAULT_FX_RATE,
    include_explanation: bool = True,
) -> dict[str, Any]:
    bundle = load_model_bundle(model_path)
    return predict_proxy_valuation_from_bundle(
        object_features=object_features,
        bundle=bundle,
        output_currency=output_currency,
        fx_rate=fx_rate,
        default_fx_rate=default_fx_rate,
        include_explanation=include_explanation,
    )


def score_proxy_valuations_from_bundle(
    objects: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    bundle: LoadedModelBundle,
    output_currency: str = DEFAULT_OUTPUT_CURRENCY,
    fx_rate: float | None = None,
    default_fx_rate: float = DEFAULT_FX_RATE,
    rank_results: bool = True,
    include_explanations: bool = False,
    listing_id_column: str = "listing_id",
) -> list[dict[str, Any]]:
    normalized_output_currency = _normalize_output_currency(output_currency)
    raw_frame = _normalize_objects_collection(objects, bundle)
    fx_rate_used = _resolve_fx_rate(
        _needs_fx_conversion_for_collection(raw_frame, normalized_output_currency),
        fx_rate,
        default_fx_rate,
    )

    prepared_frame = _prepare_objects_collection_for_valuation(raw_frame, bundle, fx_rate_used)
    scored = score_objects_from_bundle(prepared_frame, bundle, listing_price_column=bundle.target_column)
    if "input_index" not in scored.columns:
        scored["input_index"] = np.arange(len(scored))
    else:
        missing_index_mask = scored["input_index"].isna()
        scored.loc[missing_index_mask, "input_index"] = np.flatnonzero(missing_index_mask)

    if rank_results:
        scored = rank_by_undervaluation(scored)

    results: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        result: dict[str, Any] = {
            "input_index": row.get("input_index"),
            "listing_id": row.get(listing_id_column),
            "base_currency": DEFAULT_BASE_CURRENCY,
            "output_currency": normalized_output_currency,
            "listing_price": _coerce_optional_float(row.get("listing_price")),
            "listing_currency": _normalize_price_currency(row.get("listing_currency")),
            "fx_rate_used": fx_rate_used,
            "price_outputs": _build_price_outputs(
                expected_price_proxy_usd=float(row["expected_price_proxy"]),
                listing_price=_coerce_optional_float(row.get("listing_price")),
                listing_currency=row.get("listing_currency") or DEFAULT_LISTING_CURRENCY,
                output_currency=normalized_output_currency,
                fx_rate_used=fx_rate_used,
            ),
            "valuation_note": VALUATION_NOTE,
            "undervaluation_rank": row.get("undervaluation_rank"),
        }
        if include_explanations:
            explanation_payload = explain_prediction_from_bundle(row, bundle)
            result.update(explanation_payload)
        results.append(result)

    return results
