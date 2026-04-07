from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "price_usd"
LEAKAGE_COLUMNS = ["price_per_m2_usd"]
IRRELEVANT_COLUMNS = [
    "listing_id",
    "url",
    "city",
    "address",
    "description",
    "parsed_at",
    "photos_downloaded",
]
TEXT_LIKE_COLUMNS = ["amenities", "documents", "security"]
REFERENCE_YEAR = datetime.now().year


@dataclass
class FeatureConfig:
    target_column: str
    numerical_features: list[str]
    categorical_features: list[str]
    derived_numeric_features: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    log_target: bool = True

    @property
    def feature_columns(self) -> list[str]:
        return self.numerical_features + self.categorical_features


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    if "listing_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["listing_id"])

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN, "total_area_m2"], how="any")
    cleaned = cleaned[cleaned[TARGET_COLUMN] > 10000]
    cleaned = cleaned[cleaned["total_area_m2"] > 10]

    if "rooms" in cleaned.columns:
        cleaned = cleaned[(cleaned["rooms"].isna()) | (cleaned["rooms"] > 0)]
        cleaned = cleaned[(cleaned["rooms"].isna()) | (cleaned["rooms"] <= 10)]

    if "year_built" in cleaned.columns:
        cleaned.loc[
            (cleaned["year_built"] < 1900) | (cleaned["year_built"] > REFERENCE_YEAR + 1),
            "year_built",
        ] = np.nan

    if {"latitude", "longitude"}.issubset(cleaned.columns):
        invalid_geo = ~(
            cleaned["latitude"].between(41.0, 44.0, inclusive="both")
            & cleaned["longitude"].between(73.0, 76.0, inclusive="both")
        )
        cleaned.loc[invalid_geo, ["latitude", "longitude"]] = np.nan

    return cleaned


def build_feature_config(df: pd.DataFrame) -> FeatureConfig:
    excluded_columns = (
        [TARGET_COLUMN]
        + LEAKAGE_COLUMNS
        + IRRELEVANT_COLUMNS
        + TEXT_LIKE_COLUMNS
    )

    candidate_columns = [column for column in df.columns if column not in excluded_columns]
    numeric_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_columns = [
        column
        for column in candidate_columns
        if column not in numeric_columns
    ]

    derived_numeric_features = [
        "area_per_room",
        "floor_ratio",
        "building_age",
        "is_top_floor",
        "is_first_floor",
        "has_coordinates",
    ]

    return FeatureConfig(
        target_column=TARGET_COLUMN,
        numerical_features=numeric_columns + derived_numeric_features,
        categorical_features=categorical_columns,
        derived_numeric_features=derived_numeric_features,
        excluded_columns=excluded_columns,
        log_target=True,
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def create_model_features(df: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    features = df.copy()

    for column in feature_config.feature_columns:
        if column not in features.columns and column not in feature_config.derived_numeric_features:
            features[column] = np.nan

    if {"total_area_m2", "rooms"}.issubset(features.columns):
        features["area_per_room"] = _safe_divide(features["total_area_m2"], features["rooms"])
    else:
        features["area_per_room"] = np.nan

    if {"floor", "total_floors"}.issubset(features.columns):
        features["floor_ratio"] = _safe_divide(features["floor"], features["total_floors"])
        features["is_top_floor"] = (
            (features["floor"] == features["total_floors"]).astype(float)
        )
        features["is_first_floor"] = (features["floor"] == 1).astype(float)
    else:
        features["floor_ratio"] = np.nan
        features["is_top_floor"] = np.nan
        features["is_first_floor"] = np.nan

    if "year_built" in features.columns:
        features["building_age"] = REFERENCE_YEAR - features["year_built"]
        features.loc[features["building_age"] < 0, "building_age"] = np.nan
    else:
        features["building_age"] = np.nan

    if {"latitude", "longitude"}.issubset(features.columns):
        features["has_coordinates"] = (
            features["latitude"].notna() & features["longitude"].notna()
        ).astype(float)
    else:
        features["has_coordinates"] = 0.0

    model_frame = features.loc[:, feature_config.feature_columns].copy()

    for column in feature_config.categorical_features:
        model_frame[column] = model_frame[column].astype("object")

    return model_frame


def prepare_training_frame(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, FeatureConfig, dict[str, Any]]:
    cleaned_df = clean_dataset(raw_df)
    feature_config = build_feature_config(cleaned_df)
    feature_frame = create_model_features(cleaned_df, feature_config)
    target = cleaned_df[feature_config.target_column].astype(float)

    qc_summary = {
        "rows_before_cleaning": int(len(raw_df)),
        "rows_after_cleaning": int(len(cleaned_df)),
        "rows_removed": int(len(raw_df) - len(cleaned_df)),
        "target_skew_raw": float(target.skew()),
        "target_skew_log1p": float(np.log1p(target).skew()),
        "selected_numerical_features": feature_config.numerical_features,
        "selected_categorical_features": feature_config.categorical_features,
    }

    return feature_frame, target, feature_config, qc_summary


def prepare_inference_frame(
    objects: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    if isinstance(objects, dict):
        frame = pd.DataFrame([objects])
    elif isinstance(objects, list):
        frame = pd.DataFrame(objects)
    else:
        frame = objects.copy()

    return create_model_features(frame, feature_config)
