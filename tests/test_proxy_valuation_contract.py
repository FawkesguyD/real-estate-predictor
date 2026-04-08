from __future__ import annotations

import unittest

import numpy as np

from apps.api.api import _build_scoring_payload, _serialize_opportunity
from ml.model.inference import LoadedModelBundle, predict_proxy_valuation_from_bundle
from ml.model.preprocessing import FeatureConfig


class _StubModel:
    def predict(self, frame):
        area = frame["total_area_m2"].fillna(0).astype(float)
        rooms = frame["rooms"].fillna(0).astype(float)
        return np.asarray((area * 1000.0) + (rooms * 5000.0), dtype=float)


def _make_bundle() -> LoadedModelBundle:
    feature_config = FeatureConfig(
        target_column="price_usd",
        numerical_features=[
            "rooms",
            "total_area_m2",
            "floor",
            "total_floors",
            "latitude",
            "longitude",
            "photo_count",
        ],
        categorical_features=[
            "district",
            "building_type",
            "seller_type",
        ],
        derived_numeric_features=[],
        excluded_columns=[],
        log_target=False,
    )
    return LoadedModelBundle(
        model_name="linear_stub",
        model=_StubModel(),
        feature_config=feature_config,
        metrics={},
        target_column="price_usd",
        log_target=False,
    )


class ProxyValuationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = _make_bundle()
        self.object_features = {
            "listing_id": 7,
            "listing_price": 9_000_000.0,
            "listing_currency": "RUB",
            "district": "Center",
            "rooms": 3,
            "floor": 4,
            "total_floors": 12,
            "total_area_m2": 110.0,
            "building_type": "brick",
            "seller_type": "owner",
            "latitude": 55.75,
            "longitude": 37.61,
            "photo_count": 12,
        }

    def test_rub_output_keeps_raw_listing_price_and_converts_model_metrics_only(self) -> None:
        result = predict_proxy_valuation_from_bundle(
            object_features=self.object_features,
            bundle=self.bundle,
            output_currency="BOTH",
            fx_rate=100.0,
            include_explanation=False,
        )

        usd_output = result["price_outputs"]["USD"]
        rub_output = result["price_outputs"]["RUB"]

        self.assertEqual(result["listing_price"], 9_000_000.0)
        self.assertEqual(result["listing_currency"], "RUB")
        self.assertEqual(rub_output["listing_price_in_comparison_currency"], 9_000_000.0)
        self.assertEqual(usd_output["listing_price_in_comparison_currency"], 90_000.0)
        self.assertAlmostEqual(rub_output["expected_price_proxy"], usd_output["expected_price_proxy"] * 100.0)
        self.assertAlmostEqual(rub_output["delta_abs"], usd_output["delta_abs"] * 100.0)
        self.assertAlmostEqual(rub_output["delta_pct"], usd_output["delta_pct"])

    def test_usd_output_uses_single_conversion_without_touching_raw_listing_price(self) -> None:
        result = predict_proxy_valuation_from_bundle(
            object_features=self.object_features,
            bundle=self.bundle,
            output_currency="USD",
            fx_rate=100.0,
            include_explanation=False,
        )

        usd_output = result["price_outputs"]["USD"]

        self.assertEqual(result["listing_price"], 9_000_000.0)
        self.assertEqual(result["listing_currency"], "RUB")
        self.assertEqual(usd_output["listing_price_in_comparison_currency"], 90_000.0)
        self.assertEqual(usd_output["comparison_currency"], "USD")
        self.assertEqual(usd_output["predicted_price_currency"], "USD")

    def test_listing_payload_mapping_uses_available_informative_fields(self) -> None:
        row = {
            "listing_id": 11,
            "listing_price": 12_500_000.0,
            "listing_currency": "RUB",
            "district": "Airport",
            "rooms": 2,
            "floor": 8,
            "total_floors": 16,
            "area": 78.5,
            "building_type": "monolith",
            "seller_type": "agent",
            "latitude": 42.85,
            "longitude": 74.61,
            "photo_count": 9,
            "source_url": "https://example.test/listing/11",
        }

        payload = _build_scoring_payload(row, self.bundle)

        self.assertEqual(payload["listing_price"], 12_500_000.0)
        self.assertEqual(payload["listing_currency"], "RUB")
        self.assertEqual(payload["total_area_m2"], 78.5)
        self.assertEqual(payload["building_type"], "monolith")
        self.assertEqual(payload["seller_type"], "agent")
        self.assertEqual(payload["latitude"], 42.85)
        self.assertEqual(payload["longitude"], 74.61)
        self.assertEqual(payload["photo_count"], 9)
        self.assertNotIn("price_usd", payload)

    def test_flat_opportunity_dto_exposes_source_url_and_explanation_fallback(self) -> None:
        item = _serialize_opportunity(
            {
                "listing_id": 21,
                "title": "Prime asset",
                "city": "Moscow",
                "district": "Center",
                "area": 95.0,
                "rooms": 3,
                "floor": 7,
                "total_floors": 12,
                "building_type": "brick",
                "condition": "renovated",
                "year_built": 2021,
                "seller_type": "owner",
                "listing_price": 9_000_000.0,
                "listing_currency": "RUB",
                "predicted_price": 100_000.0,
                "score": 0.91,
                "top_factors": ["Area helped", "District helped"],
                "explanation_summary": None,
                "source_url": "https://example.test/listing/21",
                "is_saved": True,
                "rank_position": 1,
            },
            comparison_currency="RUB",
            fx_rate_used=90.0,
        )

        self.assertEqual(item.listing_currency, "RUB")
        self.assertEqual(item.predicted_price_currency, "RUB")
        self.assertEqual(item.comparison_currency, "RUB")
        self.assertEqual(item.source_url, "https://example.test/listing/21")
        self.assertEqual(item.top_factors, ["Area helped", "District helped"])
        self.assertIn("proxy valuation", item.explanation_summary.lower())


if __name__ == "__main__":
    unittest.main()
