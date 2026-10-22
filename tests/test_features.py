import pandas as pd
import pytest

from app.features import FEATURES, prepare_features, prepare_training_data


def test_features_clean_quantity_category_and_outlier():
    raw = pd.DataFrame(
        {
            "order_date": ["2025-02-15"],
            "order_hour": [18],
            "city": ["Delhi"],
            "dark_store_id": [None],
            "category": ["Fruits and Vegetables"],
            "quantity": ["2 units"],
            "unit_price": [15000],
            "order_value": [25000],
            "distance_km": [1.5],
            "payment_mode": ["UPI"],
            "actual_delivery_mins": [17],
            "order_status": ["Delivered"],
            "customer_rating": [7],
            "delivery_partner_id": ["DP123"],
        }
    )

    features = prepare_features(raw)

    assert list(features.columns) == FEATURES
    assert features.loc[0, "category"] == "Fruits & Vegetables"
    assert features.loc[0, "dark_store_id"] == "Unknown"
    assert pd.isna(features.loc[0, "unit_price"])
    assert pd.isna(features.loc[0, "order_value"])
    assert features.loc[0, "order_hour_missing"] == 0
    assert features.loc[0, "distance_missing"] == 0
    assert features.loc[0, "order_hour_sin"] == pytest.approx(-1)
    assert "customer_rating" not in features
    assert "delivery_partner_id" not in features
    assert "actual_delivery_mins" not in features


def test_training_filters_nondelivered_and_deduplicates_order_ids():
    raw = pd.DataFrame(
        {
            "order_id": ["a", "b", "a"],
            "order_date": ["2025-01-01", "2025-01-02", "2025-01-01"],
            "order_status": ["Delivered", "Cancelled", "Delivered"],
            "actual_delivery_mins": [14, 20, 99],
            "city": ["Delhi", "Mumbai", "Delhi"],
        }
    )

    X, y, dates = prepare_training_data(raw)

    assert len(X) == len(y) == len(dates) == 1
    assert y.iloc[0] == 14


def test_training_requires_target_and_status():
    with pytest.raises(ValueError, match="order_status"):
        prepare_training_data(pd.DataFrame({"actual_delivery_mins": [12]}))


def test_features_infers_missing_quantity_from_plausible_basket():
    features = prepare_features(
        pd.DataFrame(
            {
                "quantity": [None],
                "unit_price": [20],
                "order_value": [60],
            }
        )
    )

    assert features.loc[0, "quantity"] == 3
    assert features.loc[0, "order_value"] == 60
    assert features.loc[0, "quantity_missing"] == 1
