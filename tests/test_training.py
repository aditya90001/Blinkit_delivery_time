import numpy as np
import pandas as pd

from app.training import _validation_split, train_and_save


def test_training_compares_models_and_saves_artifact(tmp_path):
    rows = 40
    data = pd.DataFrame(
        {
            "order_id": [f"order-{index}" for index in range(rows)],
            "order_date": pd.date_range("2025-01-01", periods=rows).astype(str),
            "order_hour": np.arange(rows) % 18 + 6,
            "city": np.where(np.arange(rows) % 2, "Delhi", "Mumbai"),
            "dark_store_id": "store-1",
            "category": "Snacks",
            "quantity": 1,
            "unit_price": 40,
            "order_value": 40,
            "distance_km": np.arange(rows) / 20 + 1,
            "order_status": "Delivered",
            "actual_delivery_mins": np.arange(rows) / 3 + 10,
            "payment_mode": "UPI",
        }
    )

    results, winner = train_and_save(
        data,
        model_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        tuning_trials=1,
    )

    assert set(results["model"]) == {
        "linear_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }
    assert winner == results.iloc[0]["model"]
    assert "tuning_validation_mae" in results
    assert "tuned_parameters" in results
    assert (tmp_path / "models" / "delivery_time_model.joblib").is_file()
    assert (tmp_path / "reports" / "model_comparison.csv").is_file()
    assert (tmp_path / "reports" / "model_metadata.json").is_file()


def test_validation_split_keeps_later_dates_out_of_tuning_fit():
    dates = pd.Series(pd.date_range("2025-01-01", periods=10))
    X = pd.DataFrame({"date_order": np.arange(10)})
    y = pd.Series(np.arange(10, dtype=float))

    X_fit, X_validation, _, _ = _validation_split(X, y, dates)

    assert X_fit["date_order"].max() < X_validation["date_order"].min()
