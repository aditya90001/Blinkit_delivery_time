"""Shared cleaning and feature definitions for training and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "actual_delivery_mins"
ORDER_ID = "order_id"
ORDER_DATE = "order_date"
CATEGORICAL_FEATURES = [
    "city",
    "dark_store_id",
    "category",
    "payment_mode",
]
NUMERIC_FEATURES = [
    "order_hour",
    "order_hour_sin",
    "order_hour_cos",
    "order_day_of_week",
    "order_day_of_week_sin",
    "order_day_of_week_cos",
    "order_month",
    "order_month_sin",
    "order_month_cos",
    "order_day_of_year",
    "is_weekend",
    "order_hour_missing",
    "quantity_missing",
    "distance_missing",
    "quantity",
    "unit_price",
    "order_value",
    "distance_km",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_TRAINING_COLUMNS = {TARGET, "order_status"}


def _numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
        series = series.astype("string").str.replace(",", "", regex=False)
        series = series.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(series, errors="coerce")


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Clean raw order records and create features known at order placement."""
    frame = data.copy()
    if frame.empty:
        raise ValueError("The input dataset contains no rows.")

    for column in CATEGORICAL_FEATURES:
        if column not in frame:
            frame[column] = "Unknown"
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace("", pd.NA).fillna("Unknown")

    for column in ("order_hour", "quantity", "unit_price", "order_value", "distance_km"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = _numeric_series(frame[column]).astype(float)

    frame.loc[~frame["order_hour"].between(0, 23), "order_hour"] = np.nan
    frame["order_hour_missing"] = frame["order_hour"].isna().astype(float)
    frame["quantity_missing"] = frame["quantity"].isna().astype(float)
    frame["distance_missing"] = frame["distance_km"].isna().astype(float)

    if ORDER_DATE in frame:
        dates = pd.to_datetime(frame[ORDER_DATE], errors="coerce")
    else:
        dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    day_of_week = dates.dt.dayofweek.astype("float64")
    frame["order_day_of_week"] = day_of_week
    frame["order_day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    frame["order_day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    frame["order_month"] = dates.dt.month.astype("float64")
    frame["order_month_sin"] = np.sin(2 * np.pi * (frame["order_month"] - 1) / 12)
    frame["order_month_cos"] = np.cos(2 * np.pi * (frame["order_month"] - 1) / 12)
    frame["order_day_of_year"] = dates.dt.dayofyear.astype("float64")
    frame["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype("float64")
    frame["order_hour_sin"] = np.sin(2 * np.pi * frame["order_hour"] / 24)
    frame["order_hour_cos"] = np.cos(2 * np.pi * frame["order_hour"] / 24)

    frame["category"] = frame["category"].replace(
        {"Fruits and Vegetables": "Fruits & Vegetables"}
    )

    # The reference EDA replaces these data-entry errors with the median.
    # Mark them missing here so the model's training-only imputer supplies that median.
    frame.loc[frame["unit_price"] > 1500, "unit_price"] = np.nan
    plausible_quantity = (
        frame["quantity"].isna()
        & frame["unit_price"].between(0, 1500, inclusive="both")
        & frame["order_value"].between(0, 10000, inclusive="both")
    )
    inferred_quantity = frame["order_value"] / frame["unit_price"].replace(0, np.nan)
    frame.loc[plausible_quantity, "quantity"] = inferred_quantity.loc[
        plausible_quantity
    ].round()
    frame["quantity"] = frame["quantity"].fillna(0)

    # Recalculate order value as in the reference EDA, while excluding implausible
    # flat Rs. 25,000 entries from the model features.
    calculated_value = frame["quantity"] * frame["unit_price"]
    usable_calculation = calculated_value.notna() & calculated_value.between(0, 10000)
    frame.loc[usable_calculation, "order_value"] = calculated_value.loc[usable_calculation]
    frame.loc[frame["order_value"] > 10000, "order_value"] = np.nan
    frame.loc[frame["distance_km"] < 0, "distance_km"] = np.nan

    return frame[FEATURES]


def prepare_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return leak-free features, target, and parsed dates for valid deliveries."""
    missing = sorted(REQUIRED_TRAINING_COLUMNS.difference(data.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {', '.join(missing)}")

    frame = data.copy()
    status = frame["order_status"].astype("string").str.strip().str.casefold()
    target = _numeric_series(frame[TARGET])
    valid = (
        status.eq("delivered")
        & target.notna()
        & np.isfinite(target)
        & target.between(1, 240, inclusive="both")
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError(
            "No valid delivered orders with an actual_delivery_mins value between "
            "1 and 240 minutes were found."
        )

    if ORDER_ID in frame:
        frame = frame.drop_duplicates(subset=[ORDER_ID], keep="first")
    dates = (
        pd.to_datetime(frame[ORDER_DATE], errors="coerce")
        if ORDER_DATE in frame
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    )
    y = _numeric_series(frame[TARGET]).astype(float)
    X = prepare_features(frame)
    return X.reset_index(drop=True), y.reset_index(drop=True), dates.reset_index(drop=True)


def normalize_payload(payload: dict[str, object]) -> pd.DataFrame:
    """Normalize one validated API request to the training feature schema."""
    return prepare_features(pd.DataFrame([payload]))
