"""Reproducible time-ordered model comparison and artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from app.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    prepare_training_data,
)

MAX_TUNING_ROWS = 3_000


def _make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(steps=numeric_steps)
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _models() -> dict[str, Pipeline]:
    from xgboost import XGBRegressor

    return {
        "linear_regression": Pipeline(
            [
                ("preprocess", _make_preprocessor(scale_numeric=True)),
                ("model", Ridge(alpha=0.0, solver="lsqr")),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocess", _make_preprocessor()),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=14, min_samples_leaf=8, random_state=42
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _make_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=20,
                        min_samples_leaf=3,
                        max_features=0.8,
                        n_jobs=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocess", _make_preprocessor()),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        n_jobs=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _parameter_spaces() -> dict[str, dict[str, list[Any]]]:
    return {
        "linear_regression": {
            "model__alpha": [0.0, 0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "decision_tree": {
            "model__max_depth": [8, 12, 14, 18, None],
            "model__min_samples_leaf": [3, 5, 8, 12, 20],
            "model__max_features": [None, 0.8],
        },
        "random_forest": {
            "model__n_estimators": [60, 100, 140],
            "model__max_depth": [12, 16, 20, None],
            "model__min_samples_leaf": [2, 3, 5, 8],
            "model__max_features": [0.7, 0.8, 1.0],
        },
        "xgboost": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
            "model__subsample": [0.75, 0.85, 1.0],
            "model__colsample_bytree": [0.75, 0.85, 1.0],
            "model__reg_lambda": [1.0, 3.0, 8.0],
            "model__min_child_weight": [1, 3, 5],
        },
    }


def _validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out the latest part of the outer training set for tuning."""
    if dates.notna().all() and dates.nunique() > 1:
        ordered_dates = dates.sort_values(kind="stable")
        split_position = min(
            max(int(len(ordered_dates) * (1 - validation_fraction)), 1),
            len(X) - 1,
        )
        cutoff = ordered_dates.iloc[split_position]
        fit_mask = dates < cutoff
        validation_mask = dates >= cutoff
        if fit_mask.any() and validation_mask.any():
            return (
                X.loc[fit_mask].reset_index(drop=True),
                X.loc[validation_mask].reset_index(drop=True),
                y.loc[fit_mask].reset_index(drop=True),
                y.loc[validation_mask].reset_index(drop=True),
            )

    split_position = min(
        max(int(len(X) * (1 - validation_fraction)), 1),
        len(X) - 1,
    )
    return (
        X.iloc[:split_position].reset_index(drop=True),
        X.iloc[split_position:].reset_index(drop=True),
        y.iloc[:split_position].reset_index(drop=True),
        y.iloc[split_position:].reset_index(drop=True),
    )


def _temporal_split(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, str]:
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("test_fraction must be between 0.1 and 0.4.")
    if len(X) < 20:
        raise ValueError(f"At least 20 valid delivered orders are required; found {len(X)}.")

    if dates.notna().all() and dates.nunique() > 1:
        ordered_dates = dates.sort_values(kind="stable")
        split_position = min(max(int(len(ordered_dates) * (1 - test_fraction)), 1), len(X) - 1)
        cutoff = ordered_dates.iloc[split_position]
        train_mask = dates < cutoff
        test_mask = dates >= cutoff
        if train_mask.any() and test_mask.any():
            return (
                X.loc[train_mask].reset_index(drop=True),
                X.loc[test_mask].reset_index(drop=True),
                y.loc[train_mask].reset_index(drop=True),
                y.loc[test_mask].reset_index(drop=True),
                dates.loc[train_mask].reset_index(drop=True),
                dates.loc[test_mask].reset_index(drop=True),
                f"chronological holdout from {cutoff.date()}",
            )

    # If source dates are absent or unusable, retain the source-row order as the
    # only defensible temporal proxy rather than randomly mixing future rows in.
    split_position = min(max(int(len(X) * (1 - test_fraction)), 1), len(X) - 1)
    return (
        X.iloc[:split_position].reset_index(drop=True),
        X.iloc[split_position:].reset_index(drop=True),
        y.iloc[:split_position].reset_index(drop=True),
        y.iloc[split_position:].reset_index(drop=True),
        dates.iloc[:split_position].reset_index(drop=True),
        dates.iloc[split_position:].reset_index(drop=True),
        "source-order holdout (order_date was missing or had fewer than two distinct dates)",
    )


def train_and_save(
    data: pd.DataFrame,
    model_dir: str | Path = "models",
    report_dir: str | Path = "reports",
    test_fraction: float = 0.2,
    tuning_trials: int = 2,
) -> tuple[pd.DataFrame, str]:
    """Tune four regressors on a temporal validation set and evaluate on a holdout."""
    if tuning_trials < 1:
        raise ValueError("tuning_trials must be at least 1.")
    X, y, dates = prepare_training_data(data)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        dates_train,
        _dates_test,
        split_description,
    ) = _temporal_split(
        X, y, dates, test_fraction
    )
    X_fit, X_validation, y_fit, y_validation = _validation_split(
        X_train, y_train, dates_train
    )
    if len(X_fit) > MAX_TUNING_ROWS:
        tuning_indices = X_fit.sample(
            n=MAX_TUNING_ROWS, random_state=42
        ).index
        X_tuning = X_fit.loc[tuning_indices].reset_index(drop=True)
        y_tuning = y_fit.loc[tuning_indices].reset_index(drop=True)
    else:
        X_tuning = X_fit
        y_tuning = y_fit
    scores: list[dict[str, float | str]] = []
    tuned_parameters: dict[str, dict[str, Any]] = {}
    validation_scores: dict[str, float] = {}
    base_models = _models()
    parameter_spaces = _parameter_spaces()
    for name, pipeline in base_models.items():
        candidates = list(
            ParameterSampler(
                parameter_spaces[name],
                n_iter=min(tuning_trials, len(list(ParameterGrid(parameter_spaces[name])))),
                random_state=42,
            )
        )
        candidates.insert(0, {})
        best_validation_mae = float("inf")
        best_parameters: dict[str, Any] = {}
        for parameters in candidates:
            candidate = clone(pipeline).set_params(**parameters)
            candidate.fit(X_tuning, y_tuning)
            validation_predictions = np.asarray(
                candidate.predict(X_validation), dtype=float
            )
            validation_mae = float(
                mean_absolute_error(y_validation, validation_predictions)
            )
            if validation_mae < best_validation_mae:
                best_validation_mae = validation_mae
                best_parameters = dict(parameters)

        tuned_parameters[name] = best_parameters
        validation_scores[name] = best_validation_mae
        pipeline = clone(base_models[name]).set_params(**best_parameters)
        pipeline.fit(X_train, y_train)
        predictions = np.asarray(pipeline.predict(X_test), dtype=float)
        tuned_mae = float(mean_absolute_error(y_test, predictions))
        scores.append(
            {
                "model": name,
                "mae_minutes": tuned_mae,
                "rmse_minutes": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "r2": float(r2_score(y_test, predictions)),
                "tuning_validation_mae": best_validation_mae,
                "tuned_parameters": json.dumps(best_parameters, sort_keys=True),
            }
        )
    results = pd.DataFrame(scores).sort_values("mae_minutes").reset_index(drop=True)
    winner = str(results.iloc[0]["model"])
    best_pipeline = clone(base_models[winner]).set_params(
        **tuned_parameters[winner]
    )
    best_pipeline.fit(X, y)

    model_path = Path(model_dir)
    output_path = Path(report_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_pipeline,
            "model_name": winner,
            "features": FEATURES,
            "metrics": results.to_dict(orient="records"),
            "tuned_parameters": tuned_parameters,
            "tuning_validation_mae": validation_scores,
            "split": split_description,
            "training_rows": len(X),
            "target": "actual_delivery_mins",
            "target_unit": "minutes",
        },
        model_path / "delivery_time_model.joblib",
    )
    results.to_csv(output_path / "model_comparison.csv", index=False)
    (output_path / "model_metadata.json").write_text(
        json.dumps(
            {
                "best_model": winner,
                "split": split_description,
                "training_rows": len(X),
                "test_rows": len(X_test),
                "tuning_rows": len(X_tuning),
                "available_tuning_rows": len(X_fit),
                "validation_rows": len(X_validation),
                "tuning_trials_per_model": tuning_trials,
                "tuned_parameters": tuned_parameters,
                "tuning_validation_mae": validation_scores,
                "holdout_metrics": results.to_dict(orient="records"),
                "target": "actual_delivery_mins",
                "target_unit": "minutes",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results, winner
