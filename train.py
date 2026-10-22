"""Train and compare delivery-time regressors on the Blinkit CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.training import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/compressed_data (2).csv"),
        help="Path to the source Blinkit CSV (default: data/compressed_data (2).csv).",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--tuning-trials",
        type=int,
        default=2,
        help="Number of sampled hyperparameter settings per model.",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(
            f"Dataset not found: {args.data}. Place the Blinkit CSV in the data folder "
            "or pass its path with --data."
        )
    data = pd.read_csv(args.data)
    results, winner = train_and_save(
        data,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        test_fraction=args.test_fraction,
        tuning_trials=args.tuning_trials,
    )
    display_columns = [
        "model",
        "mae_minutes",
        "rmse_minutes",
        "r2",
        "tuning_validation_mae",
    ]
    print("\nTuned temporal holdout comparison (lower MAE is better):")
    print(
        results[display_columns].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    print(f"Tuned parameters: {args.report_dir / 'model_comparison.csv'}")
    print(f"\nSelected model: {winner}")
    print(f"Saved model: {args.model_dir / 'delivery_time_model.joblib'}")
    print(f"Saved metrics: {args.report_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
