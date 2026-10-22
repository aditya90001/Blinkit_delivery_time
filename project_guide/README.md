# Blinkit Delivery-Time Prediction: Project Guide

## What the project does

This project estimates an order's delivery duration in minutes from information
available at checkout. It cleans the supplied Blinkit order dataset, explores
delivery patterns, trains regression models, and serves predictions through a
FastAPI application and HTML form.

The prediction target is `actual_delivery_mins`. Training is limited to valid,
completed orders because cancelled and partially fulfilled records do not have
meaningful delivery times.

## How a prediction is produced

1. `data/compressed_data (2).csv` is loaded for EDA and model training.
2. `app/features.py` parses quantities, normalizes category values, handles
   invalid price and basket values, removes duplicate order IDs for training,
   and creates calendar features. The model avoids identifiers and information
   unavailable at checkout.
3. `app/training.py` creates chronological training, validation, and holdout
   periods, compares regression models, and saves the fitted pipeline and
   evaluation reports.
4. `app/main.py` loads `models/delivery_time_model.joblib` at startup and
   exposes `/`, `/health`, `/predict`, and `/docs`.
5. `templates/index.html` is the browser form. It calls the API on the same
   host and displays the predicted minutes.
6. `Dockerfile` packages the API, frontend template, and trained model.
   `render.yaml` configures the Docker web service and `/health` check.

## Important files

| Path | Purpose |
|---|---|
| `data/compressed_data (2).csv` | Source training dataset; excluded from the Docker image |
| `notebooks/blinkit_delivery_time_eda.ipynb` | EDA, cleaning, visualizations, and model comparison |
| `app/features.py` | Shared feature preparation for training and prediction |
| `app/training.py` | Regressors, temporal splits, evaluation, and artifact saving |
| `train.py` | Command-line training entry point |
| `models/delivery_time_model.joblib` | Persisted model loaded by the API |
| `reports/model_comparison.csv` | Previously saved chronological holdout metrics |
| `reports/model_metadata.json` | Previously saved model and holdout metadata |
| `app/main.py` | FastAPI application and prediction endpoints |
| `templates/index.html` | Prediction user interface |
| `requirements-api.txt` | Runtime dependencies installed in the container |
| `Dockerfile` | Image build and Render-compatible server startup |
| `render.yaml` | Render service definition |

## What is and is not measured

The latest saved report in `reports/` records the earlier, untuned model run:

| Model | MAE (minutes) | RMSE (minutes) | R² |
|---|---:|---:|---:|
| Linear regression | 1.876 | 3.329 | 0.212 |
| Random forest | 2.260 | 3.552 | 0.103 |
| Decision tree | 2.348 | 3.901 | -0.082 |
| XGBoost | 2.350 | 3.958 | -0.114 |

These are not results from the newer hyperparameter-tuning experiment. That
run was stopped before writing updated reports, so no improved score has been
verified. Regression does not have a classification-style accuracy score:
MAE and RMSE are errors in minutes, and R² is a measure of explained variance.

## Run the project locally

From the project root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/`. To check model readiness, open
`http://127.0.0.1:8001/health`. Interactive API documentation is at
`http://127.0.0.1:8001/docs`.

Training requires the CSV and the full dependencies in `requirements.txt`:

```powershell
.\venv\Scripts\python.exe train.py
```

## Next

See [INTERVIEW_QUESTIONS.md](./INTERVIEW_QUESTIONS.md) for interview
preparation, and [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md) for deploying
the Docker service.
