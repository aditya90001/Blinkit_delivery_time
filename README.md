# Blinkit Delivery Time Prediction

An end-to-end machine-learning project that estimates grocery delivery time
from checkout-time order details. It includes exploratory data analysis,
feature engineering, regression model comparison, a FastAPI service, and an
HTML prediction form.

> **Model status:** The saved evaluation scores below are from the earlier
> model comparison. The later hyperparameter-tuning run was stopped before it
> produced new verified metrics. Do not treat the old scores as tuned results.

## Contents

- [Project overview](#project-overview)
- [Dataset and target](#dataset-and-target)
- [Features and data preparation](#features-and-data-preparation)
- [Model evaluation](#model-evaluation)
- [Quick start](#quick-start)
- [API](#api)
- [EDA notebook](#eda-notebook)
- [Tests](#tests)
- [Deploy to Render](#deploy-to-render)
- [Project structure](#project-structure)
- [Learning resources](#learning-resources)

## Project overview

The API estimates `actual_delivery_mins` using information expected to be
available when an order is placed. The trained scikit-learn preprocessing
pipeline and model are saved together, then loaded by FastAPI at startup.
The browser interface is served by the same API.

## Dataset and target

The supplied CSV contains 52,750 rows and 18 columns. After removing duplicate
order IDs and retaining valid delivered orders, the training dataset contains
44,402 records. Cancelled and partially fulfilled orders are excluded because
their delivery-time values may be placeholders.

The target is **actual delivery time in minutes**. This is a regression task,
so its performance is measured with MAE, RMSE, and R²—not classification
accuracy.

## Features and data preparation

Checkout-time model inputs include:

- City, dark store, product category, and payment method
- Order hour and date-derived calendar features
- Quantity, unit price, order value, and delivery distance
- Missing-value indicators and cyclical time encodings

The shared feature pipeline parses quantity text, normalizes category labels,
handles invalid values, and prepares missing values. Training-only imputers
and encoders are fitted inside the model pipeline.

Order/customer identifiers, delivery partner assignment, order status,
customer rating, promised delivery time, and stockout status are excluded
because they can be identifiers, outcome-derived, unavailable at checkout, or
non-informative for delivered orders.

## Model evaluation

The previous saved report evaluates a chronological holdout beginning
2025-05-24. It contains 9,081 holdout orders.

| Model | MAE (minutes) | RMSE (minutes) | R² |
|---|---:|---:|---:|
| Linear regression | 1.876 | 3.329 | 0.212 |
| Random forest | 2.260 | 3.552 | 0.103 |
| Decision tree | 2.348 | 3.901 | -0.082 |
| XGBoost | 2.350 | 3.958 | -0.114 |

**Interpretation:** Linear regression had the lowest MAE in this saved run.
The R² of 0.212 means it explains about 21.2% of the target variance on this
holdout; it is not “21.2% prediction accuracy.” The errors are in minutes,
and predictions are estimates—not guaranteed ETAs.

Metrics are also saved in `reports/model_comparison.csv` and
`reports/model_metadata.json`. Reports and the raw dataset are excluded from
Git by default.

## Quick start

### Requirements

- Python 3.11 or later
- The provided dataset at `data/compressed_data (2).csv` for training

### Install dependencies

Run these commands in PowerShell from the project root:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you do not already have a virtual environment, create one first:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Train models

```powershell
.\venv\Scripts\python.exe train.py
```

To train from another CSV:

```powershell
.\venv\Scripts\python.exe train.py --data "C:\path\to\compressed_data.csv"
```

Training compares Linear/Ridge Regression, Decision Tree, Random Forest,
and XGBoost using time-ordered data. The tuning configuration can be adjusted
with `--tuning-trials`. A run writes:

- `models/delivery_time_model.joblib`
- `reports/model_comparison.csv`
- `reports/model_metadata.json`

The current saved metrics above are from the previous run; rerun training to
produce fresh metrics after any model or tuning changes.

### Start the web app

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open:

- Frontend: <http://127.0.0.1:8001/>
- API health: <http://127.0.0.1:8001/health>
- Interactive API docs: <http://127.0.0.1:8001/docs>

## API

`POST /predict` accepts JSON order details and returns the predicted duration
in minutes.

Example request:

```json
{
  "city": "Delhi",
  "category": "Snacks",
  "order_hour": 18,
  "quantity": 2,
  "unit_price": 40,
  "order_value": 80,
  "distance_km": 2.4,
  "payment_mode": "UPI"
}
```

Example response:

```json
{
  "predicted_delivery_mins": 16.25,
  "model": "linear_regression",
  "unit": "minutes"
}
```

Optional fields can be omitted; the fitted preprocessing pipeline handles
missing optional values. Required fields and validation are documented at
`/docs`.

## EDA notebook

Open [`notebooks/blinkit_delivery_time_eda.ipynb`](./notebooks/blinkit_delivery_time_eda.ipynb)
in VS Code or Jupyter. The notebook follows the supplied EDA's organization
and covers data quality, cleaning, delivery-time distributions, city/time
patterns, correlations, and model evaluation. Keep the CSV under `data/` to
run it from the project.

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest
node --test tests/test_frontend_script.js
```

## Deploy to Render

The project includes a Dockerfile and `render.yaml`. The image contains the
API, HTML template, runtime dependencies, and trained model artifact; it does
not need the raw CSV or run model training during deployment.

1. Push the project to a GitHub repository, including
   `models/delivery_time_model.joblib`.
2. Keep the raw dataset, secrets, virtual environments, and generated reports
   out of Git; these are covered by `.gitignore`.
3. In Render, create a Blueprint or Docker Web Service connected to the repo.
4. Confirm `/health` is the health-check path and deploy.
5. Open the Render service URL; the frontend is at `/`, and the API docs are
   at `/docs`.

See the [Render deployment guide](./project_guide/RENDER_DEPLOYMENT.md) for
detailed steps and troubleshooting.

## Project structure

```text
app/
  features.py       Shared cleaning and feature engineering
  training.py       Model comparison, tuning, evaluation, artifact saving
  main.py           FastAPI routes and model loading
data/               Local source dataset (ignored by Git)
models/             Trained model artifact required by the API
notebooks/          EDA and modeling notebook
project_guide/      Project overview, interview prep, Render instructions
reports/            Generated model metrics (ignored by Git)
templates/          HTML prediction interface
tests/              API, feature, training, and frontend tests
train.py            Model-training command
Dockerfile          Container build and server startup
render.yaml         Render service configuration
requirements.txt    Training and development dependencies
requirements-api.txt API runtime dependencies
```

## Learning resources

- [Project and file overview](./project_guide/README.md)
- [Interview questions and answers](./project_guide/INTERVIEW_QUESTIONS.md)
- [Render deployment guide](./project_guide/RENDER_DEPLOYMENT.md)
