# Blinkit Delivery Time Prediction

<p align="center">
  <strong>Predict an estimated grocery delivery time from order details.</strong>
  <br />
  Explore the data, compare regression models, and try the FastAPI-powered web app.
</p>

<p align="center">
  <a href="https://blinkit-delivery-time.onrender.com/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20App-168A50?style=for-the-badge" alt="Open the live app" />
  </a>
  <a href="https://blinkit-delivery-time.onrender.com/docs">
    <img src="https://img.shields.io/badge/API-Swagger%20Docs-168A50?style=for-the-badge" alt="Open API documentation" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render&logoColor=black" alt="Deployed on Render" />
</p>

> **Live app:** [blinkit-delivery-time.onrender.com](https://blinkit-delivery-time.onrender.com/) ·
> [API health](https://blinkit-delivery-time.onrender.com/health) ·
> [Interactive API docs](https://blinkit-delivery-time.onrender.com/docs)

The app provides a browser form for order details and returns an estimated
delivery duration in minutes. It is a portfolio and learning project; the
estimate is not a guaranteed delivery time.

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

The latest local tuning run evaluates a chronological holdout beginning
2025-05-24, containing 9,081 orders. Tuning used an earlier chronological
validation period; the holdout was reserved for final model comparison.

| Model | Holdout MAE (min) ↓ | Holdout RMSE (min) ↓ | R² ↑ |
|---|---:|---:|---:|
| **Linear regression (selected)** | **1.859** | 3.326 | 0.213 |
| XGBoost | 1.881 | **3.292** | **0.229** |
| Random forest | 2.106 | 3.401 | 0.177 |
| Decision tree | 2.154 | 3.594 | 0.081 |

The selected model is chosen by lowest holdout MAE. In the latest run, linear
regression achieved an MAE of **1.859 minutes**; XGBoost had a slightly lower
RMSE and higher R². R² is not “prediction accuracy.” All metrics are for the
specified historical holdout and do not guarantee future performance.

**Deployment note:** The latest model artifact and these metrics are local
project files. The Render site will use the updated model only after the
artifact is committed and pushed to GitHub and Render completes a new deploy.

Training also records tuned parameters and validation metrics in
`reports/model_comparison.csv` and `reports/model_metadata.json`. Those
generated reports and the raw dataset are excluded from Git by default.

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
with `--tuning-trials` (the default search evaluates two sampled settings per
model, as well as its starting configuration). A run writes:

- `models/delivery_time_model.joblib`
- `reports/model_comparison.csv`
- `reports/model_metadata.json`

Run training to produce fresh metrics after changing model code, the dataset,
or tuning settings. Review the holdout results before replacing the deployed
model.

### Start the web app

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open the locally running app:

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

The app is deployed at **[blinkit-delivery-time.onrender.com](https://blinkit-delivery-time.onrender.com/)**.
The project includes a Dockerfile and `render.yaml`. The image contains the
API, HTML template, runtime dependencies, and trained model artifact; it does
not need the raw CSV or run model training during deployment.

1. Push the project to a GitHub repository, including
   `models/delivery_time_model.joblib`.
2. Keep the raw dataset, secrets, virtual environments, and generated reports
   out of Git; these are covered by `.gitignore`.
3. In Render, create a Blueprint or Docker Web Service connected to the repo.
4. Confirm `/health` is the health-check path and deploy.
5. Open the [live app](https://blinkit-delivery-time.onrender.com/); the API
   health check is at [`/health`](https://blinkit-delivery-time.onrender.com/health),
   and the interactive docs are at
   [`/docs`](https://blinkit-delivery-time.onrender.com/docs).

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
