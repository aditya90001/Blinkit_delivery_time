# Blinkit Delivery-Time Prediction

An end-to-end regression project based on the supplied Blinkit EDA: data
exploration, cleaning, chronological model evaluation, and a FastAPI service
containerized for Render.

Project explanation, interview preparation, and deployment instructions are in
[`project_guide/`](./project_guide/README.md).

## Dataset

The supplied EDA reports 52,750 rows, 18 source columns, 52,500 distinct order
IDs, eight cities, and 180 order dates. It identifies text-formatted quantity
values, duplicate order IDs, inconsistent fruit-category labels, extreme
`unit_price` and `order_value` values, missing values, and invalid ratings.
The EDA also establishes that cancelled and partially fulfilled rows carry
placeholder delivery times; only delivered rows are used to train the
regression target.

The provided dataset is `data/compressed_data (2).csv`. Its observed values
match the supplied EDA: 52,750 rows, 18 columns, 250 repeated order IDs,
44,611 delivered rows before deduplication, eight cities, and 180 dates.
Deduplication leaves 44,402 delivered training rows. Actual model metrics are
computed against a chronological holdout rather than being assumed in advance.
The delivered-order median is 13.9 minutes, and 94.99% of eligible deliveries
exceed the dataset's constant 10-minute promise.

## Features and target

The API predicts `actual_delivery_mins` in minutes for an order at checkout.
The common training/API pipeline uses:

- `city`, `dark_store_id`, `category`, `payment_mode`
- `order_hour`, cyclical hour/weekday/month encodings, weekday/month/day of
  year, weekend, missing-value indicators, `quantity`, `unit_price`,
  `order_value`, and `distance_km`

The implementation intentionally excludes `order_id`, `customer_id`,
`delivery_partner_id`, `order_status`, `customer_rating`, and
`promised_delivery_mins`: identifiers are not robust predictors; status,
assigned rider, and rating are unavailable or outcome-derived at checkout;
and the supplied EDA reports that the promise is constant at 10 minutes.
`stockout_flag` is not used: inspection confirms every delivered training
order has a zero stockout flag, so the model cannot learn an effect from it.
It remains useful in the EDA for understanding cancellation and fulfillment.

Data preparation normalizes category labels and quantity strings, treats
prices above Rs. 1,500 and documented flat Rs. 25,000 order values as invalid,
derives missing quantity from plausible value/price pairs, and uses learned
training-only imputers/encoders. The notebook follows the supplied EDA's
cleaning and visualization sequence while retaining train-only imputation for
model evaluation. Duplicate-ID behavior follows the provided EDA
(`keep="first"`). Model comparisons use delivered records and a chronological
80/20 holdout when valid dates are available.

### Actual training results

Evaluation uses dates through 2025-05-23 for training and the subsequent
orders from 2025-05-24 for the holdout. MAE is the selection metric; the
selected pipeline is refitted on all 44,402 eligible delivered orders.
For reference, always predicting the training median (13.9 minutes) scores
2.335-minute MAE and 3.774-minute RMSE on that same holdout.

| Model | Holdout MAE (min) | RMSE (min) | R² |
|---|---:|---:|---:|
| Linear Regression (selected) | 1.876 | 3.329 | 0.212 |
| Random Forest | 2.260 | 3.552 | 0.103 |
| Decision Tree | 2.348 | 3.901 | -0.082 |
| XGBoost | 2.350 | 3.958 | -0.114 |

On this chronological holdout, linear regression generalizes best among the
four requested models. The selected model improves holdout MAE by about 20% against the median
baseline. The modest R² and RMSE substantially above MAE point
to unusually long delivery times that none of these models predicts well.
Treat outputs as estimates, not an SLA guarantee.

## Run locally

Use Python 3.11 or later. In PowerShell, from the project folder:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train all four regressors on the provided CSV:

```powershell
.\venv\Scripts\python.exe train.py
```

Alternatively:

```powershell
.\venv\Scripts\python.exe train.py --data "C:\path\to\compressed_data.csv"
```

The training command tunes regularized linear regression, Decision Tree,
Random Forest, and XGBoost against a chronological validation slice within the
training dates. Candidate models use a deterministic sample of up to 12,000
rows from that earlier fit period; the full chronological validation window
is used for selection, and the later-date holdout is kept out of
hyperparameter selection.
It reports holdout MAE (minutes), RMSE, R², validation MAE, and selected
parameters. The lowest-holdout-MAE pipeline is refitted on all valid delivered
records, and the run writes:

- `models/delivery_time_model.joblib`
- `reports/model_comparison.csv`
- `reports/model_metadata.json`

Choose the number of randomized hyperparameter settings per model with
`.\venv\Scripts\python.exe train.py --tuning-trials 5`. The default is 2,
plus each model's default settings are evaluated during tuning. Tuning is not
guaranteed to improve future-date performance.

Start the API after training:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/` for the prediction UI, `http://127.0.0.1:8001/docs`
for interactive API docs, and `http://127.0.0.1:8001/health` to check model
availability.
`POST /predict` accepts an order date (optional), hour (optional), city,
dark-store ID (optional), category, quantity, unit price, order value,
distance, and payment mode (optional). Missing optional features are imputed
by the fitted pipeline. The response field is `predicted_delivery_mins`.
The responsive HTML prediction form is served
directly by the same API backend. The browser form submits to `/predict` and
reads model availability from `/health`. The frontend lives in
`templates/index.html`.

Run the project tests:

```powershell
.\venv\Scripts\python.exe -m pip install pytest httpx
.\venv\Scripts\python.exe -m pytest
node --test tests/test_frontend_script.js
```

## EDA notebook

Open `notebooks/blinkit_delivery_time_eda.ipynb` in VS Code or Jupyter. Run
from the project folder after placing the CSV under `data/`. The notebook
reviews data shape/types, duplicate IDs, missingness, target eligibility,
cleaning, target distribution, city/hour patterns, feature relationships,
and the same four-model temporal comparison. It shares the project's
preprocessing and training modules.

## Docker and Render

The Docker image loads the already trained
`models/delivery_time_model.joblib` artifact. Deployment does not retrain all
four models or require the raw CSV in the image. Retrain locally with the CSV
when you need to update the model, and include the refreshed artifact in the
Render build context. Do not publish sensitive data or credentials in a
public repository or image.

Build and run locally, from a folder containing the CSV:

```powershell
docker build -t blinkit-delivery-api .
docker run --rm -p 8000:8000 blinkit-delivery-api
```

Deploy the Docker service with `render.yaml`. Render routes the service
through the automatically supplied `PORT`; `/health` is configured as its
health check.

## Project layout

```text
app/
  features.py     Shared cleaning, feature schema, training-row selection
  training.py     Temporal split, four-model comparison, artifact writing
  main.py         FastAPI validation, health, and prediction endpoints
data/             compressed_data (2).csv (provided training dataset)
templates/        Responsive HTML prediction frontend
notebooks/        EDA and model comparison notebook
tests/            Feature-cleaning and API tests
train.py          CLI training entry point
Dockerfile        Load the trained model and serve the API
render.yaml       Render Docker service configuration
```
#   B l i n k i t _ d e l i v e r y _ t i m e  
 