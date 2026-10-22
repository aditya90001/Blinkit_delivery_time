# Blinkit Delivery-Time Project: Interview Questions

## Project overview

### 1. What problem does your project solve?

It estimates delivery duration in minutes from order information that can be
known when an order is placed. A FastAPI service serves predictions through
both a JSON endpoint and a small HTML interface.

### 2. What is the target variable?

`actual_delivery_mins`, a continuous numeric value. This is a regression
problem, not a classification problem.

### 3. What are the main inputs?

Checkout-time fields such as city, store, product category, payment mode,
order hour/date, quantity, price, basket value, and delivery distance. Encoded
calendar features are also generated.

### 4. What is the end-to-end architecture?

The EDA and `app/features.py` define cleaning and features; `app/training.py`
fits and evaluates the model; the fitted pipeline is saved as a Joblib
artifact; `app/main.py` loads it and exposes FastAPI routes; and
`templates/index.html` calls the API. Docker packages the API, HTML template,
and artifact for Render.

## Data and feature engineering

### 5. Why do you train only on delivered orders?

Cancelled or partially fulfilled orders did not complete a delivery. Their
delivery-duration fields can be placeholders, so using them would corrupt the
regression target.

### 6. How did you handle duplicate orders?

The reference EDA removes repeated `order_id` values while keeping the first
row. The shared training preparation applies the same rule.

### 7. How did you clean quantity and category fields?

Quantity strings such as `"2 units"` are parsed to numbers. Plausible missing
quantities can be inferred from basket value divided by unit price. Inconsistent
fruit-category spellings are normalized to one category.

### 8. How did you handle missing values and outliers?

Invalid and missing numeric inputs are normalized during feature preparation;
the model pipeline fits numeric median imputers and categorical imputers on
training data. Documented price and basket-value errors are handled before
modeling. Training-only imputers avoid learning holdout statistics.

### 9. Why exclude order IDs, customer IDs, ratings, and delivery partners?

Identifiers can encourage memorization rather than generalization. Ratings
and partner assignments are unavailable at checkout or known only after
delivery, which would leak future information into a checkout-time prediction.

### 10. Why exclude `stockout_flag`?

The supplied EDA found no meaningful variation in this field among delivered
training orders, so it does not provide useful signal for this target.

## Modeling and evaluation

### 11. Why compare linear regression, a tree, random forest, and XGBoost?

They provide interpretable linear and nonlinear baselines plus two ensemble
approaches. Comparing them on the same split tests whether model complexity
actually helps this dataset.

### 12. Why can linear regression beat XGBoost?

More complex models do not automatically generalize better. The available
checkout-time features and historical data may be better represented by a
simple linear relationship; tree ensembles can overfit or fail to capture
patterns that transfer to later dates. Only a consistent time-based
evaluation can establish which model is better.

### 13. Why use a chronological split instead of a random split?

Predictions are for future orders. Training on earlier orders and testing on
later dates better approximates that use case and reduces the risk that
near-identical dates appear in both training and test data.

### 14. What do MAE, RMSE, and R² tell you?

- **MAE:** average absolute prediction error, in minutes; lower is better.
- **RMSE:** like MAE but penalizes large misses more heavily; lower is better.
- **R²:** the fraction of target variance explained relative to a mean
  baseline; closer to 1 is better, 0 is mean-baseline-level, and below 0 is
  worse than that baseline.

### 15. What is the model's “accuracy”?

This is regression, so there is no single classification-style accuracy
percentage. The saved earlier evaluation reports linear regression MAE of
1.876 minutes, RMSE of 3.329 minutes, and R² of 0.212. Hyperparameter tuning
was started but stopped before updated results were saved; those earlier
numbers must not be presented as tuned results.

### 16. How would you improve the model responsibly?

First compare changes on the same untouched chronological holdout. Then
consider additional checkout-time predictors, a larger representative
dataset, robust treatment of long-tail targets, and validation by city and
date. Collect traffic, weather, and courier availability only if they can be
measured at prediction time and are available at serving time too.

### 17. What is a limitation of this model?

The dataset lacks live traffic, weather, and courier-availability signals.
Predictions describe historical patterns and are not a live navigation ETA or
an SLA guarantee.

## API and deployment

### 18. What does `/health` check?

It returns service status and whether the serialized prediction model loaded.
The response includes the loaded model name or `"unavailable"`.

### 19. What does `/predict` accept and return?

It accepts validated order features such as city and category, with optional
date, hour, store, quantity, price, order value, distance, and payment mode.
It returns `predicted_delivery_mins`, model name, and the `"minutes"` unit.

### 20. Why serialize the full scikit-learn pipeline?

Saving preprocessing together with the estimator ensures inference applies
the same feature transformation and category encoding used during training.

### 21. How is this deployed on Render?

Render builds the included Dockerfile, starts Uvicorn using Render's `PORT`,
and checks `/health`. The image includes runtime dependencies, the model
artifact, the API code, and the frontend template.

### 22. How would you monitor it after deployment?

Check the health endpoint and server logs, monitor prediction errors against
eventually observed delivery times, track feature/data drift, and retrain only
after evaluating new data with time-aware validation.
