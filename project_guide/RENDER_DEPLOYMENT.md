# Deploy the Blinkit App on Render

## What is already configured

- `Dockerfile` installs the API requirements, copies the API, model, and HTML
  template, and starts Uvicorn on Render's `PORT`.
- `render.yaml` configures a Docker web service with `/health` as its health
  check.
- `models/delivery_time_model.joblib` is copied into the image. Render does
  not need the raw CSV or a training run during deployment.
- The raw dataset is excluded by `.dockerignore`.

## Before deploying

1. Ensure the project is in a GitHub repository connected to your Render
   account. Render deploys from a Git repository; the current local workspace
   has no Git repository initialized.
2. Include `Dockerfile`, `render.yaml`, `requirements-api.txt`, `app/`,
   `templates/`, and `models/delivery_time_model.joblib` in the pushed branch.
3. Do not commit `.env`, credentials, or private data. The dataset is excluded
   from the Docker context; do not publish it just to deploy the API.

## Render dashboard steps

1. In Render, choose **New +** → **Blueprint** if you want Render to read
   `render.yaml`, or choose **Web Service** and connect the repository.
2. Select the branch containing these files and approve the repository access
   request if prompted.
3. For a Blueprint, review the detected service `blinkit-delivery-time-api`.
   For a manually created service, select **Docker** as the runtime and use
   the repository root as the build context.
4. Confirm the health-check path is `/health`. Do not hard-code a port:
   Render provides `PORT` and the Docker command already uses it.
5. Create the service and wait for the Docker build and deployment to finish.
6. Open the Render URL. The HTML frontend is at `/`, API health at `/health`,
   and interactive API docs at `/docs`.

## Validate the deployed service

The health response should contain `"status":"ok"` and
`"model_loaded":true`. Then submit a test order in the frontend. The API can
also be tested from the Render page's `/docs` endpoint with `POST /predict`.
If `/health` reports the model is not loaded, review the build log and verify
that `models/delivery_time_model.joblib` exists in the deployed branch and
that the runtime dependency versions match the environment that created it.

## Redeploy after changing the model

Train locally using the project CSV, review the holdout metrics, and include
the refreshed `models/delivery_time_model.joblib` in a commit. Push that
commit to the connected branch; Render will build and deploy the new artifact.
The current model report is from the previous, untuned run. The interrupted
tuning attempt did not produce verified new metrics.

## Current deployment blocker

This workspace is not a Git repository, so it cannot be connected directly to
Render yet. Push the project to GitHub first, then connect that repository in
Render. Keep the dataset private and outside the image.
