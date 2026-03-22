import os
import time
import mlflow
from mlflow.tracking import MlflowClient

time.sleep(120)
MODEL_REGISTRY_NAME = "porto-training"

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

client = MlflowClient()

# read pipeline id
with open("/pipeline/pipeline_id") as f:
    pipeline_id = f.read().strip()

print(f"Selecting best model for pipeline_id={pipeline_id}")

# get all experiments
experiments = client.search_experiments()

best_run = None
best_auc = -1

for exp in experiments:

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.pipeline_id = '{pipeline_id}'",
        order_by=["metrics.val_auc DESC"],
        max_results=1,
    )

    if not runs:
        continue

    run = runs[0]
    auc = run.data.metrics.get("val_auc", 0)

    if auc > best_auc:
        best_auc = auc
        best_run = run

if best_run is None:
    raise RuntimeError("No runs found for current pipeline.")

print(f"Best run: {best_run.info.run_id}, val_auc={best_auc}")

versions = client.search_model_versions(
    f"name='{MODEL_REGISTRY_NAME}'"
)

best_version = None

for v in versions:
    if v.run_id == best_run.info.run_id:
        best_version = v.version
        break

if best_version is None:
    raise RuntimeError("Best run model not found in registry")

print(f"Best model version: {best_version}")

# assign alias
client.set_registered_model_alias(
    name=MODEL_REGISTRY_NAME,
    alias="production",
    version=best_version,
)

print(f"Alias 'production' now points to version {best_version}")
