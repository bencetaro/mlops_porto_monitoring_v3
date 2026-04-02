# MLOps Inference & Monitoring Pipeline With Kubernetes (v3)

This project demonstrates a Kubernetes (Minikube) orchestrated single‑node demo cluster. It is designed as a portfolio-ready demo that emphasizes reproducibility, observability, and ML model lifecycle management.

---

## Architecture Overview

**Core services:**
- **MLflow**: experiment tracking and model registry
- **Postgres**: MLflow backend store
- **MinIO**: artifact storage (models, metrics, artifacts)
- **Inference API (FastAPI)**: production-style model serving
- **Streamlit**: user-friendly inference UI
- **Prometheus + Grafana**: monitoring and dashboards
- **Node Exporter**: host-level metrics

**Workflow:**
1. **Seed jobs** initialize required assets and monitoring configs.
2. **Data prep job** produces cleaned/processed datasets.
3. **Training job** runs multiple configurations (indexed/parallelized), logs runs to MLflow.
4. **Model selector** evaluates recent runs and sets the best model as `production` in MLflow.
5. **Inference stack** serves the selected model, with metrics and dashboards enabled.

---

## Kubernetes Orchestration

The Kubernetes manifests in `k8s/` are organized in a strict apply order to ensure infrastructure dependencies are ready before training and inference workloads start. The training job uses `Indexed Jobs` to run multiple model configurations, and the model selector promotes the best run in MLflow.

**Key points:**
- **Stateful dependencies** (Postgres/MinIO) are provisioned via PVCs
- **Job TTL cleanup** is enabled to avoid stale completed pods
- **ConfigMaps/Secrets** isolate runtime configuration
- **Inference + monitoring** are long-running Deployments with Services

---

## Training & Model Selection

- Training runs inside dedicated containers using configuration files under `src/training/configs/`.
- Each training job logs metrics and artifacts to MLflow.
- The model selector selects the best `val_auc` within the latest pipeline and sets its MLflow alias to `production`.

---

## Note
A complete step-by-step workflow is provided in `quickstart.md`.

---

## Project Structure

        root/
        ├── data/
        | ├── inference/
        | │ ├── test_inference_api_1.csv
        | │ └── test_inference_api_2.csv
        | └── training/
        |  └── raw/
        |   ├── test.csv.zip
        |   └── train.csv.zip
        ├── docker/
        | ├── Dockerfile.inference
        | ├── Dockerfile.training
        | ├── Dockerfile.streamlit
        | └── run_training_pipeline.sh
        ├── images/...
        ├── k8s/
        │ ├── 00-namespace.yml
        │ ├── 01-secrets.yml
        │ ├── 02-configmaps.yml
        │ ├── 03-storage.yml
        │ ├── 05-seed-jobs.yml
        │ ├── 10-infra.yml
        │ ├── 15-data-prep.yml
        │ ├── 20-training.yml
        │ ├── 25-model-selector.yml
        │ ├── 30-inference.yml
        │ └── 40-expose-templates.yml
        ├── monitoring/
        │ ├── prometheus.yml
        │ └── grafana/
        │  ├── datasource.yml
        │  ├── provider.yml
        │  └── dashboards/
        |   ├── InferenceDashb.json
        |   └── porto.json
        ├── scripts/
        │ └── healthcheck.sh
        ├── src/
        | ├── inference/
        | │ ├── api_service.py
        | │ ├── helpers.py
        | │ ├── client.py
        | │ ├── schemas.py
        | │ └── ui/
        | |  ├── inference_ui.py
        | |  ├── model_comparison.py
        | |  └── training_ui.py
        | └── training/
        |  ├── train_model.py
        |  ├── data_prep.py
        |  ├── helpers.py
        |  ├── select_best_model.py
        |  └── configs/
        |    ├── rf_baseline.yml
        |    ├── rf_tuned.yml
        |    ├── lgbm_baseline.yml
        |    └── lgbm_tuned.yml
        └── requirements.txt

---

## Snapshots (from v2 verion)

### Inference UI Snapshot

![Dashboard](https://github.com/bencetaro/mlops_porto_monitoring_v3/blob/master/images/6.png)

---

### Grafana Dashboard Snapshots

![Dashboard](https://github.com/bencetaro/mlops_porto_monitoring_v3/blob/master/images/1.png)
![Dashboard](https://github.com/bencetaro/mlops_porto_monitoring_v3/blob/master/images/2.png)
![Dashboard](https://github.com/bencetaro/mlops_porto_monitoring_v3/blob/master/images/3.png)

#### Display Control

- With dashboard environment variables we can also control which model or endpoint to be displayed in an interactive view.

![Dashboard](https://github.com/bencetaro/mlops_porto_monitoring_v3/blob/master/images/4.png)
![Dashboard](https://github.com/bencetaro/mlops_porto_monitoring_v3/blob/master/images/5.png)
