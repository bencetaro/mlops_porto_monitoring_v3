#!/bin/bash
set -e

PIPELINE_ID=$(cat /pipeline/pipeline_id)
echo "[INFO] Pipeline ID: $PIPELINE_ID"
export PIPELINE_ID

echo "[INFO] Waiting for MLflow server..."
until curl -s http://mlflow-server:5000/health; do
sleep 2
done
echo "[INFO] MLflow server is ready."

echo "[INFO] Training model (from $CONFIG_FILE)..."
python -m src.training.train_model --input_dir /data/training/processed --config /src/training/configs/$CONFIG_FILE
echo "[INFO] Training pipeline completed successfully."
