import os
import json
from functools import lru_cache
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


MODEL_NAME = os.getenv("MODEL_NAME", "porto-training")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@lru_cache(maxsize=2)
def _expected_feature_order(prep_path: str):
    """Read the training feature order from the preprocessed training CSV."""
    processed_dir = os.path.dirname(prep_path)
    train_preprocessed = os.path.join(processed_dir, "train_preprocessed.csv")
    if not os.path.exists(train_preprocessed):
        return None

    cols = pd.read_csv(train_preprocessed, nrows=0).columns.tolist()
    return [c for c in cols if c != "target"]


def _preprocess_legacy_dict(df: pd.DataFrame, preprocessors: dict) -> pd.DataFrame:
    """Apply legacy numeric and categorical transformers to raw input."""
    num_transformer = preprocessors["num"]
    cat_transformer = preprocessors["cat"]

    numeric_cols = list(num_transformer.feature_names_in_)
    categorical_cols = list(cat_transformer.feature_names_in_)

    work_df = df.copy()
    for col in numeric_cols + categorical_cols:
        if col not in work_df.columns:
            work_df[col] = np.nan

    x_num = num_transformer.transform(work_df[numeric_cols])
    x_cat = cat_transformer.transform(work_df[categorical_cols])
    return pd.concat(
        [
            pd.DataFrame(x_num, columns=numeric_cols, index=work_df.index),
            pd.DataFrame(x_cat, columns=categorical_cols, index=work_df.index),
        ],
        axis=1,
    )


def _preprocess_training_tuple(df: pd.DataFrame, preprocessors: tuple, prep_path: str) -> pd.DataFrame:
    """Apply the training-time preprocessing tuple to raw input."""
    (
        num_imputer,
        cat_imputer,
        bin_imputer,
        scaler,
        le_encoder,
        cols_to_drop,
        bin_cols,
        cat_cols,
        num_cols,
    ) = preprocessors

    bin_cols = list(bin_cols)
    cat_cols = list(cat_cols)
    num_cols = list(num_cols)

    work_df = df.copy()
    work_df.replace(-1, np.nan, inplace=True)
    work_df.drop(columns=[c for c in cols_to_drop if c in work_df.columns], inplace=True, errors="ignore")

    for expected_col in num_cols:
        if expected_col not in work_df.columns and expected_col.endswith("_num"):
            base_col = expected_col[:-4]
            if base_col in work_df.columns:
                work_df[expected_col] = work_df[base_col]

    required_cols = list(dict.fromkeys(num_cols + cat_cols + bin_cols))
    for col in required_cols:
        if col not in work_df.columns:
            work_df[col] = np.nan

    num_imputed = pd.DataFrame(num_imputer.transform(work_df[num_cols]), columns=num_cols, index=work_df.index)
    cat_imputed = pd.DataFrame(cat_imputer.transform(work_df[cat_cols]), columns=cat_cols, index=work_df.index)
    bin_imputed = pd.DataFrame(bin_imputer.transform(work_df[bin_cols]), columns=bin_cols, index=work_df.index)

    x_num = scaler.transform(num_imputed)
    x_cat = le_encoder.transform(cat_imputed)
    x_bin = np.round(bin_imputed)

    x = pd.concat(
        [
            pd.DataFrame(x_num, columns=num_cols, index=work_df.index),
            pd.DataFrame(x_cat, columns=cat_cols, index=work_df.index),
            pd.DataFrame(x_bin, columns=bin_cols, index=work_df.index),
        ],
        axis=1,
    )

    expected_order = _expected_feature_order(prep_path)
    if expected_order:
        for col in expected_order:
            if col not in x.columns:
                x[col] = 0.0
        x = x[expected_order]
    return x


def inference_preprocessing(df: pd.DataFrame, prep_path: str):
    """Load preprocessors and transform a raw dataframe for inference."""
    if not os.path.exists(prep_path):
        raise RuntimeError(f"Preprocessor file not found at {prep_path}")

    preprocessors = joblib.load(prep_path)
    try:
        if isinstance(preprocessors, dict) and "num" in preprocessors and "cat" in preprocessors:
            return _preprocess_legacy_dict(df, preprocessors)
        if isinstance(preprocessors, tuple) and len(preprocessors) >= 9:
            return _preprocess_training_tuple(df, preprocessors, prep_path)
        raise RuntimeError(f"Unsupported preprocessor format at {prep_path}")
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {e}") from e


def _legacy_metas(output_dir="/output"):
    """Load legacy model metadata JSON files from disk."""
    params_dir = os.path.join(output_dir, "params")
    if not os.path.exists(params_dir):
        return []
    metas = []
    for file_name in os.listdir(params_dir):
        if file_name.startswith("meta_") and file_name.endswith(".json"):
            with open(os.path.join(params_dir, file_name), encoding="utf-8") as jf:
                metas.append(json.load(jf))
    return metas


def _registry_alias_map(client: MlflowClient):
    """Build a map of MLflow aliases to their associated versions."""
    try:
        registered = client.get_registered_model(MODEL_NAME)
        aliases = getattr(registered, "aliases", {}) or {}
        if isinstance(aliases, dict):
            return aliases
        return {getattr(a, "alias", ""): str(getattr(a, "version", "")) for a in aliases}
    except Exception:
        return {}


def _mlflow_metas():
    """Collect model metadata from MLflow runs and registry versions."""
    client = MlflowClient()
    alias_map = _registry_alias_map(client)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    rows = []
    for v in versions:
        run = client.get_run(v.run_id)
        aliases = [a for a, ver in alias_map.items() if str(ver) == str(v.version)]
        rows.append(
            {
                "model_name": f"{MODEL_NAME}:v{v.version}",
                "version": int(v.version),
                "aliases": ", ".join(sorted(aliases)) if aliases else "",
                "val_auc": run.data.metrics.get("val_auc"),
                "train_time_seconds": run.data.metrics.get("train_time_seconds"),
                "config_name": run.data.tags.get("config_name"),
                "pipeline_id": run.data.tags.get("pipeline_id"),
                "model_type": run.data.params.get("model_type"),
                "run_id": v.run_id,
            }
        )
    rows.sort(key=lambda r: r["version"], reverse=True)
    return rows


def list_model_meta(output_dir="/output"):
    """Return model metadata from legacy files or MLflow registry."""
    legacy = _legacy_metas(output_dir=output_dir)
    if legacy:
        return legacy
    return _mlflow_metas()


def load_meta(model_name, output_dir="/output"):
    """Load a single model's metadata by name."""
    legacy_path = os.path.join(output_dir, "params", f"meta_{model_name}.json")
    if os.path.exists(legacy_path):
        with open(legacy_path, encoding="utf-8") as file:
            return json.load(file)

    metas = list_model_meta(output_dir=output_dir)
    for meta in metas:
        if meta.get("model_name") == model_name:
            return meta
    return None


def _plot_artifact_candidates(plot_type: str, model_type: str = None):
    """Return possible MLflow artifact paths for a given plot type."""
    names = []
    if plot_type == "fi":
        names.append(f"plots/feature_importances_{model_type}.png" if model_type else "")
        names += ["plots/feature_importances_lightgbm.png", "plots/feature_importances_random_forest.png"]
    elif plot_type == "mi":
        names.append(f"plots/mutual_info_{model_type}.png" if model_type else "")
        names += ["plots/mutual_info_lightgbm.png", "plots/mutual_info_random_forest.png"]
    elif plot_type == "cm":
        names.append(f"plots/confusion_matrix_{model_type}.png" if model_type else "")
        names += ["plots/confusion_matrix_lightgbm.png", "plots/confusion_matrix_random_forest.png"]
    return [n for n in names if n]


def get_plot_path(model_name, plot_type, output_dir="/output", meta=None):
    """Resolve a plot path locally or by downloading from MLflow artifacts."""
    local_path = os.path.join(output_dir, "plots", f"{plot_type}_{model_name}.png")
    if os.path.exists(local_path):
        return local_path

    if not meta:
        return local_path

    run_id = meta.get("run_id")
    if not run_id:
        return local_path

    model_type = meta.get("model_type")
    for artifact_path in _plot_artifact_candidates(plot_type, model_type):
        try:
            downloaded = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
            if downloaded and Path(downloaded).exists():
                return downloaded
        except Exception:
            continue
    return local_path
