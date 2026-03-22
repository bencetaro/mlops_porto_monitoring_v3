import matplotlib.pyplot as plt
import mlflow, os
from mlflow.tracking import MlflowClient

MODEL_REGISTRY_NAME = "porto-training"

def plot_mutual_info(mi_df, model_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(mi_df["feature"], mi_df["mi_score"])
    ax.set_xlabel("MI Score")
    ax.set_title("Feature Importance based on MI")
    ax.invert_yaxis()
    plt.tight_layout()
    mlflow.log_figure(fig, f"plots/mutual_info_{model_type}.png")
    plt.close(fig)

def plot_feature_importances(fi_df, model_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_df["feature"], fi_df["importance"])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance of {model_type}")
    ax.invert_yaxis()
    plt.tight_layout()
    mlflow.log_figure(fig, f"plots/feature_importances_{model_type}.png")
    plt.close(fig)

def plot_confusion_matrix(cm, model_type):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    tick_marks = range(len(cm))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    mlflow.log_figure(fig, f"plots/confusion_matrix_{model_type}.png")
    plt.close(fig)

def mlflow_log(model_type, model):
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id

    if model_type == "random_forest":
        mlflow.sklearn.log_model(model, artifact_path="model")
    elif model_type == "lightgbm":
        mlflow.lightgbm.log_model(model, artifact_path="model")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_REGISTRY_NAME
    )
    return MODEL_REGISTRY_NAME, registered_model.version
