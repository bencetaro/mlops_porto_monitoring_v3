import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse, os, logging, yaml
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgbm
import src.training.helpers as helpers
import mlflow

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def main(input_dir, config):
    logging.info("Start training script.")
    logging.info("Loading preprocessed data...")
    train = pd.read_csv(os.path.join(input_dir, "train_preprocessed.csv"))

    # Read config vars
    apply_rusr = config.get("apply_rusr", False)
    apply_skfcv = config.get("apply_skfcv", False)
    apply_mi_score = config.get("apply_mi_score", False)
    apply_fi_score = config.get("apply_fi_score", False)
    apply_cm = config.get("apply_cm", False)
    model_type = config.get("model_type", "lightgbm")
    n_skf_splits = config.get("n_skf_splits", 5)
    n_rs_iter = config.get("n_rs_iter", 3)
    random_state = config.get("random_state", 42)
    test_size = config.get("test_size", 0.2)

    # Setup MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.autolog(disable=True)

    pipeline_id = os.getenv('PIPELINE_ID')
    config_name = os.getenv("CONFIG_FILE", "default_config").split(".")[0]
    experiment_name = os.getenv('EXPERIMENT_NAME')
    mlflow.set_experiment(f"{experiment_name}_{config_name}_{pipeline_id}")

    # Separate features and target
    X = train.drop("target", axis=1)
    y = train["target"]
    logging.info(f"Train shape: {X.shape}, Class distribution:\n{y.value_counts()}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Balance dataset with RandomUnderSampler
    if apply_rusr:
        rus = RandomUnderSampler(sampling_strategy={0: int(y_train.value_counts()[1]*4), 1: y_train.value_counts()[1]}, random_state=random_state)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        logging.info(f"Balanced training set shape: {X_train.shape}, class ratio: {np.bincount(y_train)}")

    # Model training
    with mlflow.start_run(run_name=config_name):
        mlflow.set_tag("experiment_name", experiment_name)
        mlflow.set_tag("config_name", config_name)
        mlflow.set_tag("pipeline_id", pipeline_id)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("train_class_distribution", str(y_train.value_counts().to_dict()))
        mlflow.log_param("val_class_distribution", str(y_val.value_counts().to_dict()))
        mlflow.log_param("num_features", X_train.shape[1])

        # Calculate MI scores
        if apply_mi_score:
            logging.info("Calculating Mutual Information scores...")
            mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)
            mi_df = pd.DataFrame({"feature": X_train.columns, "mi_score": mi_scores}).sort_values("mi_score", ascending=False)
            logging.info(f"Top 10 features by Mutual Information:\n{mi_df.head(10)}")
            helpers.plot_mutual_info(mi_df.head(10), model_type)

        if model_type == "random_forest":
            if apply_skfcv:
                param_grid = config.get("skfcv_hparams", {}).get("random_forest", {"n_estimators": [100, 200],})
                model = RandomForestClassifier(random_state=random_state)
                cv = StratifiedKFold(n_splits=n_skf_splits, shuffle=True, random_state=random_state)
                logging.info("Starting Random Forest random search...")
                start = datetime.now()
                search = RandomizedSearchCV(model, param_grid, scoring="roc_auc", n_iter=n_rs_iter, cv=cv, n_jobs=-1, random_state=random_state)
                search.fit(X_train, y_train)
                logging.info(f"Training done in {datetime.now()-start}")
                best_model = search.best_estimator_
                logging.info(f"Best params: {search.best_params_}")
                train_time = datetime.now() - start

                if apply_fi_score:
                    fi_scores = best_model.feature_importances_
                    fi_df = pd.DataFrame({"feature": X_train.columns, "importance": fi_scores}).sort_values("importance", ascending=False)
                    logging.info(f"Top 10 features by Feature Importance:\n{fi_df.head(10)}")
                    helpers.plot_feature_importances(fi_df.head(10), model_type)

                if apply_cm:
                    y_pred_label = best_model.predict(X_val)
                    cm = confusion_matrix(y_val, y_pred_label)
                    helpers.plot_confusion_matrix(cm, model_type)

                # Validate & Save
                y_pred = best_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)
                logging.info(f"Validation ROC-AUC: {auc:.4f}")
                mlflow.log_params(search.best_params_)
                mlflow.log_metric("val_auc", auc)
                mlflow.log_metric("train_time_seconds", train_time.total_seconds())
                registered_name, version = helpers.mlflow_log(model_type, best_model)
                logging.info(f"Model registered: {registered_name}, version: {version}")

            else:
                logging.info("Start training Random Forest with default params...")
                start = datetime.now()
                params = config.get("hparams", {}).get("random_forest", {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'class_weight': 'balanced'})
                best_model = RandomForestClassifier(random_state=random_state)
                best_model.set_params(**params)
                best_model.fit(X_train, y_train)
                train_time = datetime.now() - start

                if apply_fi_score:
                    fi_scores = best_model.feature_importances_
                    fi_df = pd.DataFrame({"feature": X_train.columns, "importance": fi_scores}).sort_values("importance", ascending=False)
                    logging.info(f"Top 10 features by Feature Importance:\n{fi_df.head(10)}")
                    helpers.plot_feature_importances(fi_df.head(10), model_type)

                if apply_cm:
                    y_pred_label = best_model.predict(X_val)
                    cm = confusion_matrix(y_val, y_pred_label)
                    helpers.plot_confusion_matrix(cm, model_type)

                # Validate & Save
                y_pred = best_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)
                logging.info(f"Validation ROC-AUC: {auc:.4f}")
                mlflow.log_metric("val_auc", auc)
                mlflow.log_metric("train_time_seconds", train_time.total_seconds())
                registered_name, version = helpers.mlflow_log(model_type, best_model)
                logging.info(f"Model registered: {registered_name}, version: {version}")

        elif model_type == "lightgbm":
            if apply_skfcv:
                param_grid = config.get("skfcv_hparams", {}).get("lightgbm", {"n_estimators": [100, 200],})
                model = lgbm.LGBMClassifier(objective="binary", random_state=random_state)
                cv = StratifiedKFold(n_splits=n_skf_splits, shuffle=True, random_state=random_state)
                logging.info("Starting LightGBM random search...")
                start = datetime.now()
                search = RandomizedSearchCV(model, param_grid, scoring="roc_auc", n_iter=n_rs_iter, cv=cv, n_jobs=-1, random_state=random_state)
                search.fit(X_train, y_train)
                logging.info(f"Training done in {datetime.now()-start}")
                best_model = search.best_estimator_
                logging.info(f"Best params: {search.best_params_}")
                train_time = datetime.now() - start

                if apply_fi_score:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    lgbm.plot_importance(best_model, max_num_features=10, importance_type="gain", ax=ax, title=f"Feature Importance of {model_type} model")
                    plt.tight_layout()
                    mlflow.log_figure(fig, f"plots/feature_importances_{model_type}.png")
                    plt.close(fig)

                if apply_cm:
                    y_pred_label = best_model.predict(X_val)
                    cm = confusion_matrix(y_val, y_pred_label)
                    helpers.plot_confusion_matrix(cm, model_type)

                # Validate & Save
                y_pred = best_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)
                logging.info(f"Validation ROC-AUC: {auc:.4f}")
                mlflow.log_params(search.best_params_)
                mlflow.log_metric("val_auc", auc)
                mlflow.log_metric("train_time_seconds", train_time.total_seconds())
                registered_name, version = helpers.mlflow_log(model_type, best_model)
                logging.info(f"Model registered: {registered_name}, version: {version}")

            else:
                logging.info("Start training LightGBM with default params...")
                start = datetime.now()
                params = config.get("hparams", {}).get("lightgbm", {'subsample': 0.6, 'num_leaves': 15, 'n_estimators': 300, 'min_child_samples': 20,})    
                best_model = lgbm.LGBMClassifier(objective="binary", random_state=random_state)
                best_model.set_params(**params)
                best_model.fit(X_train, y_train)
                train_time = datetime.now() - start

                if apply_fi_score:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    lgbm.plot_importance(best_model, max_num_features=10, importance_type="gain", ax=ax, title=f"Feature Importance of {model_type} model")
                    plt.tight_layout()
                    mlflow.log_figure(fig, f"plots/feature_importances_{model_type}.png")
                    plt.close(fig)

                if apply_cm:
                    y_pred_label = best_model.predict(X_val)
                    cm = confusion_matrix(y_val, y_pred_label)
                    helpers.plot_confusion_matrix(cm, model_type)

                # Validate & Save
                y_pred = best_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)
                logging.info(f"Validation ROC-AUC: {auc:.4f}")
                mlflow.log_metric("val_auc", auc)
                mlflow.log_metric("train_time_seconds", train_time.total_seconds())
                registered_name, version = helpers.mlflow_log(model_type, best_model)
                logging.info(f"Model registered: {registered_name}, version: {version}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(args.input_dir, config)

