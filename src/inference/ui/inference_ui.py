import os
import streamlit as st
import requests
import pandas as pd
import time
import json


def _model_options(models_resp):
    aliases = models_resp.get("aliases", [])
    versions = [str(v) for v in models_resp.get("versions", [])]
    options = []
    for entry in aliases + versions:
        if entry not in options:
            options.append(entry)
    if not options:
        options = [models_resp.get("default", "production")]
    return options


def show_inference_ui():
    st.set_page_config(layout="wide")
    st.title("Porto Seguro's Safe Driver Prediction")
    st.header("Inference UI")
    st.markdown(
    """
    This demo showcases a machine learning system for insurance claim risk prediction
    based on the Porto Seguro Safe Driver Prediction dataset.

    [Original dataset source](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/overview)
    """
    )
    st.sidebar.header("Settings")

    api_base_url = st.sidebar.text_input("Inference API base URL", value="http://inference-api:8000")
    default_model = "production"

    try:
        models_resp = requests.get(f"{api_base_url}/models", timeout=2).json()
        model_options = _model_options(models_resp)
        default_model = models_resp.get("default", default_model)
    except Exception:
        model_options = [default_model]

    selected_model = st.sidebar.selectbox("Model version / alias", model_options)
    mode = st.sidebar.selectbox("Inference mode", ["Single inference", "Batch inference"])

    try:
        r = requests.get(f"{api_base_url}/health", timeout=2)
        st.sidebar.success("API healthy" if r.status_code == 200 else "API error")
    except Exception:
        st.sidebar.error("API unreachable")

    # Increase st.expander font size
    st.markdown("""
    <style>
    /* Expander header */
    .streamlit-expanderHeader {
        font-size: 22px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.expander("Project Overview", expanded=False, icon="🔹"):
        st.markdown(
        """
        This is a binary classification problem where we need to predict the probability of a driver 
        filing an insurance claim or not based on anonymized driver and vehicle characteristics.

        **Different models were trained:**
        - Random Forest with a simple parameter setup
        - LightGBM with a simple parameter setup
        - Random Forest with parameter search and Stratified K-Fold Cross-Validation
        - LightGBM with parameter search and Stratified K-Fold Cross-Validation

        *(Since this is just a demonstration project, experimentation was limited to only these model configurations.)*

        **Test the model:**
        - Single Inference (predict risk for one driver - demo columns only)
        - Batch Inference (upload a CSV and predict multiple drivers)
        """)

    with st.expander("Project Details", expanded=False, icon="🔹"):
        col1, col2 = st.columns([3,2]) # [3,2] sets ratios for cols
        with col1:
            with st.container(border=True):
                st.subheader("Processing Steps")
                step_cols = st.columns(2)
                step_cols[0].markdown("""
                #### Training Pipeline

                1. Raw data ingestion
                2. Feature preprocessing
                    - missing value handling
                    - scaling & encoding
                3. Random Undersampling
                4. Model training
                5. Model evaluation
                6. Model registration (MLflow)
                """)
                step_cols[1].markdown(
                """
                #### Inference Pipeline

                1. Request validation
                2. Apply preprocessing pipeline
                3. Model prediction
                4. Return probability score
                5. Record metrics (Prometheus)
                """
                )

        with col2:
            # Extract info from preproccesed dataset
            dataset_path = os.getenv("INFERENCE_DATASET_PATH", "/data/training/processed/train_preprocessed.csv")
            dataset_rows = None
            dataset_cols = None
            target_col = None
            if os.path.exists(dataset_path):
                try:
                    with open(dataset_path, "rb") as f:
                        dataset_rows = max(sum(1 for _ in f) - 1, 0)
                    header_cols = pd.read_csv(dataset_path, nrows=0).columns.tolist()
                    dataset_cols = len(header_cols)
                    target_col = "target" if "target" in header_cols else None
                except Exception:
                    dataset_rows = None
                    dataset_cols = None
                    target_col = None

            with st.container(border=True):
                st.subheader("Preprocessed Dataset Info")
                kpi_cols = st.columns(3)
                kpi_cols[0].metric("Rows", f"{dataset_rows:,}" if dataset_rows is not None else "N/A")
                kpi_cols[1].metric("Features", f"{dataset_cols - 1:,}" if dataset_cols and target_col else (f"{dataset_cols:,}" if dataset_cols else "N/A"))
                kpi_cols[2].metric("Target", "Binary")

    # ---
    st.divider()

    # Inference part
    if mode == "Single inference":
        st.header("Single record inference")
        with st.form("single_form"):
            record_id = st.number_input("id", min_value=0, step=1)
            ps_ind_01 = st.number_input("ps_ind_01", value=0)
            ps_ind_02_cat = st.number_input("ps_ind_02_cat", value=0)
            ps_ind_03 = st.number_input("ps_ind_03", value=0)
            ps_car_01_cat = st.number_input("ps_car_01_cat", value=0)
            extra_cols = st.text_area("Optional extra features as JSON object", value="{}", height=120)
            submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                extra_payload = {}
                if extra_cols.strip():
                    extra_payload = json.loads(extra_cols)
                    if not isinstance(extra_payload, dict):
                        raise ValueError("Extra JSON must be an object.")
                features = {
                    "ps_ind_01": ps_ind_01,
                    "ps_ind_02_cat": ps_ind_02_cat,
                    "ps_ind_03": ps_ind_03,
                    "ps_car_01_cat": ps_car_01_cat,
                }
                features.update(extra_payload)
                payload = {"root": {"id": int(record_id), **features}}

                start = time.time()
                r = requests.post(f"{api_base_url}/predict?model={selected_model}", json=payload)
                latency = time.time() - start
                if r.status_code == 200:
                    res = r.json()
                    st.success("Prediction successful")
                    st.metric("Prediction", round(res["prediction"], 4))
                    st.caption(f"Latency: {latency:.3f}s")
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))

    else:
        st.header("Batch inference")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            has_id = "id" in df.columns
            ids = df["id"].reset_index(drop=True) if has_id else None
            df_data = df.drop(columns=["id"]) if has_id else df

            st.subheader("Input preview")
            st.dataframe(df_data.head().astype(str))

            if st.button("Run batch inference"):
                payload = {"root": df_data.to_dict(orient="records")}
                try:
                    start = time.time()
                    r = requests.post(f"{api_base_url}/predict/batch?model={selected_model}", json=payload)
                    latency = time.time() - start
                    if r.status_code == 200:
                        preds = pd.DataFrame(r.json())
                        preds["prediction"] = preds["prediction"].astype(float)
                        result = pd.concat([ids, preds], axis=1) if has_id else pd.concat([df_data.reset_index(drop=True), preds], axis=1)
                        st.success("Batch prediction successful")
                        st.caption(f"Latency: {latency:.3f}s")
                        st.subheader("Predictions")
                        st.dataframe(result.head().astype(str))
                        st.download_button("Download CSV", result.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(str(e))

    st.divider()
    st.caption(
    """
    Demo ML system built with **Streamlit, FastAPI, MLflow, and Docker.**
    """
    )
