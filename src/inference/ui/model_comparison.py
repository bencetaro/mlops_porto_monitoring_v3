import streamlit as st
import pandas as pd
from src.inference.helpers import list_model_meta
import plotly.express as px


def plot_auc_chart(df):
    metric_col = "val_auc" if "val_auc" in df.columns else "auc"
    df = df.dropna(subset=[metric_col]).sort_values(metric_col)
    fig = px.bar(df, x=metric_col, y="model_name", orientation="h", text=metric_col)
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")


def show_model_comparison():
    st.title("Model Comparison")
    metas = list_model_meta()
    if not metas:
        st.warning("No model metadata found in MLflow.")
        return

    model_names = [m["model_name"] for m in metas]
    selected = st.multiselect("Which models?", model_names, default=model_names)
    all_keys = sorted(set().union(*[m.keys() for m in metas]) - {"model_name"})
    default_attrs = [k for k in ["val_auc", "train_time_seconds", "aliases", "config_name", "pipeline_id"] if k in all_keys]
    attributes = st.multiselect("Attributes to compare?", all_keys, default=default_attrs)

    if not selected or not attributes:
        st.info("Select models and attributes")
        return

    rows = []
    for m in metas:
        if m["model_name"] not in selected:
            continue
        row = {"model_name": m["model_name"]}
        for attr in attributes:
            row[attr] = m.get(attr, None)
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df.astype(str))
    if "auc" in df.columns or "val_auc" in df.columns:
        st.subheader("AUC Comparison")
        plot_auc_chart(df)
