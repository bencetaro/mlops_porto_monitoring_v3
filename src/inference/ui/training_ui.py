import streamlit as st
import os
import pandas as pd
from src.inference.helpers import list_model_meta, load_meta, get_plot_path


def show_training_ui():
    st.title("Training Dashboard")

    metas = list_model_meta()
    if not metas:
        st.warning("No training metadata found in MLflow.")
        return

    selected = st.selectbox("Select model", [m["model_name"] for m in metas])
    meta = load_meta(selected)
    if not meta:
        st.error("Meta missing")
        return

    st.subheader("Training Metadata")
    meta_df = pd.DataFrame([(k, v) for k, v in meta.items()], columns=["field", "value"])
    st.table(meta_df.astype(str))

    for plot_type, name in [("fi", "Feature Importance"), ("mi", "Mutual Information"), ("cm", "Confusion Matrix")]:
        st.subheader(f"{name} Plot")
        path = get_plot_path(selected, plot_type, meta=meta)
        if os.path.exists(path):
            st.image(path)
        else:
            st.info(f"{name} plot not available from current backend.")
