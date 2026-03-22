import os
import requests
import streamlit as st

from src.inference.ui.inference_ui import show_inference_ui
from src.inference.ui.training_ui import show_training_ui
from src.inference.ui.model_comparison import show_model_comparison

st.set_page_config(page_title="MLOps UI", layout="wide")

st.sidebar.title("Navigation")

API_BASE_URL = os.getenv("INFERENCE_API_BASE_URL", "http://inference-api:8000")

def _track_page_view(page_name: str) -> None:
    page_key = page_name.strip().lower().replace(" ", "_")
    try:
        requests.post(f"{API_BASE_URL}/ui/page_view", params={"page": page_key}, timeout=0.5)
    except Exception:
        pass

page = st.sidebar.radio(
    "Go to",
    [
        "Inference UI",
        "Training UI",
        "Model Comparison"
    ]
)

if st.session_state.get("last_page") != page:
    _track_page_view(page)
    st.session_state["last_page"] = page

if page == "Inference UI":
    show_inference_ui()

elif page == "Training UI":
    show_training_ui()

elif page == "Model Comparison":
    show_model_comparison()
