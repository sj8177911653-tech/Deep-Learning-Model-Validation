import streamlit as st

st.set_page_config(
    page_title="Model Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Deep Learning Model Performance Dashboard")
st.write("Robust Model Validation Suite")

st.subheader("Model Evaluation Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Baseline Accuracy", "97.08%")
col2.metric("Dropout Accuracy", "97.65%")
col3.metric("L2 Accuracy", "96.79%")
col4.metric("Early Stopping", "97.06%")

st.divider()

st.subheader("Accuracy Comparison")

st.image("accuracy_comparison.png", use_container_width=True)

st.subheader("Loss Comparison")

st.image("loss_comparison.png", use_container_width=True)

st.divider()

st.subheader("Learning Curves")

st.image("accuracy_learning_curves.png", use_container_width=True)
st.image("loss_learning_curves.png", use_container_width=True)

st.success("Dashboard Loaded Successfully!")