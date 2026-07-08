"""
Three tabs, loads fast, tells a clear story in under 30 seconds:
Model Performance / Feature Store Explorer / Drift Monitor.
"""
from pathlib import Path

import duckdb
import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PulseML", layout="wide")
st.title("PulseML — Credit Risk Pipeline")

tab1, tab2, tab3 = st.tabs(["Model Performance", "Feature Store Explorer", "Drift Monitor"])

with tab1:
    st.subheader("Validation performance vs. baseline")
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("pulseml-credit-risk")
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            m = runs[0].data.metrics
            col1, col2 = st.columns(2)
            col1.metric("Validation AUC", f"{m.get('val_auc', 0):.3f}",
                        delta=f"{m.get('val_auc', 0) - m.get('baseline_auc', 0):+.3f} vs baseline")
            col2.metric("Validation AUC-PR", f"{m.get('val_auc_pr', 0):.3f}")
        else:
            st.info("No MLflow runs found yet. Run `python -m src.training.train` first.")
    except Exception as e:
        st.warning(f"Could not load MLflow experiment: {e}")

with tab2:
    st.subheader("Look up a single entity's point-in-time features")
    entity_id = st.number_input("Entity ID (sk_id_curr)", value=100002, step=1)
    db_path = Path("data/feature_store.duckdb")
    if db_path.exists():
        con = duckdb.connect(str(db_path), read_only=True)
        result = con.execute(
            "SELECT * FROM features_application WHERE sk_id_curr = ?", [entity_id]
        ).df()
        con.close()
        st.dataframe(result)
    else:
        st.info("Feature store not found. Run `python -m src.features.setup_feature_store` first.")

with tab3:
    st.subheader("Drift report (early vs. late application period)")
    report_path = Path("reports/drift_reports/drift_report_v1.html")
    if report_path.exists():
        st.components.v1.html(report_path.read_text(), height=800, scrolling=True)
    else:
        st.info("No drift report found. Run `python -m src.monitoring.drift_report` first.")
