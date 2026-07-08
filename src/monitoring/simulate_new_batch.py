"""
End-to-end "new data arrived" simulation: take the late time-slice, score it
with the registered model, and regenerate the drift report. This is the demo
moment that proves the pipeline is actually wired end-to-end.
"""
from src.monitoring.drift_report import generate_drift_report
from src.monitoring.generate_reference import get_reference_and_current
from src.serving.predict import score_batch
import duckdb
import pandas as pd


def run(stage: str = "Staging"):
    con = duckdb.connect("data/feature_store.duckdb", read_only=True)
    entities = con.execute("SELECT sk_id_curr, application_date FROM entities ORDER BY application_date").df()
    con.close()

    split_date = entities["application_date"].quantile(0.7)
    new_batch_ids = entities.loc[entities["application_date"] > split_date, "sk_id_curr"].tolist()

    scored = score_batch(new_batch_ids, stage=stage)
    print(f"Scored {len(scored)} new-batch entities. Mean score: {scored['score'].mean():.4f}")

    report_path = generate_drift_report(version="latest_batch")
    return scored, report_path


if __name__ == "__main__":
    run()
