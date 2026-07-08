"""
The Phase 1 milestone script: ingest raw data, compute all feature groups,
load them into DuckDB, and prove point-in-time retrieval works for a sample
of entities. Run this after src/ingestion/load_raw.py.
"""
import duckdb
import pandas as pd
from pathlib import Path

from src.features.build_features import (
    compute_application_features,
    compute_bureau_features,
    compute_previous_loan_features,
)
from src.features.point_in_time import get_training_feature_set

DB_PATH = "data/feature_store.duckdb"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def build_store():
    app_df = pd.read_csv("data/raw/application_train.csv")
    bureau_df = pd.read_csv("data/raw/bureau.csv")
    prev_df = pd.read_csv("data/raw/previous_application.csv")

    con = duckdb.connect(DB_PATH)
    con.execute(SCHEMA_PATH.read_text())

    entities = app_df[["SK_ID_CURR", "TARGET", "APPLICATION_DATE"]].rename(
        columns={"SK_ID_CURR": "sk_id_curr", "TARGET": "target", "APPLICATION_DATE": "application_date"}
    )
    con.execute("DELETE FROM entities")
    con.register("entities_df", entities)
    con.execute(f"INSERT INTO entities ({', '.join(entities.columns)}) SELECT * FROM entities_df")

    app_features = compute_application_features(app_df)
    bureau_features = compute_bureau_features(bureau_df)
    prev_features = compute_previous_loan_features(prev_df)

    for table, feats in [
        ("features_application", app_features),
        ("features_bureau_history", bureau_features),
        ("features_previous_loans", prev_features),
    ]:
        con.execute(f"DELETE FROM {table}")
        con.register("feats_df", feats)
        con.execute(f"INSERT INTO {table} ({', '.join(feats.columns)}) SELECT * FROM feats_df")

    con.close()
    print(f"Feature store built at {DB_PATH}: {len(entities)} entities.")


def sanity_check_point_in_time():
    con = duckdb.connect(DB_PATH, read_only=True)
    sample_ids = con.execute("SELECT sk_id_curr FROM entities LIMIT 5").df()["sk_id_curr"].tolist()
    result = get_training_feature_set(con, sample_ids, pd.Timestamp.now())
    con.close()
    print(f"Point-in-time retrieval OK: got {len(result)} rows for {len(sample_ids)} sample entities.")
    return result


if __name__ == "__main__":
    build_store()
    sanity_check_point_in_time()
