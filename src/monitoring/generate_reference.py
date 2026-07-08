"""
Splits the feature store by application_date into an "early" (reference) and
"late" (current) period. Real distribution shift over time is far more
convincing -- and honest -- in a demo than synthetically injected drift.
"""
import duckdb
import pandas as pd

from src.features.point_in_time import get_training_feature_set
from src.training.train import FEATURE_COLS


def get_reference_and_current(split_quantile: float = 0.7):
    con = duckdb.connect("data/feature_store.duckdb", read_only=True)
    entities = con.execute("SELECT sk_id_curr, application_date FROM entities ORDER BY application_date").df()
    split_date = entities["application_date"].quantile(split_quantile)

    early_ids = entities.loc[entities["application_date"] <= split_date, "sk_id_curr"].tolist()
    late_ids = entities.loc[entities["application_date"] > split_date, "sk_id_curr"].tolist()

    reference_df = get_training_feature_set(con, early_ids, pd.Timestamp.now())[FEATURE_COLS]
    current_df = get_training_feature_set(con, late_ids, pd.Timestamp.now())[FEATURE_COLS]
    con.close()

    print(f"Reference (early) period: {len(reference_df)} rows, up to {split_date}")
    print(f"Current (late) period: {len(current_df)} rows, after {split_date}")
    return reference_df.fillna(0), current_df.fillna(0)


if __name__ == "__main__":
    get_reference_and_current()
