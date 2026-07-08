"""
Loads the registered (Staging or Production) model and scores a batch of
entity IDs using point-in-time-correct features as of "now".
"""
import duckdb
import mlflow
import pandas as pd

from src.features.point_in_time import get_training_feature_set
from src.training.train import FEATURE_COLS
from src.training.registry import MODEL_NAME


def score_batch(entity_ids: list[int], stage: str = "Staging") -> pd.DataFrame:
    con = duckdb.connect("data/feature_store.duckdb", read_only=True)
    features = get_training_feature_set(con, entity_ids, pd.Timestamp.now())
    con.close()

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{stage}")
    X = features[FEATURE_COLS].fillna(0)
    features["score"] = model.predict(X)
    return features[["sk_id_curr", "score"]]


if __name__ == "__main__":
    import sys
    ids = [int(x) for x in sys.argv[1:]] or [100002, 100003, 100004]
    print(score_batch(ids))
