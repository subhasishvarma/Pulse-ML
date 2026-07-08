"""
Given entity IDs + an as-of timestamp, returns the most recent feature row
per entity at or before that timestamp -- never after. This is exactly the
bug feature stores exist to prevent: training on information from the future.
"""
import duckdb
import pandas as pd


def get_features_as_of(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    entity_ids: list[int],
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    query = f"""
        SELECT f.*
        FROM {table_name} f
        INNER JOIN (
            SELECT sk_id_curr, MAX(feature_timestamp) AS max_ts
            FROM {table_name}
            WHERE sk_id_curr IN ({','.join(map(str, entity_ids))})
              AND feature_timestamp <= ?
            GROUP BY sk_id_curr
        ) latest
        ON f.sk_id_curr = latest.sk_id_curr AND f.feature_timestamp = latest.max_ts
    """
    return con.execute(query, [as_of]).df()


def get_training_feature_set(
    con: duckdb.DuckDBPyConnection,
    entity_ids: list[int],
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Joins all three feature tables for a point-in-time-correct training set."""
    app = get_features_as_of(con, "features_application", entity_ids, as_of)
    bureau = get_features_as_of(con, "features_bureau_history", entity_ids, as_of)
    prev = get_features_as_of(con, "features_previous_loans", entity_ids, as_of)

    merged = app.merge(
        bureau.drop(columns=["feature_timestamp"]), on="sk_id_curr", how="left"
    ).merge(
        prev.drop(columns=["feature_timestamp"]), on="sk_id_curr", how="left"
    )
    return merged
