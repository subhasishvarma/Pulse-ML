import duckdb
import pandas as pd
import pytest

from src.features.point_in_time import get_features_as_of, get_training_feature_set


@pytest.fixture
def tiny_store(tmp_path):
    """A tiny in-memory-like DuckDB with two entities, each with two feature
    snapshots at different timestamps -- enough to prove point-in-time logic."""
    db_path = str(tmp_path / "test_store.duckdb")
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE features_application (
            sk_id_curr BIGINT, feature_timestamp TIMESTAMP,
            amt_income_total DOUBLE, amt_credit DOUBLE, amt_annuity DOUBLE,
            days_employed INTEGER, credit_income_ratio DOUBLE, annuity_income_ratio DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO features_application VALUES
        (1, '2023-01-01 00:00:00', 50000, 100000, 5000, -500, 2.0, 0.1),
        (1, '2023-06-01 00:00:00', 55000, 100000, 5000, -650, 1.82, 0.09),
        (2, '2023-03-01 00:00:00', 70000, 200000, 8000, -1000, 2.86, 0.11)
    """)
    con.execute("""
        CREATE TABLE features_bureau_history (
            sk_id_curr BIGINT, feature_timestamp TIMESTAMP,
            num_prior_loans INTEGER, num_active_loans INTEGER,
            avg_days_credit_overdue DOUBLE, total_debt_to_credit_ratio DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE features_previous_loans (
            sk_id_curr BIGINT, feature_timestamp TIMESTAMP,
            num_previous_applications INTEGER, num_approved INTEGER,
            num_refused INTEGER, approval_rate DOUBLE, avg_amt_application DOUBLE
        )
    """)
    yield con
    con.close()


def test_point_in_time_returns_only_row_at_or_before_asof(tiny_store):
    result = get_features_as_of(
        tiny_store, "features_application", [1], pd.Timestamp("2023-03-01")
    )
    assert len(result) == 1
    assert result["amt_income_total"].iloc[0] == 50000  # the Jan snapshot, not June


def test_point_in_time_never_leaks_future_data(tiny_store):
    result = get_features_as_of(
        tiny_store, "features_application", [1], pd.Timestamp("2023-02-01")
    )
    # only the Jan snapshot exists before Feb -- June snapshot must never appear
    assert (result["feature_timestamp"] < pd.Timestamp("2023-02-01")).all()


def test_point_in_time_picks_latest_available_snapshot(tiny_store):
    result = get_features_as_of(
        tiny_store, "features_application", [1], pd.Timestamp("2023-12-31")
    )
    assert result["amt_income_total"].iloc[0] == 55000  # June snapshot is latest before year-end


def test_point_in_time_multiple_entities_independent(tiny_store):
    result = get_features_as_of(
        tiny_store, "features_application", [1, 2], pd.Timestamp("2023-12-31")
    )
    assert set(result["sk_id_curr"]) == {1, 2}
    assert len(result) == 2  # one row per entity, not one per snapshot


def test_smoke_full_feature_join_on_tiny_fixture(tiny_store):
    result = get_training_feature_set(tiny_store, [1, 2], pd.Timestamp("2023-12-31"))
    assert len(result) == 2
    assert "amt_income_total" in result.columns
    assert "num_prior_loans" in result.columns  # left-joined, may be null but column must exist
