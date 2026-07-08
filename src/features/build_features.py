"""
Pure functions: raw dataframe in, feature dataframe out. Loading into DuckDB
is a separate step (load_features_to_store) so feature logic can be unit
tested without touching a database at all.
"""
import pandas as pd

FEATURE_TIMESTAMP = pd.Timestamp.now()


def compute_application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED"]].copy()
    safe_income = df["AMT_INCOME_TOTAL"].replace(0, pd.NA)
    out["credit_income_ratio"] = df["AMT_CREDIT"] / safe_income
    out["annuity_income_ratio"] = df["AMT_ANNUITY"] / safe_income
    out["feature_timestamp"] = FEATURE_TIMESTAMP
    return out.rename(columns={
        "SK_ID_CURR": "sk_id_curr",
        "AMT_INCOME_TOTAL": "amt_income_total",
        "AMT_CREDIT": "amt_credit",
        "AMT_ANNUITY": "amt_annuity",
        "DAYS_EMPLOYED": "days_employed",
    })


def compute_bureau_features(bureau_df: pd.DataFrame) -> pd.DataFrame:
    grouped = bureau_df.groupby("SK_ID_CURR")
    out = grouped.agg(
        num_prior_loans=("SK_ID_CURR", "count"),
        num_active_loans=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
        avg_days_credit_overdue=("DAYS_CREDIT_OVERDUE", "mean"),
        sum_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        sum_credit=("AMT_CREDIT_SUM", "sum"),
    ).reset_index()

    safe_credit = out["sum_credit"].replace(0, pd.NA)
    out["total_debt_to_credit_ratio"] = out["sum_debt"] / safe_credit
    out["feature_timestamp"] = FEATURE_TIMESTAMP
    return out.drop(columns=["sum_debt", "sum_credit"]).rename(columns={"SK_ID_CURR": "sk_id_curr"})


def compute_previous_loan_features(prev_df: pd.DataFrame) -> pd.DataFrame:
    grouped = prev_df.groupby("SK_ID_CURR")
    out = grouped.agg(
        num_previous_applications=("SK_ID_CURR", "count"),
        num_approved=("NAME_CONTRACT_STATUS", lambda s: (s == "Approved").sum()),
        num_refused=("NAME_CONTRACT_STATUS", lambda s: (s == "Refused").sum()),
        avg_amt_application=("AMT_APPLICATION", "mean"),
    ).reset_index()

    out["approval_rate"] = out["num_approved"] / out["num_previous_applications"].replace(0, pd.NA)
    out["feature_timestamp"] = FEATURE_TIMESTAMP
    return out.rename(columns={"SK_ID_CURR": "sk_id_curr"})


def load_features_to_store(df: pd.DataFrame, table_name: str, db_path: str = "data/feature_store.duckdb"):
    import duckdb  # imported lazily so feature-compute functions stay testable without duckdb installed
    con = duckdb.connect(db_path)
    con.register("feats_df", df)
    # explicit column list: INSERT ... SELECT * matches by position, and schema.sql declares
    # feature_timestamp as column 2 while these dataframes put it last -- naming it explicitly
    # avoids silently inserting a DOUBLE into a TIMESTAMP column.
    con.execute(f"INSERT INTO {table_name} ({', '.join(df.columns)}) SELECT * FROM feats_df")
    con.close()


if __name__ == "__main__":
    app_df = pd.read_csv("data/raw/application_train.csv")
    bureau_df = pd.read_csv("data/raw/bureau.csv")
    prev_df = pd.read_csv("data/raw/previous_application.csv")

    app_features = compute_application_features(app_df)
    bureau_features = compute_bureau_features(bureau_df)
    prev_features = compute_previous_loan_features(prev_df)

    print(f"features_application: {len(app_features)} rows, {app_features.shape[1]} cols")
    print(f"features_bureau_history: {len(bureau_features)} rows, {bureau_features.shape[1]} cols")
    print(f"features_previous_loans: {len(prev_features)} rows, {prev_features.shape[1]} cols")
