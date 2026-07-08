"""
Generates synthetic data with the same shape as Kaggle's Home Credit Default
Risk dataset (application_train, bureau, previous_application).

v2: the first version of this generator set TARGET from a simple two-feature
linear threshold rule, and scattered application dates independently of every
feature. That gave XGBoost nothing nonlinear to exploit over the logistic
baseline, and gave Evidently no real distribution shift to detect -- both
"wow" moments in the guide had nothing to show. This version fixes that:

1. TARGET depends on interaction/threshold combinations across application,
   bureau, and previous-application features (not just two linear terms) --
   the kind of structure a tree ensemble can split on but a linear model can't.
2. Two raw columns genuinely drift over the 3-year application window (mean
   income rises with inflation; mean bureau delinquency worsens), so an
   early-vs-late split has real covariate shift, not synthetic noise.

Use this ONLY if you don't want to download the real dataset. For the real
project, download from https://www.kaggle.com/c/home-credit-default-risk
and drop the three CSVs into data/raw/ instead -- everything downstream
(schema, features, training) is written against these exact column names,
so it works unchanged either way. Real data also has genuinely richer
nonlinear structure than any hand-built generator (published Home Credit
kernels get XGBoost AUC ~0.74-0.76) -- treat this generator as a sandbox
stand-in, not a permanent substitute.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
N_APPLICANTS = 6000
START_DATE = pd.Timestamp("2023-01-01")
WINDOW_DAYS = 1095  # ~3 years


def _time_fraction(app_date: pd.Series) -> np.ndarray:
    """0.0 at the start of the window, 1.0 at the end -- the drift dial."""
    return ((app_date - START_DATE).dt.days / WINDOW_DAYS).to_numpy()


def generate_applications() -> pd.DataFrame:
    sk_id = np.arange(100_000, 100_000 + N_APPLICANTS)
    app_date = START_DATE + pd.to_timedelta(rng.integers(0, WINDOW_DAYS, size=N_APPLICANTS), unit="D")
    t = _time_fraction(pd.Series(app_date))

    # genuine drift #1: mean income rises ~18% across the window (inflation)
    base_income = rng.lognormal(mean=11.5, sigma=0.5, size=N_APPLICANTS)
    income = base_income * (1 + 0.18 * t)

    credit = income * rng.uniform(1.5, 6.0, size=N_APPLICANTS)
    annuity = credit / rng.uniform(8, 25, size=N_APPLICANTS)
    days_employed = -rng.integers(30, 12000, size=N_APPLICANTS)

    return pd.DataFrame({
        "SK_ID_CURR": sk_id,
        "AMT_INCOME_TOTAL": income.round(2),
        "AMT_CREDIT": credit.round(2),
        "AMT_ANNUITY": annuity.round(2),
        "DAYS_EMPLOYED": days_employed,
        "APPLICATION_DATE": app_date,
        "_time_fraction": t,  # dropped before saving; used to drive bureau drift + target
    })


def generate_bureau(app_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sk_id, t in zip(app_df["SK_ID_CURR"], app_df["_time_fraction"]):
        n_prior = rng.poisson(2.5)
        # genuine drift #2: mean overdue days worsens from ~5 to ~25 across the window
        overdue_mean = 5 + 20 * t
        for _ in range(n_prior):
            rows.append({
                "SK_ID_CURR": sk_id,
                "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], p=[0.35, 0.65]),
                "DAYS_CREDIT_OVERDUE": max(0, rng.normal(overdue_mean, 15)),
                "AMT_CREDIT_SUM": rng.lognormal(10, 0.7),
                "AMT_CREDIT_SUM_DEBT": max(0, rng.lognormal(9, 0.9)),
            })
    return pd.DataFrame(rows)


def generate_previous_application(app_ids: np.ndarray) -> pd.DataFrame:
    rows = []
    for sk_id in app_ids:
        n_prev = rng.poisson(1.8)
        for _ in range(n_prev):
            rows.append({
                "SK_ID_CURR": sk_id,
                "NAME_CONTRACT_STATUS": rng.choice(
                    ["Approved", "Refused", "Cancelled"], p=[0.7, 0.2, 0.1]
                ),
                "AMT_APPLICATION": rng.lognormal(10.5, 0.6),
                "AMT_CREDIT": rng.lognormal(10.4, 0.6),
            })
    return pd.DataFrame(rows)


def assign_targets(app_df: pd.DataFrame, bureau_df: pd.DataFrame, prev_df: pd.DataFrame) -> np.ndarray:
    """
    TARGET is a mix of a linear-findable component (so a logistic baseline
    has real signal too) and XOR-style interaction terms that have ~zero
    marginal linear correlation with either input alone -- a linear model
    structurally cannot fit these, only a tree ensemble that can split on
    combinations can. This is what gives XGBoost genuine room to beat the
    baseline (a plain additive/threshold rule doesn't -- a linear model can
    partly reconstruct it from the individual thresholds).
    """
    credit_income_ratio = app_df["AMT_CREDIT"] / app_df["AMT_INCOME_TOTAL"]
    annuity_income_ratio = app_df["AMT_ANNUITY"] / app_df["AMT_INCOME_TOTAL"]

    bureau_agg = bureau_df.groupby("SK_ID_CURR").agg(
        num_active_loans=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
        avg_overdue=("DAYS_CREDIT_OVERDUE", "mean"),
    )
    prev_agg = prev_df.groupby("SK_ID_CURR").agg(
        approval_rate=("NAME_CONTRACT_STATUS", lambda s: (s == "Approved").mean()),
    )

    merged = app_df[["SK_ID_CURR"]].merge(bureau_agg, on="SK_ID_CURR", how="left") \
                                     .merge(prev_agg, on="SK_ID_CURR", how="left")
    merged["num_active_loans"] = merged["num_active_loans"].fillna(0)
    merged["avg_overdue"] = merged["avg_overdue"].fillna(0)
    merged["approval_rate"] = merged["approval_rate"].fillna(0.5)  # no history -> neutral prior

    cir_z = (credit_income_ratio - credit_income_ratio.mean()) / credit_income_ratio.std()
    overdue_z = (merged["avg_overdue"] - merged["avg_overdue"].mean()) / merged["avg_overdue"].std()

    # linear-findable component: a logistic model can fit this part directly
    linear_component = 0.55 * cir_z + 0.35 * overdue_z - 1.1 * merged["approval_rate"]

    # XOR interactions: high risk when exactly one condition holds, not when both
    # or neither do -- each input alone carries ~no linear signal about the target
    a = credit_income_ratio > credit_income_ratio.median()
    b = merged["num_active_loans"] >= 2
    xor1 = (a != b).astype(float)

    c = merged["avg_overdue"] > 15
    d = annuity_income_ratio > annuity_income_ratio.median()
    xor2 = (c != d).astype(float)

    risk_score = linear_component + 1.4 * xor1 + 1.1 * xor2

    noise = rng.normal(0, 0.5, size=len(app_df))
    logit = -3.6 + risk_score.to_numpy() + noise
    prob = 1 / (1 + np.exp(-logit))
    return rng.binomial(1, prob)


if __name__ == "__main__":
    app_df = generate_applications()
    bureau_df = generate_bureau(app_df)
    prev_df = generate_previous_application(app_df["SK_ID_CURR"].to_numpy())

    target = assign_targets(app_df, bureau_df, prev_df)
    app_df["TARGET"] = target
    app_out = app_df.drop(columns=["_time_fraction"])[
        ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
         "DAYS_EMPLOYED", "APPLICATION_DATE"]
    ]

    app_out.to_csv(RAW_DIR / "application_train.csv", index=False)
    bureau_df.to_csv(RAW_DIR / "bureau.csv", index=False)
    prev_df.to_csv(RAW_DIR / "previous_application.csv", index=False)

    print(f"application_train: {len(app_out)} rows, target rate {app_out['TARGET'].mean():.3%}")
    print(f"bureau: {len(bureau_df)} rows")
    print(f"previous_application: {len(prev_df)} rows")
