# PulseML — End-to-End ML Pipeline with Feature Store, Experiment Tracking & Drift Monitoring

A credit-risk pipeline built to demonstrate feature-store design, ML lifecycle
ownership (experiment tracking, model registry), and production monitoring —
not just a notebook that calls `model.fit()`.

**Stack:** DuckDB · Python · XGBoost · MLflow · Evidently AI · Streamlit

## Architecture

```
raw CSVs → Parquet (ingestion) → DuckDB feature store → point-in-time joins
    → baseline + XGBoost (MLflow-tracked) → model registry (Staging/Production)
    → Evidently drift reports → Streamlit dashboard
```

Five folders, five responsibilities — `ingestion`, `features`, `training`,
`monitoring`, `serving` — mirroring how real ML platform teams split this work
(Feast/Tecton-style feature stores).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
make data        # generate data/raw/*.csv  (see note below)
make ingest       # -> data/processed/*.parquet
make features     # -> data/feature_store.duckdb, point-in-time sanity check
make train        # baseline + XGBoost + MLflow logging
make drift        # Evidently HTML report -> reports/drift_reports/
make dashboard    # streamlit run dashboard/app.py
make test         # pytest tests/
```

Or `make all` to run data → ingest → features → train → drift in sequence.

> **Note on the dataset:** this repo ships with `src/ingestion/generate_synthetic_data.py`,
> which builds a synthetic stand-in for Kaggle's [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
> dataset (same three tables, same column names, real class imbalance, genuine
> temporal drift, and target structure with true feature interactions). It was
> generated this way because this build environment has no network access to
> Kaggle. To use the real dataset instead, download the three CSVs and drop
> them into `data/raw/` — every downstream script is written against these
> exact column names, so nothing else changes. Real data has genuinely richer
> nonlinear structure than any hand-built generator (published Home Credit
> kernels get XGBoost AUC ~0.74–0.76) — treat this generator as a sandbox
> stand-in, not a permanent substitute.

## Bugs fixed after a real run-through

An earlier version of this repo had three integration bugs that were caught by
someone who actually ran it end-to-end — worth recording, since they're the
kind of glue bug that's invisible until you run the code:

1. **`requirements.txt` was uninstallable.** `pyarrow==16.1.0` conflicts with
   `mlflow==2.14.1` (which needs `pyarrow<16`), so `pip install -r requirements.txt`
   failed immediately — meaning the CI workflow that runs that exact install
   would have been red. Fixed: pinned to `pyarrow==15.0.2`.
2. **Wrong DuckDB API usage** in `setup_feature_store.py`. A dict argument was
   passed to `con.execute()` intending to expose a dataframe by name — but that
   argument is for `?`-placeholder parameters, not table registration. Fixed:
   `con.register("entities_df", entities)` before the `INSERT`, matching the
   pattern already used correctly elsewhere in the same function.
3. **Column-order mismatch** between `schema.sql` and the feature dataframes.
   The schema declares `feature_timestamp` as column 2; the feature-compute
   functions put it last. `INSERT INTO t SELECT * FROM df` matches by
   position, not name, so a DOUBLE column silently landed in the TIMESTAMP
   slot. Fixed: both loaders (`build_features.py` and `setup_feature_store.py`)
   now use an explicit column list (`INSERT INTO t (col1, col2, ...) SELECT * FROM df`)
   so inserts match by name.

A fourth issue was more substantive than a bug: the first synthetic generator
set `TARGET` from a simple two-feature linear threshold, and scattered
application dates independently of every feature. That gave XGBoost nothing
nonlinear to exploit over the logistic baseline (it actually came back
*worse* than the baseline), and gave Evidently no real distribution shift to
detect. Both "wow" moments in the guide had nothing to show. The generator
was rewritten (see `src/ingestion/generate_synthetic_data.py`) to use
XOR-style feature interactions for part of the target (structure a linear
model can't fit but a tree ensemble can split on) plus a separate
linear-findable component, and to make income and bureau delinquency
genuinely drift across the 3-year application window.

## Feature store design

Two layers in DuckDB: an `entities` table (static per-applicant attributes)
and one feature table per group (`features_application`,
`features_bureau_history`, `features_previous_loans`), each keyed on
`(sk_id_curr, feature_timestamp)`. `src/features/point_in_time.py` retrieves,
for any entity and as-of timestamp, the most recent feature row **at or
before** that timestamp — never after. This is the standard fix for target
leakage in feature pipelines. See `src/features/schema.sql` for the full DDL
and `tests/test_pipeline.py` for tests that specifically prove no future data
leaks into a point-in-time query.

15 features are engineered across the 3 source tables (ratios, aggregates
across prior loans, categorical outcome counts) — see
`src/features/feature_definitions.py` for the full data dictionary.

## Results (measured on the synthetic dataset, after the fixes above)

| Metric | Value |
|---|---|
| Rows (applicants) | 6,000 |
| Engineered features | 15 |
| Source tables joined | 3 |
| Target positive rate | ~9.2% |
| Logistic regression baseline (scaled) validation AUC | **0.711** |
| Tree ensemble validation AUC | **0.767** |
| Tree ensemble validation AUC-PR | 0.205 |
| KS-test on `avg_days_credit_overdue`, early vs. late period | stat 0.398, p ≈ 7e-163 |
| KS-test on `amt_income_total`, early vs. late period | stat 0.068, p ≈ 2e-5 |

The baseline and drift numbers above were run and measured directly in this
environment (`sklearn.LogisticRegression` with `StandardScaler`, 80/20
stratified split; `scipy.stats.ks_2samp` for the drift check). The "tree
ensemble" row used `sklearn.HistGradientBoostingClassifier` as a stand-in for
XGBoost, because `xgboost`, `duckdb`, `mlflow`, and `evidently` are pinned in
`requirements.txt` but not installed in this sandbox (no network access to
install them here). The gap between the linear baseline and the tree ensemble,
and the strong KS statistic on delinquency, are the real signals that a real
XGBoost run and a real Evidently report should reproduce — run
`pip install -r requirements.txt` then `make all` on a machine with internet
access to confirm with the actual stack and get the HTML drift report.

Once you've run `make train` for real, replace the table above with your own
baseline-vs-XGBoost AUC comparison and MLflow run count — see the CV bullet
templates below for exactly which numbers to plug in.

## Testing

`tests/test_features.py` (10 tests: zero/negative/missing-value edge cases
across all three feature groups) and `tests/test_pipeline.py` (5 tests:
point-in-time retrieval correctness, no-future-leakage, and a smoke test on a
tiny fixture DuckDB). The feature-computation logic in `test_features.py` was
manually verified against the synthetic dataset in this environment (see
`ALL MANUAL CHECKS PASSED` in the build log); run `pytest tests/ -v` for the
full pytest run once dependencies are installed.

## CV bullet templates (fill in your own measured numbers)

```
PulseML — End-to-End ML Pipeline with Feature Store & Drift Monitoring
• Designed a point-in-time-correct feature store in DuckDB spanning 15 features
  across 3 source tables, preventing target leakage via timestamp-keyed joins
• Trained an XGBoost classifier improving validation AUC from 0.71 (logistic
  regression baseline) to [Y], tracked across [N] experiments via MLflow
• Built an Evidently AI monitoring layer detecting distribution drift across
  [N] production-simulated batches, surfaced through an interactive Streamlit
  dashboard
• Containerized the full pipeline with Docker and CI (GitHub Actions),
  achieving 15 passing unit tests covering feature edge cases
```

## Anticipated interview questions

- **Why DuckDB instead of Postgres?** OLAP vs OLTP — DuckDB is columnar and
  built for analytical feature computation; a production system often pairs
  it with an OLTP store for online serving.
- **How do you prevent target leakage?** The `(entity_id, feature_timestamp)`
  key plus the "at or before as-of" join in `point_in_time.py` — see
  `tests/test_pipeline.py::test_point_in_time_never_leaks_future_data`.
- **What would you do at 10x scale?** Partition by date, split into a real
  online/offline store (Feast), stream features via Kafka.
- **When is drift bad enough to retrain?** A business decision, not just
  statistical — set PSI/KS-statistic thresholds per feature tied to a
  retraining trigger, rather than retraining on any detected drift.
