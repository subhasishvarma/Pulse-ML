"""
Trains a logistic-regression baseline, then an XGBoost model with randomized
hyperparameter search, and logs both to MLflow so every run is comparable.
"""
import duckdb
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.point_in_time import get_training_feature_set

FEATURE_COLS = [
    "amt_income_total", "amt_credit", "amt_annuity", "days_employed",
    "credit_income_ratio", "annuity_income_ratio",
    "num_prior_loans", "num_active_loans", "avg_days_credit_overdue", "total_debt_to_credit_ratio",
    "num_previous_applications", "num_approved", "num_refused", "approval_rate", "avg_amt_application",
]

PARAM_DIST = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [200, 400, 600, 800],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


def load_training_data() -> pd.DataFrame:
    con = duckdb.connect("data/feature_store.duckdb", read_only=True)
    entities = con.execute("SELECT sk_id_curr, target FROM entities").df()
    features = get_training_feature_set(con, entities["sk_id_curr"].tolist(), pd.Timestamp.now())
    con.close()
    return features.merge(entities, on="sk_id_curr", how="inner")


def run():
    df = load_training_data()
    X = df[FEATURE_COLS].fillna(0)
    y = df["target"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # class imbalance: Home Credit-style data is ~8-10% positive class
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    baseline.fit(X_train_scaled, y_train)
    baseline_auc = roc_auc_score(y_val, baseline.predict_proba(X_val_scaled)[:, 1])
    print(f"Baseline AUC: {baseline_auc:.4f}")

    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )
    search = RandomizedSearchCV(
        model, PARAM_DIST, n_iter=30, scoring="roc_auc", cv=5, verbose=1, n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    val_auc = roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])
    val_auc_pr = average_precision_score(y_val, best_model.predict_proba(X_val)[:, 1])
    print(f"XGBoost validation AUC: {val_auc:.4f} (baseline: {baseline_auc:.4f})")
    print(f"XGBoost validation AUC-PR: {val_auc_pr:.4f}")  # accuracy alone is misleading on imbalanced data

    mlflow.set_experiment("pulseml-credit-risk")
    with mlflow.start_run(run_name="xgb_randomsearch_v1"):
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("val_auc_pr", val_auc_pr)
        mlflow.log_metric("baseline_auc", baseline_auc)
        mlflow.xgboost.log_model(best_model, "model")
        run_id = mlflow.active_run().info.run_id

    print(f"Logged MLflow run: {run_id}")
    return run_id, best_model, val_auc, baseline_auc


if __name__ == "__main__":
    run()
