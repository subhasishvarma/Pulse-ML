"""
Standalone entry point for re-running just the hyperparameter search step,
e.g. to try a wider param grid without re-running the full train+log flow.
Reuses PARAM_DIST and FEATURE_COLS from train.py as the single source of truth.
"""
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from src.training.train import PARAM_DIST, load_training_data, FEATURE_COLS


def search_best_params(n_iter: int = 30, cv: int = 5):
    df = load_training_data()
    X = df[FEATURE_COLS].fillna(0)
    y = df["target"]

    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )
    search = RandomizedSearchCV(
        model, PARAM_DIST, n_iter=n_iter, scoring="roc_auc", cv=cv, verbose=1, n_jobs=-1, random_state=42
    )
    search.fit(X, y)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV AUC: {search.best_score_:.4f}")
    return search.best_params_


if __name__ == "__main__":
    search_best_params()
