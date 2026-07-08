"""
Registers a trained model and moves it through lifecycle stages. Demonstrates
that "best on validation set" is not the same claim as "safe to deploy" --
promotion is a deliberate, logged decision, not automatic.
"""
from mlflow import MlflowClient
import mlflow

MODEL_NAME = "pulseml-credit-risk-classifier"


def register(run_id: str) -> int:
    result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)
    print(f"Registered {MODEL_NAME} version {result.version} from run {run_id}")
    return result.version


def promote_to_staging(version: int):
    client = MlflowClient()
    client.transition_model_version_stage(name=MODEL_NAME, version=version, stage="Staging")
    print(f"{MODEL_NAME} v{version} -> Staging")


def promote_to_production(version: int):
    """Only call this after checking the Staging model against the drift/eval
    checks in Phase 3 -- promotion should be a deliberate decision, not a rubber stamp."""
    client = MlflowClient()
    client.transition_model_version_stage(name=MODEL_NAME, version=version, stage="Production")
    print(f"{MODEL_NAME} v{version} -> Production")


if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not run_id:
        raise SystemExit("Usage: python -m src.training.registry <run_id>")
    version = register(run_id)
    promote_to_staging(version)
