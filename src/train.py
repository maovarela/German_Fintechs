"""
Train the model and log everything to MLflow (file-based, no remote server).

Hyperparameters can be overridden via env vars:
  N_ESTIMATORS  (default: 200)
  MAX_DEPTH     (default: None = unlimited)
"""
import os

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.model import build_model
from src.preprocess import load_and_prepare

TRACKING_URI = "file:///app/mlruns"
EXPERIMENT_NAME = "german_fintech_classifier"

N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "200"))
MAX_DEPTH_STR = os.environ.get("MAX_DEPTH", "None")
MAX_DEPTH = None if MAX_DEPTH_STR == "None" else int(MAX_DEPTH_STR)
RANDOM_STATE = 42
TEST_SIZE = 0.2


def train() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[train] Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    model = build_model(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )

    with mlflow.start_run() as run:
        print(f"[train] MLflow run ID: {run.info.run_id}")

        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "max_depth": str(MAX_DEPTH),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        })

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metrics({"roc_auc": roc_auc, "accuracy": accuracy})
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:3],
        )

        print(f"\n[train] Classification report:\n{classification_report(y_test, y_pred)}")
        print(f"[train] ROC-AUC: {roc_auc:.4f} | Accuracy: {accuracy:.4f}")
        print(f"[train] Model logged to run: {run.info.run_id}")


if __name__ == "__main__":
    train()
