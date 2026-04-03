"""
Model factory. Returns a sklearn Pipeline.

Wrapping in a Pipeline ensures preprocessing steps and the estimator
are serialized together by MLflow as a single artifact.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def build_model(
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Pipeline:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",  # handles class imbalance
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("clf", clf)])
