"""Cross-validation utilities for baseline models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from propensity_prediction.data.preprocessing import create_preprocessor
from propensity_prediction.models.baseline import get_baseline_models


DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42


def create_model_pipeline(model: Any) -> Pipeline:
    """Create a preprocessing + model pipeline.

    The preprocessor is fitted independently inside each CV fold.
    """
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", clone(model)),
        ]
    )


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Perform stratified cross-validation for one model."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    roc_auc_scores = []
    pr_auc_scores = []

    for train_indices, validation_indices in cv.split(X, y):
        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]

        y_train = y.iloc[train_indices]
        y_validation = y.iloc[validation_indices]

        pipeline = create_model_pipeline(model)

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_validation)[:, 1]

        roc_auc_scores.append(
            roc_auc_score(y_validation, probabilities)
        )

        pr_auc_scores.append(
            average_precision_score(
                y_validation,
                probabilities,
            )
        )

    return {
        "roc_auc_scores": roc_auc_scores,
        "pr_auc_scores": pr_auc_scores,
        "roc_auc_mean": float(np.mean(roc_auc_scores)),
        "roc_auc_std": float(np.std(roc_auc_scores)),
        "pr_auc_mean": float(np.mean(pr_auc_scores)),
        "pr_auc_std": float(np.std(pr_auc_scores)),
    }


def cross_validate_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, dict[str, Any]]:
    """Run cross-validation for all baseline models."""
    models = get_baseline_models()

    results = {}

    for model_name, model in models.items():
        results[model_name] = cross_validate_model(
            model=model,
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
        )

    return results


def summarize_cross_validation_results(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Convert cross-validation results into a summary DataFrame."""
    rows = []

    for model_name, result in results.items():
        rows.append(
            {
                "model": model_name,
                "roc_auc_mean": result["roc_auc_mean"],
                "roc_auc_std": result["roc_auc_std"],
                "pr_auc_mean": result["pr_auc_mean"],
                "pr_auc_std": result["pr_auc_std"],
            }
        )

    return pd.DataFrame(rows).set_index("model")