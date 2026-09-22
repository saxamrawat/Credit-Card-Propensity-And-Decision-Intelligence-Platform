"""End-to-end training and evaluation of baseline models."""

from __future__ import annotations

from typing import Any

import pandas as pd

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.data.preprocessing import (
    create_preprocessor,
)
from propensity_prediction.data.split import split_dataset
from propensity_prediction.models.baseline import (
    evaluate_model,
    get_baseline_models,
    train_model,
)


def run_baseline_experiment(
    df: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate all baseline models.

    The test set is intentionally not used. Models are trained on the
    training set and evaluated on the validation set.
    """
    if df is None:
        df = build_processed_dataset()

    dataset_split = split_dataset(df)

    X_train = dataset_split.X_train
    X_validation = dataset_split.X_validation

    y_train = dataset_split.y_train
    y_validation = dataset_split.y_validation

    preprocessor = create_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)

    X_validation_transformed = preprocessor.transform(X_validation)

    models = get_baseline_models()

    results: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        trained_model = train_model(
            model,
            X_train_transformed,
            y_train,
        )

        metrics = evaluate_model(
            trained_model,
            X_validation_transformed,
            y_validation,
        )

        results[model_name] = {
            "model": trained_model,
            "metrics": metrics,
        }

    return results


def print_baseline_results(
    results: dict[str, dict[str, Any]],
) -> None:
    """Print a compact summary of baseline validation results."""

    print("\nBaseline Validation Results")
    print("=" * 70)

    for model_name, result in results.items():
        metrics = result["metrics"]

        print(f"\n{model_name}")
        print("-" * 70)
        print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:   {metrics['pr_auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision:{metrics['precision']:.4f}")
        print(f"Recall:   {metrics['recall']:.4f}")
        print(f"F1:       {metrics['f1']:.4f}")


if __name__ == "__main__":
    baseline_results = run_baseline_experiment()
    print_baseline_results(baseline_results)