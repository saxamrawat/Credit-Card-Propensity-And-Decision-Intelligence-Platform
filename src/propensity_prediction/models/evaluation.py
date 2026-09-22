"""Utilities for comparing baseline model results."""

from __future__ import annotations

from typing import Any

import pandas as pd


METRIC_COLUMNS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


def compare_model_results(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Convert model evaluation results into a comparison DataFrame."""
    rows = []

    for model_name, result in results.items():
        metrics = result["metrics"]

        row = {
            "model": model_name,
        }

        for metric in METRIC_COLUMNS:
            row[metric] = metrics[metric]

        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def calculate_improvement_over_dummy(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metric differences relative to the dummy baseline."""
    if "dummy" not in comparison.index:
        raise ValueError("Comparison must contain a dummy baseline.")

    dummy_metrics = comparison.loc["dummy"]

    improvements = comparison[METRIC_COLUMNS].subtract(
        dummy_metrics[METRIC_COLUMNS]
    )

    return improvements


def get_confusion_matrices(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return confusion matrices for all evaluated models."""
    return {
        model_name: result["metrics"]["confusion_matrix"]
        for model_name, result in results.items()
    }