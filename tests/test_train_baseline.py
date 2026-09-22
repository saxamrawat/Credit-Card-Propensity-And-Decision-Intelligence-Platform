import numpy as np
import pandas as pd

from propensity_prediction.models.train_baseline import (
    run_baseline_experiment,
)


def create_test_dataset() -> pd.DataFrame:
    """Create a small dataset matching the project schema."""

    rows = []

    for index in range(40):
        default = index % 2

        row = {
            "ID": index + 1,
            "LIMIT_BAL": 10000 + (index * 1000),
            "SEX": 1 if index % 2 == 0 else 2,
            "EDUCATION": (index % 4) + 1,
            "MARRIAGE": (index % 3) + 1,
            "AGE": 25 + (index % 20),
            "PAY_0": default,
            "PAY_2": default,
            "PAY_3": default,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 1000 + index,
            "BILL_AMT2": 900 + index,
            "BILL_AMT3": 800 + index,
            "BILL_AMT4": 700 + index,
            "BILL_AMT5": 600 + index,
            "BILL_AMT6": 500 + index,
            "PAY_AMT1": 100 + index,
            "PAY_AMT2": 100 + index,
            "PAY_AMT3": 100 + index,
            "PAY_AMT4": 100 + index,
            "PAY_AMT5": 100 + index,
            "PAY_AMT6": 100 + index,
            "DEFAULT": default,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def test_run_baseline_experiment():
    df = create_test_dataset()

    results = run_baseline_experiment(df)

    assert set(results.keys()) == {
        "dummy",
        "logistic_regression",
        "decision_tree",
    }


def test_baseline_results_contain_models_and_metrics():
    df = create_test_dataset()

    results = run_baseline_experiment(df)

    for result in results.values():
        assert "model" in result
        assert "metrics" in result

        metrics = result["metrics"]

        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics


def test_baseline_metrics_are_valid():
    df = create_test_dataset()

    results = run_baseline_experiment(df)

    for result in results.values():
        metrics = result["metrics"]

        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

        assert metrics["confusion_matrix"].shape == (2, 2)


def test_baseline_experiment_is_reproducible():
    df = create_test_dataset()

    first = run_baseline_experiment(df)
    second = run_baseline_experiment(df)

    for model_name in first:
        first_metrics = first[model_name]["metrics"]
        second_metrics = second[model_name]["metrics"]

        assert np.isclose(
            first_metrics["roc_auc"],
            second_metrics["roc_auc"],
        )

        assert np.isclose(
            first_metrics["pr_auc"],
            second_metrics["pr_auc"],
        )