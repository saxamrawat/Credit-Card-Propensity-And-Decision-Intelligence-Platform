import numpy as np
import pandas as pd
import pytest

from propensity_prediction.models.evaluation import (
    calculate_improvement_over_dummy,
    compare_model_results,
    get_confusion_matrices,
)


def create_results():
    return {
        "dummy": {
            "metrics": {
                "roc_auc": 0.50,
                "pr_auc": 0.22,
                "accuracy": 0.78,
                "precision": 0.00,
                "recall": 0.00,
                "f1": 0.00,
                "confusion_matrix": np.array(
                    [[78, 0], [22, 0]]
                ),
            }
        },
        "logistic_regression": {
            "metrics": {
                "roc_auc": 0.75,
                "pr_auc": 0.45,
                "accuracy": 0.80,
                "precision": 0.55,
                "recall": 0.40,
                "f1": 0.46,
                "confusion_matrix": np.array(
                    [[70, 8], [13, 9]]
                ),
            }
        },
    }


def test_compare_model_results():
    results = create_results()

    comparison = compare_model_results(results)

    assert isinstance(comparison, pd.DataFrame)

    assert list(comparison.index) == [
        "dummy",
        "logistic_regression",
    ]

    for metric in [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        assert metric in comparison.columns


def test_compare_model_results_values():
    results = create_results()

    comparison = compare_model_results(results)

    assert comparison.loc[
        "logistic_regression", "roc_auc"
    ] == pytest.approx(0.75)

    assert comparison.loc[
        "logistic_regression", "pr_auc"
    ] == pytest.approx(0.45)


def test_calculate_improvement_over_dummy():
    results = create_results()

    comparison = compare_model_results(results)

    improvements = calculate_improvement_over_dummy(comparison)

    assert improvements.loc[
        "logistic_regression", "roc_auc"
    ] == pytest.approx(0.25)

    assert improvements.loc[
        "logistic_regression", "pr_auc"
    ] == pytest.approx(0.23)


def test_calculate_improvement_requires_dummy():
    results = create_results()

    comparison = compare_model_results(results)

    comparison = comparison.drop(index="dummy")

    with pytest.raises(ValueError):
        calculate_improvement_over_dummy(comparison)


def test_get_confusion_matrices():
    results = create_results()

    matrices = get_confusion_matrices(results)

    assert set(matrices.keys()) == {
        "dummy",
        "logistic_regression",
    }

    assert matrices["dummy"].shape == (2, 2)
    assert matrices["logistic_regression"].shape == (2, 2)