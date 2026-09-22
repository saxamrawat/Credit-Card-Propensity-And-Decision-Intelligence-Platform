import numpy as np
import pandas as pd

from propensity_prediction.models.baseline import (
    create_logistic_regression_model,
)
from propensity_prediction.models.cross_validation import (
    create_model_pipeline,
    cross_validate_baseline_models,
    cross_validate_model,
    summarize_cross_validation_results,
)


def create_test_dataset() -> pd.DataFrame:
    """Create a synthetic dataset matching the project schema."""

    rows = []

    for index in range(100):
        default = index % 2

        rows.append(
            {
                "ID": index + 1,
                "LIMIT_BAL": 10000 + index * 1000,
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
        )

    return pd.DataFrame(rows)


def test_create_model_pipeline():
    model = create_logistic_regression_model()

    pipeline = create_model_pipeline(model)

    assert list(pipeline.named_steps.keys()) == [
        "preprocessor",
        "model",
    ]


def test_cross_validate_model():
    df = create_test_dataset()

    X = df.drop(columns=["DEFAULT"])
    y = df["DEFAULT"]

    model = create_logistic_regression_model()

    results = cross_validate_model(
        model=model,
        X=X,
        y=y,
        n_splits=5,
    )

    assert len(results["roc_auc_scores"]) == 5
    assert len(results["pr_auc_scores"]) == 5

    assert 0 <= results["roc_auc_mean"] <= 1
    assert 0 <= results["pr_auc_mean"] <= 1

    assert results["roc_auc_std"] >= 0
    assert results["pr_auc_std"] >= 0


def test_cross_validate_baseline_models():
    df = create_test_dataset()

    X = df.drop(columns=["DEFAULT"])
    y = df["DEFAULT"]

    results = cross_validate_baseline_models(
        X=X,
        y=y,
        n_splits=5,
    )

    assert set(results.keys()) == {
        "dummy",
        "logistic_regression",
        "decision_tree",
    }

    for result in results.values():
        assert len(result["roc_auc_scores"]) == 5
        assert len(result["pr_auc_scores"]) == 5


def test_summarize_cross_validation_results():
    df = create_test_dataset()

    X = df.drop(columns=["DEFAULT"])
    y = df["DEFAULT"]

    results = cross_validate_baseline_models(
        X=X,
        y=y,
        n_splits=5,
    )

    summary = summarize_cross_validation_results(results)

    assert summary.shape == (3, 4)

    assert "roc_auc_mean" in summary.columns
    assert "roc_auc_std" in summary.columns
    assert "pr_auc_mean" in summary.columns
    assert "pr_auc_std" in summary.columns


def test_cross_validation_is_reproducible():
    df = create_test_dataset()

    X = df.drop(columns=["DEFAULT"])
    y = df["DEFAULT"]

    model = create_logistic_regression_model()

    first = cross_validate_model(
        model=model,
        X=X,
        y=y,
        n_splits=5,
        random_state=42,
    )

    second = cross_validate_model(
        model=model,
        X=X,
        y=y,
        n_splits=5,
        random_state=42,
    )

    assert np.allclose(
        first["roc_auc_scores"],
        second["roc_auc_scores"],
    )

    assert np.allclose(
        first["pr_auc_scores"],
        second["pr_auc_scores"],
    )