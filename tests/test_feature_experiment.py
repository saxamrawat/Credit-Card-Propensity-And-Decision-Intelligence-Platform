import numpy as np
import pandas as pd
import pytest

from propensity_prediction.models.feature_experiment import (
    ENGINEERED_FEATURES,
    compare_feature_experiments,
    run_feature_experiment,
)


def make_sample_data(n: int = 200) -> pd.DataFrame:
    """Create a small dataset with the columns required by the pipeline."""
    rng = np.random.default_rng(42)

    data = {
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": rng.integers(10000, 500000, size=n),
        "SEX": rng.choice([1, 2], size=n),
        "EDUCATION": rng.choice([0, 1, 2, 3, 4], size=n),
        "MARRIAGE": rng.choice([0, 1, 2, 3], size=n),
        "AGE": rng.integers(21, 70, size=n),
        "DEFAULT": np.tile([0, 1], n // 2),
    }

    for column in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        data[column] = rng.integers(-2, 9, size=n)

    for column in [
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    ]:
        data[column] = rng.integers(-10000, 200000, size=n)

    for column in [
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]:
        data[column] = rng.integers(0, 50000, size=n)

    return pd.DataFrame(data)


def test_engineered_feature_list_contains_20_features():
    assert len(ENGINEERED_FEATURES) == 20
    assert len(set(ENGINEERED_FEATURES)) == 20


def test_feature_experiment_returns_both_experiments():
    results = run_feature_experiment(make_sample_data())

    assert list(results.index) == [
        "baseline",
        "engineered_features",
    ]


def test_feature_experiment_returns_expected_metrics():
    results = run_feature_experiment(make_sample_data())

    expected_metrics = {
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }

    assert set(results.columns) == expected_metrics
    assert np.isfinite(results.to_numpy()).all()


def test_comparison_calculates_change_from_baseline():
    results = pd.DataFrame(
        {
            "roc_auc": [0.70, 0.75],
            "pr_auc": [0.40, 0.45],
        },
        index=["baseline", "engineered_features"],
    )

    comparison = compare_feature_experiments(results)

    assert comparison.loc["change_vs_baseline", "roc_auc"] == pytest.approx(0.05)
    assert comparison.loc["change_vs_baseline", "pr_auc"] == pytest.approx(0.05)


def test_comparison_rejects_missing_experiment():
    results = pd.DataFrame(
        {"roc_auc": [0.70]},
        index=["baseline"],
    )

    with pytest.raises(ValueError):
        compare_feature_experiments(results)