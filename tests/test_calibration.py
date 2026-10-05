import numpy as np
import pandas as pd

from propensity_prediction.models.calibration import (
    calculate_calibration_curve,
    calculate_probability_metrics,
    evaluate_thresholds,
)


def test_evaluate_thresholds_returns_expected_columns():
    y_true = pd.Series([0, 0, 1, 1, 1])
    probabilities = np.array(
        [0.1, 0.3, 0.6, 0.8, 0.9]
    )

    result = evaluate_thresholds(
        y_true,
        probabilities,
        thresholds=[0.3, 0.5, 0.7],
    )

    assert list(result.columns) == [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "positive_rate",
    ]

    assert len(result) == 3


def test_thresholds_are_preserved():
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array(
        [0.1, 0.7, 0.2, 0.8]
    )

    thresholds = [0.2, 0.5, 0.8]

    result = evaluate_thresholds(
        y_true,
        probabilities,
        thresholds,
    )

    assert result["threshold"].tolist() == thresholds


def test_probability_metrics_are_finite():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array(
        [0.1, 0.2, 0.7, 0.9]
    )

    result = calculate_probability_metrics(
        y_true,
        probabilities,
    )

    assert set(result) == {
        "roc_auc",
        "pr_auc",
        "brier_score",
    }

    assert np.isfinite(
        list(result.values())
    ).all()


def test_brier_score_is_non_negative():
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array(
        [0.1, 0.8, 0.2, 0.9]
    )

    result = calculate_probability_metrics(
        y_true,
        probabilities,
    )

    assert result["brier_score"] >= 0


def test_calibration_curve_has_expected_columns():
    y_true = pd.Series(
        [0, 0, 0, 1, 1, 1] * 10
    )

    probabilities = np.array(
        [0.1, 0.2, 0.3, 0.6, 0.7, 0.8] * 10
    )

    result = calculate_calibration_curve(
        y_true,
        probabilities,
        n_bins=5,
    )

    assert list(result.columns) == [
        "mean_predicted_probability",
        "observed_default_rate",
    ]

    assert not result.empty