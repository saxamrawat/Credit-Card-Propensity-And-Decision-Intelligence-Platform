import numpy as np
import pandas as pd
import pytest

from propensity_prediction.explainability.local_explanation import (
    FEATURE_GROUPS,
    explain_customer_prediction,
)
from propensity_prediction.explainability.global_importance import build_final_model
from propensity_prediction.features.engineering import engineer_features


def _sample_features(n=8):
    index = np.arange(n, dtype=float)
    data = {
        "LIMIT_BAL": 10000 + index * 1000,
        "AGE": 20 + index,
        "SEX": (index.astype(int) % 2) + 1,
        "EDUCATION": (index.astype(int) % 3) + 1,
        "MARRIAGE": (index.astype(int) % 2) + 1,
    }
    for month in (0, 2, 3, 4, 5, 6):
        data[f"PAY_{month}"] = index % (month + 2)
    for month in range(1, 7):
        data[f"BILL_AMT{month}"] = 500 + index * (month * 25)
        data[f"PAY_AMT{month}"] = 100 + index * (month * 10)
    return pd.DataFrame(data)


class _ProbabilityModel:
    def predict_proba(self, X):
        logit = (
            -2.0
            + 0.30 * X["PAY_STATUS_MAX"].to_numpy()
            + 0.01 * X["BILL_TO_LIMIT_MEAN"].to_numpy()
            + 0.00001 * X["PAY_AMT_MEAN"].to_numpy()
            + 0.01 * X["AGE"].to_numpy()
        )
        probability = 1 / (1 + np.exp(-logit))
        return np.column_stack([1 - probability, probability])


def test_grouped_shapley_contributions_reconcile_to_probability_difference():
    rows = _sample_features()
    explanation = explain_customer_prediction(
        model=_ProbabilityModel(),
        customer=rows.iloc[[0]],
        background=rows.iloc[1:],
        background_size=4,
    )

    assert explanation.method == "grouped interventional Shapley values"
    assert explanation.background_size == 4
    assert set(explanation.contributions["feature_group"]) == set(FEATURE_GROUPS)
    assert np.isclose(
        explanation.contributions["contribution"].sum(),
        explanation.predicted_probability - explanation.baseline_probability,
        atol=1e-12,
    )
    assert explanation.contributions["absolute_contribution"].is_monotonic_decreasing


def test_explanation_is_reproducible_for_fixed_seed():
    rows = _sample_features()
    kwargs = {
        "model": _ProbabilityModel(),
        "customer": rows.iloc[[0]],
        "background": rows.iloc[1:],
        "background_size": 3,
    }
    first = explain_customer_prediction(**kwargs)
    second = explain_customer_prediction(**kwargs)

    assert first.baseline_probability == second.baseline_probability
    pd.testing.assert_frame_equal(first.contributions, second.contributions)


def test_series_customer_and_duplicate_background_indices_are_supported():
    rows = _sample_features()
    background = rows.iloc[1:]
    background.index = [0, 0, 1, 1, 2, 2, 3]

    explanation = explain_customer_prediction(
        model=_ProbabilityModel(),
        customer=rows.iloc[0],
        background=background,
        background_size=5,
    )

    assert explanation.background_size == 5
    assert np.isclose(
        explanation.contributions["contribution"].sum(),
        explanation.predicted_probability - explanation.baseline_probability,
    )


def test_invalid_customer_and_background_are_rejected():
    rows = _sample_features()
    with pytest.raises(ValueError, match="exactly one row"):
        explain_customer_prediction(_ProbabilityModel(), rows.iloc[:2], rows)
    with pytest.raises(ValueError, match="at least one row"):
        explain_customer_prediction(_ProbabilityModel(), rows.iloc[[0]], rows.iloc[:0])
    with pytest.raises(ValueError, match="greater than zero"):
        explain_customer_prediction(_ProbabilityModel(), rows.iloc[[0]], rows, 0)


def test_missing_features_are_rejected():
    rows = _sample_features().drop(columns="PAY_AMT1")
    with pytest.raises(ValueError, match="Missing required original feature"):
        explain_customer_prediction(_ProbabilityModel(), rows.iloc[[0]], rows)


def test_selected_pipeline_accepts_local_explanation_inputs():
    rows = _sample_features()
    model = build_final_model()
    model.fit(engineer_features(rows), pd.Series([0, 1] * 4))

    explanation = explain_customer_prediction(
        model=model,
        customer=rows.iloc[[0]],
        background=rows.iloc[1:],
        background_size=3,
    )

    assert 0 <= explanation.baseline_probability <= 1
    assert 0 <= explanation.predicted_probability <= 1
    assert np.isclose(
        explanation.contributions["contribution"].sum(),
        explanation.predicted_probability - explanation.baseline_probability,
    )
