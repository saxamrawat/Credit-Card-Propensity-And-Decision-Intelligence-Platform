import numpy as np
import pandas as pd
import pytest

from propensity_prediction.explainability.risk_segmentation import (
    assign_risk_bands,
    fit_risk_bands,
    summarize_risk_bands,
)


def test_four_quantile_bands_and_summary():
    probabilities = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.80])
    y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    definition = fit_risk_bands(probabilities)
    result = summarize_risk_bands(
        y_true,
        probabilities,
        definition,
        customer_ids=np.arange(1, 9),
    )

    assert definition.labels == ("Lower", "Moderate", "Elevated", "Highest")
    assert len(definition.cut_points) == 3
    assert result.summary["customer_count"].sum() == len(probabilities)
    assert result.assignments["ID"].tolist() == list(range(1, 9))
    assert result.summary["observed_default_rate"].between(0, 1).all()
    assert result.definition == definition


def test_equal_probabilities_are_not_split_across_bands():
    probabilities = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3])
    definition = fit_risk_bands(probabilities, n_bands=4)
    assigned = assign_risk_bands(probabilities, definition)

    assert assigned.loc[probabilities == 0.1].nunique() == 1
    assert assigned.loc[probabilities == 0.2].nunique() == 1
    assert assigned.loc[probabilities == 0.3].nunique() == 1
    assert len(definition.labels) < 4


def test_constant_scores_produce_one_band():
    definition = fit_risk_bands([0.25, 0.25, 0.25, 0.25])

    assert definition.labels == ("Single score band",)
    assert assign_risk_bands([0.25, 0.25], definition).nunique() == 1


def test_boundaries_cover_new_scores_outside_reference_range():
    definition = fit_risk_bands([0.2, 0.3, 0.4, 0.5])
    assigned = assign_risk_bands([0.0, 1.0], definition)

    assert not assigned.isna().any()
    assert list(assigned.astype(str)) == ["Lower", "Highest"]


@pytest.mark.parametrize(
    "scores",
    [[], [np.nan, 0.3], [-0.1, 0.4], [0.2, np.inf], [1.2]],
)
def test_invalid_scores_are_rejected(scores):
    with pytest.raises(ValueError, match="probabilities"):
        fit_risk_bands(scores)


def test_summary_rejects_mismatched_lengths_and_nonbinary_targets():
    definition = fit_risk_bands([0.1, 0.2])
    with pytest.raises(ValueError, match="equal lengths"):
        summarize_risk_bands([0], [0.1, 0.2], definition)
    with pytest.raises(ValueError, match="binary values"):
        summarize_risk_bands([0, 2], [0.1, 0.2], definition)
