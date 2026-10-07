import numpy as np
import pandas as pd
import pytest

from propensity_prediction.explainability.feature_relationships import (
    ENGINEERED_FEATURE_SOURCES,
    analyze_engineered_feature_relationships,
)


def _sample_features(n=12):
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


def test_analyze_returns_lineage_and_ranked_associations():
    X = _sample_features()
    X["ID"] = np.arange(len(X))
    X["DEFAULT"] = np.arange(len(X)) % 2

    result = analyze_engineered_feature_relationships(X)

    assert len(result) == 20
    assert set(result["engineered_feature"]) == set(ENGINEERED_FEATURE_SOURCES)
    assert result["max_abs_spearman"].between(0, 1).all()
    assert "PAY_0" in result.loc[
        result["engineered_feature"] == "PAY_STATUS_MAX", "formula_sources"
    ].iloc[0]
    assert result["source_group"].notna().all()


def test_target_and_identifier_do_not_change_results():
    X = _sample_features()
    baseline = analyze_engineered_feature_relationships(X)
    X["ID"] = np.arange(len(X))
    X["DEFAULT"] = 1 - (np.arange(len(X)) % 2)
    actual = analyze_engineered_feature_relationships(X)

    pd.testing.assert_frame_equal(baseline, actual)


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="correlation_threshold"):
        analyze_engineered_feature_relationships(_sample_features(), 0)


def test_missing_original_column_is_reported():
    with pytest.raises(ValueError, match="Missing original feature columns"):
        analyze_engineered_feature_relationships(_sample_features().drop(columns="PAY_AMT1"))
