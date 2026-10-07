from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from propensity_prediction.explainability import customer_explanations as module
from propensity_prediction.explainability.local_explanation import FEATURE_GROUPS
from propensity_prediction.features.engineering import engineer_features


def _sample_features(n=8):
    index = np.arange(n, dtype=float)
    data = {
        "ID": np.arange(100, 100 + n),
        "LIMIT_BAL": 10000 + index * 1000,
        "AGE": 20 + index,
        "SEX": (index.astype(int) % 2) + 1,
        "EDUCATION": (index.astype(int) % 3) + 1,
        "MARRIAGE": (index.astype(int) % 2) + 1,
        "DEFAULT": (index.astype(int) % 2),
    }
    for month in (0, 2, 3, 4, 5, 6):
        data[f"PAY_{month}"] = index % (month + 2)
    for month in range(1, 7):
        data[f"BILL_AMT{month}"] = 500 + index * (month * 25)
        data[f"PAY_AMT{month}"] = 100 + index * (month * 10)
    return pd.DataFrame(data)


class _ProbabilityModel:
    def fit(self, X, y):
        self.fit_columns = list(X.columns)
        return self

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


def test_build_customer_explanation_has_probability_context_and_wording():
    rows = _sample_features()
    result = module.build_customer_explanation(
        model=_ProbabilityModel(),
        customer=rows.iloc[[0]],
        background=rows.iloc[1:6],
        background_size=4,
    )

    assert result.customer_id == 100
    assert 0 <= result.estimated_default_probability <= 1
    assert 0 <= result.reference_probability <= 1
    assert np.isclose(
        result.difference_from_reference,
        result.estimated_default_probability - result.reference_probability,
    )
    assert "probability of default for next month" in result.probability_context
    assert "not a certainty" in result.probability_context
    assert "not causes or intervention effects" in result.contribution_summary
    assert set(result.contributions["feature_group"]) == set(FEATURE_GROUPS)
    assert "percentage points" in " ".join(result.contributions["analyst_wording"])


def test_build_customer_explanation_rejects_multiple_rows():
    rows = _sample_features()
    with pytest.raises(ValueError, match="exactly one row"):
        module.build_customer_explanation(
            _ProbabilityModel(), rows.iloc[:2], rows.iloc[2:]
        )


def test_validation_entry_point_fits_training_rows_and_uses_id_only_for_lookup(
    monkeypatch,
):
    rows = _sample_features()
    split = SimpleNamespace(
        X_train=rows.iloc[:6].drop(columns="DEFAULT"),
        y_train=rows.iloc[:6]["DEFAULT"],
        X_validation=rows.iloc[6:].drop(columns="DEFAULT"),
    )
    model = _ProbabilityModel()
    monkeypatch.setattr(module, "build_processed_dataset", lambda: rows)
    monkeypatch.setattr(module, "split_dataset", lambda _: split)
    monkeypatch.setattr(module, "build_final_model", lambda: model)
    monkeypatch.setattr(
        module,
        "prepare_engineered_features",
        lambda X: engineer_features(X.drop(columns="ID").copy()),
    )

    result = module.explain_validation_customer(customer_id=106, background_size=3)

    assert result.customer_id == 106
    assert "ID" not in model.fit_columns


def test_validation_entry_point_rejects_ids_outside_validation(monkeypatch):
    rows = _sample_features()
    split = SimpleNamespace(
        X_train=rows.iloc[:6].drop(columns="DEFAULT"),
        y_train=rows.iloc[:6]["DEFAULT"],
        X_validation=rows.iloc[6:].drop(columns="DEFAULT"),
    )
    monkeypatch.setattr(module, "build_processed_dataset", lambda: rows)
    monkeypatch.setattr(module, "split_dataset", lambda _: split)

    with pytest.raises(ValueError, match="not found in the validation split"):
        module.explain_validation_customer(customer_id=100)
