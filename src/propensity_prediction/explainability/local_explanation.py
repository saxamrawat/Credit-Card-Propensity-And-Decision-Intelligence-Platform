"""Grouped local Shapley explanations for the selected model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from propensity_prediction.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)
from propensity_prediction.features.engineering import engineer_features


RANDOM_STATE = 42
DEFAULT_BACKGROUND_SIZE = 64

# These disjoint groups partition the original predictors. Derived fields
# are recomputed from their source inputs for every coalition, so a derived
# field is never counted as a separate player alongside its source columns.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "repayment_status_history": ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    "billing_and_credit_limit": (
        "LIMIT_BAL", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    ),
    "payment_amount_history": (
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ),
    "customer_profile": ("AGE", "SEX", "EDUCATION", "MARRIAGE"),
}

ORIGINAL_FEATURES = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
)


@dataclass(frozen=True)
class LocalGroupedExplanation:
    """Local probability contrast and its grouped Shapley allocation."""

    baseline_probability: float
    predicted_probability: float
    contributions: pd.DataFrame
    background_size: int
    method: str = "grouped interventional Shapley values"


def explain_customer_prediction(
    model: Pipeline,
    customer: pd.DataFrame | pd.Series,
    background: pd.DataFrame,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    random_state: int = RANDOM_STATE,
) -> LocalGroupedExplanation:
    """Explain one prediction with exact Shapley values over four groups.

    ``background`` should be a representative training sample containing
    original input columns (and may contain ID/DEFAULT). The baseline is
    the mean model probability over a deterministic sample of that
    background. For each coalition, present groups come from the customer
    and the remaining groups come from the same background rows. All
    engineered features are recomputed before prediction.

    Contributions sum to the customer's probability minus the baseline.
    This is an interventional contrast: replacing groups independently
    can break relationships between them, so contributions describe model
    behavior under these constructed comparisons, not causes or effects
    of real interventions.
    """
    if background_size <= 0:
        raise ValueError("background_size must be greater than zero.")
    if background.empty:
        raise ValueError("background must contain at least one row.")

    customer_frame = (
        customer.to_frame().T if isinstance(customer, pd.Series) else customer.copy()
    )
    if len(customer_frame) != 1:
        raise ValueError("customer must contain exactly one row.")

    missing = sorted(set(ORIGINAL_FEATURES) - set(customer_frame.columns))
    missing += sorted(set(ORIGINAL_FEATURES) - set(background.columns))
    if missing:
        raise ValueError(f"Missing required original feature columns: {sorted(set(missing))}")

    rng = np.random.default_rng(random_state)
    sample_size = min(background_size, len(background))
    sampled_positions = rng.choice(len(background), size=sample_size, replace=False)
    background_sample = background.iloc[sampled_positions][ORIGINAL_FEATURES].reset_index(drop=True)
    customer_row = customer_frame.iloc[0][ORIGINAL_FEATURES]

    def mean_probability(original_rows: pd.DataFrame) -> float:
        model_rows = engineer_features(original_rows[ORIGINAL_FEATURES])
        return float(model.predict_proba(model_rows)[:, 1].mean())

    group_names = tuple(FEATURE_GROUPS)
    coalition_values: dict[frozenset[str], float] = {}
    all_groups = frozenset(group_names)

    for subset_size in range(len(group_names) + 1):
        for subset in combinations(group_names, subset_size):
            coalition = frozenset(subset)
            hybrid = background_sample.copy()
            for group in coalition:
                columns = list(FEATURE_GROUPS[group])
                hybrid.loc[:, columns] = customer_row[columns].to_numpy()
            coalition_values[coalition] = mean_probability(hybrid)

    contributions = []
    n_groups = len(group_names)
    for group in group_names:
        value = 0.0
        others = [name for name in group_names if name != group]
        for subset_size in range(n_groups):
            weight = (
                factorial(subset_size)
                * factorial(n_groups - subset_size - 1)
                / factorial(n_groups)
            )
            for subset in combinations(others, subset_size):
                coalition = frozenset(subset)
                value += weight * (
                    coalition_values[coalition | {group}]
                    - coalition_values[coalition]
                )
        contributions.append(
            {
                "feature_group": group,
                "contribution": value,
                "direction": "higher than baseline" if value > 0 else "lower than baseline" if value < 0 else "no change",
            }
        )

    baseline = coalition_values[frozenset()]
    prediction = coalition_values[all_groups]
    contribution_frame = pd.DataFrame(contributions)
    contribution_frame["absolute_contribution"] = contribution_frame["contribution"].abs()
    contribution_frame = contribution_frame.sort_values(
        "absolute_contribution", ascending=False
    ).reset_index(drop=True)

    return LocalGroupedExplanation(
        baseline_probability=baseline,
        predicted_probability=prediction,
        contributions=contribution_frame,
        background_size=sample_size,
    )
