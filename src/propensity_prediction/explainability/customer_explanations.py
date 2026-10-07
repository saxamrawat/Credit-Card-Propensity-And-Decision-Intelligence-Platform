"""Analyst-facing summaries for individual default-risk predictions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.data.split import split_dataset
from propensity_prediction.explainability.global_importance import (
    build_final_model,
    prepare_engineered_features,
)
from propensity_prediction.explainability.local_explanation import (
    LocalGroupedExplanation,
    explain_customer_prediction,
)


GROUP_LABELS = {
    "repayment_status_history": "repayment-status history",
    "billing_and_credit_limit": "billing amounts and credit limit",
    "payment_amount_history": "payment amounts",
    "customer_profile": "customer profile fields",
}


@dataclass(frozen=True)
class CustomerExplanation:
    """Structured model output and analyst-readable local explanation."""

    customer_id: object | None
    estimated_default_probability: float
    reference_probability: float
    difference_from_reference: float
    probability_context: str
    contribution_summary: str
    contributions: pd.DataFrame
    method: str


def build_customer_explanation(
    model,
    customer: pd.DataFrame | pd.Series,
    background: pd.DataFrame,
    background_size: int = 64,
    random_state: int = 42,
) -> CustomerExplanation:
    """Create an analyst-ready summary for a single model prediction."""
    customer_frame = customer.to_frame().T if isinstance(customer, pd.Series) else customer.copy()
    if len(customer_frame) != 1:
        raise ValueError("customer must contain exactly one row.")

    customer_id = customer_frame["ID"].iloc[0] if "ID" in customer_frame else None
    local = explain_customer_prediction(
        model=model,
        customer=customer_frame,
        background=background,
        background_size=background_size,
        random_state=random_state,
    )
    return _format_customer_explanation(customer_id, local)


def _format_customer_explanation(
    customer_id: object | None,
    local: LocalGroupedExplanation,
) -> CustomerExplanation:
    contributions = local.contributions.copy()
    contributions["contribution_percentage_points"] = (
        contributions["contribution"] * 100
    )
    contributions["group_label"] = contributions["feature_group"].map(GROUP_LABELS)
    contributions["analyst_wording"] = contributions.apply(
        _contribution_wording, axis=1
    )

    difference = local.predicted_probability - local.baseline_probability
    relative_position = (
        "at the reference" if difference == 0 else
        f"{'above' if difference > 0 else 'below'} that reference"
    )
    probability_context = (
        f"The model estimates a {local.predicted_probability:.1%} probability of "
        f"default for next month. The reference-sample mean model estimate is "
        f"{local.baseline_probability:.1%}; this customer's estimate is "
        f"{abs(difference) * 100:.1f} percentage points {relative_position}. "
        "This is a model estimate, not a certainty about an individual customer."
    )

    material = contributions.loc[
        contributions["absolute_contribution"] > 1e-12
    ]
    if material.empty:
        contribution_summary = (
            "No feature group changed the model estimate relative to the "
            "reference sample at the displayed precision."
        )
    else:
        phrases = material["analyst_wording"].tolist()
        contribution_summary = (
            "The largest group model attributions relative to the reference are "
            + "; ".join(phrases)
            + ". These are model attributions under the selected background "
            "comparison, not causes or intervention effects."
        )

    return CustomerExplanation(
        customer_id=customer_id,
        estimated_default_probability=local.predicted_probability,
        reference_probability=local.baseline_probability,
        difference_from_reference=difference,
        probability_context=probability_context,
        contribution_summary=contribution_summary,
        contributions=contributions,
        method=local.method,
    )


def _contribution_wording(row: pd.Series) -> str:
    return (
        f"{row['group_label']}: "
        f"{row['contribution_percentage_points']:+.1f} percentage points"
    )


def explain_validation_customer(
    customer_id: int,
    background_size: int = 64,
) -> CustomerExplanation:
    """Fit on training rows and explain a customer selected from validation.

    The identifier is used only to retrieve a validation row. It is removed
    from model inputs by the existing feature preparation function.
    """
    split = split_dataset(build_processed_dataset())
    matches = split.X_validation.loc[split.X_validation["ID"] == customer_id]
    if matches.empty:
        raise ValueError(f"Customer ID {customer_id} was not found in the validation split.")

    model = build_final_model()
    X_train = prepare_engineered_features(split.X_train)
    model.fit(X_train, split.y_train)

    return build_customer_explanation(
        model=model,
        customer=matches.iloc[[0]],
        background=split.X_train,
        background_size=background_size,
    )
