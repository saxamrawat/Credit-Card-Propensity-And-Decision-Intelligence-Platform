"""Describe overlap between engineered and original model features."""

from __future__ import annotations

import pandas as pd

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)
from propensity_prediction.data.split import split_dataset
from propensity_prediction.features.engineering import engineer_features


CORRELATION_THRESHOLD = 0.70

# Formula lineage is explicit and independent of observed correlation.
ENGINEERED_FEATURE_SOURCES: dict[str, tuple[str, ...]] = {
    "PAY_STATUS_MEAN": ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    "PAY_STATUS_MAX": ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    "PAY_STATUS_MIN": ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    "PAY_STATUS_STD": ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    "PAY_STATUS_DELAYED_MONTHS": ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    "PAY_STATUS_TREND": ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    "BILL_AMT_MEAN": ("BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"),
    "BILL_AMT_MAX": ("BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"),
    "BILL_AMT_MIN": ("BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"),
    "BILL_AMT_STD": ("BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"),
    "BILL_AMT_TREND": ("BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"),
    "PAY_AMT_MEAN": ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"),
    "PAY_AMT_MAX": ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"),
    "PAY_AMT_MIN": ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"),
    "PAY_AMT_STD": ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"),
    "PAY_AMT_TREND": ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"),
    "PAYMENT_TO_BILL_MEAN": ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"),
    "PAYMENT_TO_BILL_MIN": ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"),
    "PAYMENT_TO_BILL_MAX": ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"),
    "BILL_TO_LIMIT_MEAN": ("LIMIT_BAL", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"),
}


def analyze_engineered_feature_relationships(
    X: pd.DataFrame,
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """Rank each engineered feature's bivariate associations with originals.

    Spearman correlation is used to capture monotonic relationships and
    reduce sensitivity to the scale/skew of financial amounts. This is a
    descriptive overlap diagnostic, not a measure of conditional or causal
    importance. ``X`` must contain original features only; target and ID are
    discarded if supplied.
    """
    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1].")

    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
    missing = sorted(set(feature_columns) - set(X.columns))
    if missing:
        raise ValueError(f"Missing original feature columns: {missing}")

    original = X[feature_columns].copy()
    engineered = engineer_features(original)
    rows = []
    for feature, sources in ENGINEERED_FEATURE_SOURCES.items():
        correlations = engineered[list(sources)].corrwith(
            engineered[feature], method="spearman"
        ).dropna()
        ranked = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
        strong = ranked[ranked.abs() >= correlation_threshold]
        rows.append(
            {
                "engineered_feature": feature,
                "source_group": _source_group(feature),
                "formula_sources": ", ".join(sources),
                "top_original_feature": ranked.index[0] if len(ranked) else None,
                "top_spearman": float(ranked.iloc[0]) if len(ranked) else float("nan"),
                "max_abs_spearman": float(ranked.abs().iloc[0]) if len(ranked) else float("nan"),
                "strong_original_features": ", ".join(strong.index),
                "n_strong_original_features": int(len(strong)),
                "bivariate_overlap_label": (
                    "strong association with original feature(s)"
                    if len(strong)
                    else "no original feature reaches threshold"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["source_group", "max_abs_spearman"], ascending=[True, False]
    ).reset_index(drop=True)


def _source_group(feature: str) -> str:
    if feature.startswith("PAY_STATUS_"):
        return "repayment_status"
    if feature.startswith("BILL_TO_LIMIT_"):
        return "billing_relative_to_limit"
    if feature.startswith("PAYMENT_TO_BILL_"):
        return "payment_relative_to_billing"
    if feature.startswith("BILL_AMT_"):
        return "billing_amount_history"
    return "payment_amount_history"


def get_training_feature_relationships() -> pd.DataFrame:
    """Analyze the training split only; validation/test are not accessed."""
    split = split_dataset(build_processed_dataset())
    return analyze_engineered_feature_relationships(split.X_train)


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    print(get_training_feature_relationships().to_string(index=False))
