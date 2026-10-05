import numpy as np
import pandas as pd

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.features.engineering import (
    engineer_features,
)


ENGINEERED_COLUMNS = [
    "PAY_STATUS_MEAN",
    "PAY_STATUS_MAX",
    "PAY_STATUS_MIN",
    "PAY_STATUS_STD",
    "PAY_STATUS_DELAYED_MONTHS",
    "PAY_STATUS_TREND",
    "BILL_AMT_MEAN",
    "BILL_AMT_MAX",
    "BILL_AMT_MIN",
    "BILL_AMT_STD",
    "BILL_AMT_TREND",
    "PAY_AMT_MEAN",
    "PAY_AMT_MAX",
    "PAY_AMT_MIN",
    "PAY_AMT_STD",
    "PAY_AMT_TREND",
    "PAYMENT_TO_BILL_MEAN",
    "PAYMENT_TO_BILL_MIN",
    "PAYMENT_TO_BILL_MAX",
    "BILL_TO_LIMIT_MEAN",
]


def get_engineered_dataset() -> pd.DataFrame:
    """Build the actual processed dataset and engineer its features."""
    df = build_processed_dataset()

    return engineer_features(df)


def test_engineered_dataset_preserves_row_count():
    df = build_processed_dataset()

    engineered = engineer_features(df)

    assert engineered.shape[0] == 30_000


def test_engineered_dataset_preserves_ids():
    df = build_processed_dataset()

    engineered = engineer_features(df)

    assert engineered["ID"].equals(df["ID"])
    assert engineered["ID"].is_unique


def test_engineered_dataset_preserves_target():
    df = build_processed_dataset()

    engineered = engineer_features(df)

    assert engineered["DEFAULT"].equals(df["DEFAULT"])


def test_all_expected_engineered_columns_exist():
    engineered = get_engineered_dataset()

    for column in ENGINEERED_COLUMNS:
        assert column in engineered.columns


def test_engineered_columns_are_numeric():
    engineered = get_engineered_dataset()

    for column in ENGINEERED_COLUMNS:
        assert pd.api.types.is_numeric_dtype(
            engineered[column]
        )


def test_no_infinite_values():
    engineered = get_engineered_dataset()

    numeric_columns = engineered.select_dtypes(
        include=np.number
    )

    assert not np.isinf(numeric_columns.to_numpy()).any()


def test_payment_to_bill_features_are_finite_when_defined():
    engineered = get_engineered_dataset()
    ratio_columns = [ "PAYMENT_TO_BILL_MEAN",
                      "PAYMENT_TO_BILL_MIN",
                      "PAYMENT_TO_BILL_MAX", ]
    for column in ratio_columns:
        values = engineered[column].dropna()

        # Each feature should have observed values.
        assert not values.empty

        # All defined values must be finite.
        assert np.isfinite(values).all()

def test_bill_to_limit_mean_is_finite():
    engineered = get_engineered_dataset()
    values = engineered["BILL_TO_LIMIT_MEAN"].dropna()

    assert np.isfinite(values).all()

def test_credit_utilization_proxy_is_finite():
    engineered = get_engineered_dataset()

    values = engineered["BILL_TO_LIMIT_MEAN"].dropna()

    assert np.isfinite(values).all()


def test_delayed_month_count_is_between_zero_and_six():
    engineered = get_engineered_dataset()

    values = engineered["PAY_STATUS_DELAYED_MONTHS"]

    assert values.min() >= 0
    assert values.max() <= 6


def test_engineered_dataset_has_no_duplicate_rows():
    engineered = get_engineered_dataset()

    assert not engineered.duplicated().any()


def test_feature_engineering_does_not_modify_input():
    df = build_processed_dataset()

    original = df.copy(deep=True)

    engineer_features(df)

    pd.testing.assert_frame_equal(df, original)