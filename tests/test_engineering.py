import numpy as np
import pandas as pd

from propensity_prediction.features.engineering import (
    engineer_features,
)


def create_test_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "LIMIT_BAL": [10000, 20000, 30000],
            "SEX": [1, 2, 1],
            "EDUCATION": [1, 2, 3],
            "MARRIAGE": [1, 2, 3],
            "AGE": [25, 30, 35],
            "PAY_0": [2, 0, -1],
            "PAY_2": [1, 0, 0],
            "PAY_3": [0, 1, 0],
            "PAY_4": [0, 0, 0],
            "PAY_5": [0, 0, 0],
            "PAY_6": [0, 0, 0],
            "BILL_AMT1": [1000, 2000, 3000],
            "BILL_AMT2": [900, 1800, 2700],
            "BILL_AMT3": [800, 1600, 2400],
            "BILL_AMT4": [700, 1400, 2100],
            "BILL_AMT5": [600, 1200, 1800],
            "BILL_AMT6": [500, 1000, 1500],
            "PAY_AMT1": [500, 1000, 1500],
            "PAY_AMT2": [400, 800, 1200],
            "PAY_AMT3": [300, 600, 900],
            "PAY_AMT4": [200, 400, 600],
            "PAY_AMT5": [100, 200, 300],
            "PAY_AMT6": [50, 100, 150],
            "DEFAULT": [0, 1, 0],
        }
    )


def test_engineer_features_preserves_row_count():
    df = create_test_dataframe()

    result = engineer_features(df)

    assert result.shape[0] == df.shape[0]


def test_engineer_features_preserves_original_columns():
    df = create_test_dataframe()

    result = engineer_features(df)

    assert set(df.columns).issubset(result.columns)


def test_engineered_columns_are_created():
    df = create_test_dataframe()

    result = engineer_features(df)

    expected_columns = {
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
    }

    assert expected_columns.issubset(result.columns)


def test_delayed_month_count():
    df = create_test_dataframe()

    result = engineer_features(df)

    assert result.loc[0, "PAY_STATUS_DELAYED_MONTHS"] == 2
    assert result.loc[1, "PAY_STATUS_DELAYED_MONTHS"] == 1
    assert result.loc[2, "PAY_STATUS_DELAYED_MONTHS"] == 0


def test_payment_to_bill_handles_zero_bills():
    df = create_test_dataframe()

    df.loc[0, "BILL_AMT1"] = 0

    result = engineer_features(df)

    assert not np.isinf(
        result.loc[0, "PAYMENT_TO_BILL_MEAN"]
    )


def test_bill_to_limit_mean():
    df = create_test_dataframe()

    result = engineer_features(df)

    expected = (
        df.loc[0, [
            "BILL_AMT6",
            "BILL_AMT5",
            "BILL_AMT4",
            "BILL_AMT3",
            "BILL_AMT2",
            "BILL_AMT1",
        ]].mean()
        / df.loc[0, "LIMIT_BAL"]
    )

    assert result.loc[0, "BILL_TO_LIMIT_MEAN"] == expected


def test_target_is_not_used_to_create_features():
    df = create_test_dataframe()

    result_without_target = engineer_features(
        df.drop(columns=["DEFAULT"])
    )

    result_with_target = engineer_features(df)

    engineered_columns = [
        column
        for column in result_with_target.columns
        if column not in df.columns
    ]

    for column in engineered_columns:
        assert np.allclose(
            result_without_target[column].to_numpy(),
            result_with_target[column].to_numpy(),
            equal_nan=True,
        )


def test_original_dataframe_is_not_modified():
    df = create_test_dataframe()
    original = df.copy(deep=True)

    engineer_features(df)

    pd.testing.assert_frame_equal(df, original)