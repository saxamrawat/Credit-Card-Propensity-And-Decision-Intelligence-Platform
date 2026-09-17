import pandas as pd
import pytest

from propensity_prediction.data.cleaning import (
    clean_data,
    standardize_columns,
)
from propensity_prediction.data.load import load_raw_data


@pytest.fixture
def raw_df():
    return load_raw_data()


def test_standardize_columns(raw_df):
    cleaned = standardize_columns(raw_df)

    assert cleaned.columns.tolist() == [
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "DEFAULT",
    ]


def test_clean_data_preserves_shape(raw_df):
    cleaned = clean_data(raw_df)

    assert cleaned.shape == (30_000, 25)


def test_clean_data_has_unique_ids(raw_df):
    cleaned = clean_data(raw_df)

    assert cleaned["ID"].is_unique


def test_clean_data_has_valid_categories(raw_df):
    cleaned = clean_data(raw_df)

    assert cleaned["SEX"].isin([1, 2]).all()
    assert cleaned["EDUCATION"].isin([0, 1, 2, 3, 4]).all()
    assert cleaned["MARRIAGE"].isin([0, 1, 2, 3]).all()
    assert cleaned["DEFAULT"].isin([0, 1]).all()


def test_education_invalid_values_are_mapped(raw_df):
    cleaned = clean_data(raw_df)

    assert not cleaned["EDUCATION"].isin([5, 6]).any()


def test_no_missing_values_after_cleaning(raw_df):
    cleaned = clean_data(raw_df)

    assert not cleaned.isna().any().any()


def test_clean_data_does_not_modify_raw_data(raw_df):
    original = raw_df.copy(deep=True)

    clean_data(raw_df)

    pd.testing.assert_frame_equal(raw_df, original)