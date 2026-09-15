import pytest
import pandas as pd

from propensity_prediction.data.load import load_raw_data

@pytest.fixture
def raw_data():
    return load_raw_data()

def test_load_returns_dataframe(raw_data):
    assert isinstance(raw_data, pd.DataFrame)


def test_load_returns_expected_shape(raw_data):
    assert raw_data.shape == (30000, 25)

def test_load_returns_expected_columns(raw_data):
    expected_columns = pd.MultiIndex.from_tuples([
        ("Unnamed: 0_level_0", "ID"),
        ("X1", "LIMIT_BAL"),
        ("X2", "SEX"),
        ("X3", "EDUCATION"),
        ("X4", "MARRIAGE"),
        ("X5", "AGE"),
        ("X6", "PAY_0"),
        ("X7", "PAY_2"),
        ("X8", "PAY_3"),
        ("X9", "PAY_4"),
        ("X10", "PAY_5"),
        ("X11", "PAY_6"),
        ("X12", "BILL_AMT1"),
        ("X13", "BILL_AMT2"),
        ("X14", "BILL_AMT3"),
        ("X15", "BILL_AMT4"),
        ("X16", "BILL_AMT5"),
        ("X17", "BILL_AMT6"),
        ("X18", "PAY_AMT1"),
        ("X19", "PAY_AMT2"),
        ("X20", "PAY_AMT3"),
        ("X21", "PAY_AMT4"),
        ("X22", "PAY_AMT5"),
        ("X23", "PAY_AMT6"),
        ("Y", "default payment next month"),
    ])

    assert isinstance(raw_data.columns, pd.MultiIndex)
    assert raw_data.columns.equals(expected_columns)

def test_load_invalid_path_raises_error():
    with pytest.raises(FileNotFoundError):
        load_raw_data("data/raw/does_not_exist.xls")