from __future__ import annotations

import pandas as pd


COLUMN_RENAME_MAP = {
    ("X1", "LIMIT_BAL"): "LIMIT_BAL",
    ("X2", "SEX"): "SEX",
    ("X3", "EDUCATION"): "EDUCATION",
    ("X4", "MARRIAGE"): "MARRIAGE",
    ("X5", "AGE"): "AGE",
    ("X6", "PAY_0"): "PAY_0",
    ("X7", "PAY_2"): "PAY_2",
    ("X8", "PAY_3"): "PAY_3",
    ("X9", "PAY_4"): "PAY_4",
    ("X10", "PAY_5"): "PAY_5",
    ("X11", "PAY_6"): "PAY_6",
    ("X12", "BILL_AMT1"): "BILL_AMT1",
    ("X13", "BILL_AMT2"): "BILL_AMT2",
    ("X14", "BILL_AMT3"): "BILL_AMT3",
    ("X15", "BILL_AMT4"): "BILL_AMT4",
    ("X16", "BILL_AMT5"): "BILL_AMT5",
    ("X17", "BILL_AMT6"): "BILL_AMT6",
    ("X18", "PAY_AMT1"): "PAY_AMT1",
    ("X19", "PAY_AMT2"): "PAY_AMT2",
    ("X20", "PAY_AMT3"): "PAY_AMT3",
    ("X21", "PAY_AMT4"): "PAY_AMT4",
    ("X22", "PAY_AMT5"): "PAY_AMT5",
    ("X23", "PAY_AMT6"): "PAY_AMT6",
    ("Y", "default payment next month"): "DEFAULT",
}


EXPECTED_COLUMNS = [
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


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw UCI MultiIndex columns into semantic column names.

    The raw dataframe is not modified.
    """
    cleaned = df.copy()

    renamed_columns = []

    for column in cleaned.columns:
        if column[1] == "ID":
            renamed_columns.append("ID")
        elif column in COLUMN_RENAME_MAP:
            renamed_columns.append(COLUMN_RENAME_MAP[column])
        else:
            raise ValueError(f"Unexpected column found: {column}")

    cleaned.columns = renamed_columns

    return cleaned


def clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle undocumented categorical values identified during validation.

    EDUCATION:
        documented: 1-4
        observed invalid values: 0, 5, 6
        invalid values are represented as 0 = Unknown/Other

    MARRIAGE:
        documented: 1-3
        observed invalid value: 0
        0 is retained as the explicit Unknown/Other category.
    """
    cleaned = df.copy()

    cleaned["EDUCATION"] = cleaned["EDUCATION"].where(
        cleaned["EDUCATION"].isin([1, 2, 3, 4]),
        0,
    )

    cleaned["MARRIAGE"] = cleaned["MARRIAGE"].where(
        cleaned["MARRIAGE"].isin([1, 2, 3]),
        0,
    )

    return cleaned


def validate_cleaned_data(df: pd.DataFrame) -> None:
    """
    Validate the standardized dataframe after cleaning.
    """
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    if len(df.columns) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"Expected {len(EXPECTED_COLUMNS)} columns, "
            f"found {len(df.columns)}"
        )

    if df["ID"].duplicated().any():
        raise ValueError("ID values must be unique.")

    if df.isna().any().any():
        raise ValueError("Cleaned data contains missing values.")

    if not df["DEFAULT"].isin([0, 1]).all():
        raise ValueError("DEFAULT must contain only 0 and 1.")

    if not df["SEX"].isin([1, 2]).all():
        raise ValueError("SEX contains invalid values.")

    if not df["EDUCATION"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("EDUCATION contains invalid values.")

    if not df["MARRIAGE"].isin([0, 1, 2, 3]).all():
        raise ValueError("MARRIAGE contains invalid values.")

    if (df["LIMIT_BAL"] < 0).any():
        raise ValueError("LIMIT_BAL contains negative values.")

    payment_columns = [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    if (df[payment_columns] < 0).any().any():
        raise ValueError("Payment amount columns contain negative values.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a standardized and validated dataframe from the raw dataset.

    The input dataframe is never modified.
    """
    cleaned = standardize_columns(df)
    cleaned = clean_categorical_values(cleaned)

    validate_cleaned_data(cleaned)

    return cleaned