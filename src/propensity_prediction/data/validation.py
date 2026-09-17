# Validate raw data

import pandas as pd

def structure(df : pd.DataFrame):
    print("Shape:", df.shape)
    print("Number of Columns: " ,len(df.columns))
    print("Number of Rows: " ,len(df))
    print("Mutli-index Columns: " , df.columns)
    print("Unique Columns ? : " , df.columns.is_unique)

    return df.shape

def missing(df: pd.DataFrame):
    print("Total missing values:", df.isna().sum().sum())
    print("Rows with missing values:", df.isna().any(axis=1).sum())
    print("Columns with missing values:", df.isna().any(axis=0).sum())

    return df.isna().sum().sum()

def duplicates(df: pd.DataFrame):
    duplicate_count = df.duplicated().sum()

    print("Number of duplicate rows:", duplicate_count)
    print("Percentage of duplicate rows:", (duplicate_count / len(df)) * 100)

    return df[df.duplicated(keep=False)]

def duplicate_ids(df: pd.DataFrame):
    id_columns = [column for column in df.columns if column[1] == "ID"]

    if len(id_columns) != 1:
        raise ValueError(f"Expected exactly one ID column, found {len(id_columns)}")

    id_column = id_columns[0]
    duplicate_rows = int(df[id_column].duplicated(keep=False).sum())
    unique_ids = int(df[id_column].nunique())

    return {
        "unique_ids": unique_ids,
        "duplicate_rows": duplicate_rows,
        "ids_unique": unique_ids == len(df),
    }

def data_type_validation(df: pd.DataFrame):
    print("Data Types:")
    print(df.dtypes)

    print("\nData Type Counts:")
    print(df.dtypes.value_counts())

def target_validation(df: pd.DataFrame):
    target = ("Y", "default payment next month")

    print("Unique values:")
    print(df[target].unique())

    print("\nCounts:")
    print(df[target].value_counts())

    print("\nProportions:")
    print(df[target].value_counts(normalize=True))

    print("\nInvalid target values:")
    print(sorted(set(df[target].dropna()) - {0, 1}))

def range_validation(df: pd.DataFrame):

    # Categorical variables
    valid_categories = {
        ("X2", "SEX"): {1, 2},
        ("X3", "EDUCATION"): {1, 2, 3, 4},
        ("X4", "MARRIAGE"): {1, 2, 3},
        ("Y",
         "default payment next month"): {0, 1},
    }

    for column, valid_values in valid_categories.items():

        actual_values = set(df[column].dropna().unique())
        invalid_values = actual_values - valid_values

        print(f"{column[0]} invalid values: {invalid_values}")

    # Age — sanity check
    age = ("X5", "AGE")

    print("\nAGE")
    print("Minimum:", df[age].min())
    print("Maximum:", df[age].max())

    # Credit limit
    limit_bal = ("X1", "LIMIT_BAL")

    print("\nLIMIT_BAL")
    print("Negative values:", (df[limit_bal] < 0).sum())

    # Previous payments
    payment_columns = [
        ("X18", "PAY_AMT1"),
        ("X19", "PAY_AMT2"),
        ("X20", "PAY_AMT3"),
        ("X21", "PAY_AMT4"),
        ("X22", "PAY_AMT5"),
        ("X23", "PAY_AMT6"),
    ]

    for column in payment_columns:
        print(
            f"{column[1]} negative values:",
            (df[column] < 0).sum()
        )