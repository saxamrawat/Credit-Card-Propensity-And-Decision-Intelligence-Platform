from __future__ import annotations


IDENTIFIER_COLUMNS = [
    "ID",
]


NUMERICAL_FEATURES = [
    "LIMIT_BAL",
    "AGE",
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
]


CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]


ORDINAL_FEATURES = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


TARGET_COLUMNS = [
    "DEFAULT",
]


def get_feature_schema() -> dict[str, list[str]]:
    """
    Return the project's canonical feature representation schema.
    """
    return {
        "identifier": IDENTIFIER_COLUMNS.copy(),
        "numerical": NUMERICAL_FEATURES.copy(),
        "categorical": CATEGORICAL_FEATURES.copy(),
        "ordinal": ORDINAL_FEATURES.copy(),
        "target": TARGET_COLUMNS.copy(),
    }


def get_feature_columns() -> list[str]:
    """
    Return all columns that may be supplied as model features.
    """
    return (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + ORDINAL_FEATURES
    )


def validate_feature_schema(columns: list[str]) -> None:
    """
    Validate that the supplied dataset contains the expected schema columns.
    """
    expected_columns = set(
        IDENTIFIER_COLUMNS
        + NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + ORDINAL_FEATURES
        + TARGET_COLUMNS
    )

    actual_columns = set(columns)

    missing_columns = expected_columns - actual_columns

    if missing_columns:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing_columns)}"
        )

    feature_groups = [
        IDENTIFIER_COLUMNS,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        ORDINAL_FEATURES,
        TARGET_COLUMNS,
    ]

    seen: set[str] = set()

    for group in feature_groups:
        overlap = seen.intersection(group)

        if overlap:
            raise ValueError(
                f"Columns appear in multiple schema groups: {sorted(overlap)}"
            )

        seen.update(group)