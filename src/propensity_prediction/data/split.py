from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "DEFAULT"

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

RANDOM_STATE = 42


@dataclass
class DatasetSplit:
    """
    Container for the train, validation, and test datasets.
    """

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def split_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """
    Split a cleaned dataset into training, validation, and test sets.

    Split proportions:
        Training:   70%
        Validation: 15%
        Test:       15%

    Stratification is performed using the target column.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    if len(df) == 0:
        raise ValueError("Cannot split an empty dataframe.")

    X = df.drop(columns=[target_column])
    y = df[target_column].copy()

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        stratify=y,
        random_state=random_state,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        stratify=y_holdout,
        random_state=random_state,
    )

    return DatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )