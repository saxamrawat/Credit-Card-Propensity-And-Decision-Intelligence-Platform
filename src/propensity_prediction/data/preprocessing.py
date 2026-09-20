from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from propensity_prediction.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)


def create_preprocessor() -> ColumnTransformer:
    """
    Create the preprocessing transformer for model features.

    The transformer is returned unfitted.

    Numerical features:
        StandardScaler

    Categorical features:
        OneHotEncoder with unknown-category handling

    Ordinal repayment-status features:
        Passed through unchanged
    """

    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
            (
                "ordinal",
                "passthrough",
                ORDINAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_preprocessor_feature_names(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """
    Return feature names produced by a fitted preprocessor.
    """

    return preprocessor.get_feature_names_out().tolist()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the model feature dataframe.

    ID and DEFAULT are excluded.
    """

    feature_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + ORDINAL_FEATURES
    )

    return df[feature_columns].copy()