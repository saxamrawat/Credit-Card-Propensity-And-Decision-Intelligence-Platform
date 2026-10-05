from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from propensity_prediction.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)

def create_preprocessor(
    numerical_features=None,
    categorical_features=None,
    ordinal_features=None,
):
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

    numerical_features = (
        NUMERICAL_FEATURES
        if numerical_features is None
        else numerical_features
    )

    categorical_features = (
        CATEGORICAL_FEATURES
        if categorical_features is None
        else categorical_features
    )

    ordinal_features = (
        ORDINAL_FEATURES
        if ordinal_features is None
        else ordinal_features
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "ordinal",
                "passthrough",
                ordinal_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


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