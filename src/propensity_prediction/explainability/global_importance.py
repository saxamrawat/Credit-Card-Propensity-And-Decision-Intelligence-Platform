"""Global feature importance for the selected credit-default model."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.data.preprocessing import create_preprocessor
from propensity_prediction.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)
from propensity_prediction.data.split import split_dataset
from propensity_prediction.features.engineering import (
    ENGINEERED_FEATURES,
    engineer_features,
)


RANDOM_STATE = 42

FINAL_MODEL_PARAMS = {
    "l2_regularization": 1.0,
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 31,
}


def build_final_model() -> Pipeline:
    """Build the final selected HGB preprocessing/model pipeline."""

    numerical_features = NUMERICAL_FEATURES + ENGINEERED_FEATURES

    preprocessor = create_preprocessor(
        numerical_features=numerical_features,
        categorical_features=CATEGORICAL_FEATURES,
        ordinal_features=ORDINAL_FEATURES,
    )

    model = HistGradientBoostingClassifier(
        **FINAL_MODEL_PARAMS,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def prepare_engineered_features(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the exact feature set used by the final model."""

    X = X.drop(columns=["ID"]).copy()

    return engineer_features(X)


def fit_final_model() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """
    Fit the final selected model using training data only.

    Returns:
        A tuple containing:
        - fitted model pipeline
        - validation features
        - validation target
    """

    df = build_processed_dataset()
    split = split_dataset(df)

    X_train = prepare_engineered_features(split.X_train)
    X_validation = prepare_engineered_features(split.X_validation)

    model = build_final_model()

    model.fit(
        X_train,
        split.y_train,
    )

    return (
        model,
        X_validation,
        split.y_validation,
    )


def calculate_global_permutation_importance(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Calculate permutation importance on the validation set.

    Average precision is used because default is an imbalanced
    binary classification target and PR-AUC was the primary
    model-selection metric in Phase 5.
    """

    result = permutation_importance(
        estimator=model,
        X=X_validation,
        y=y_validation,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": X_validation.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance = importance.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    return importance


def get_global_feature_importance(
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Fit the final model and calculate global permutation importance."""

    model, X_validation, y_validation = fit_final_model()

    return calculate_global_permutation_importance(
        model=model,
        X_validation=X_validation,
        y_validation=y_validation,
        n_repeats=n_repeats,
    )


def get_top_features(
    importance: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Return the top n features by permutation importance."""

    if n <= 0:
        raise ValueError("n must be greater than zero.")

    return importance.head(n).copy()