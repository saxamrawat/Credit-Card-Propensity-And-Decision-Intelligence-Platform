from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)
from propensity_prediction.data.split import split_dataset
from propensity_prediction.features.engineering import engineer_features
from propensity_prediction.data.preprocessing import create_preprocessor


ENGINEERED_FEATURES = [
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
]


def create_engineered_preprocessor() -> ColumnTransformer:
    """Create preprocessing for original and engineered features."""
    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERICAL_FEATURES + ENGINEERED_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("ordinal", "passthrough", ORDINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _calculate_validation_metrics(
    y_true: pd.Series,
    probabilities,
) -> dict[str, float]:
    """Calculate validation metrics using a 0.5 classification threshold."""
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def _prepare_features(
    X: pd.DataFrame,
    include_engineered: bool,
) -> pd.DataFrame:
    """Prepare a feature set while excluding the identifier."""
    features = X.drop(columns=["ID"]).copy()

    if include_engineered:
        features = engineer_features(features)

    return features


def _run_single_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    include_engineered: bool,
) -> dict[str, float]:
    """Fit preprocessing and Logistic Regression on training data only."""
    train_features = _prepare_features(X_train, include_engineered)
    validation_features = _prepare_features(
        X_validation, include_engineered
    )

    if include_engineered:
        preprocessor = create_engineered_preprocessor()
    else:
        preprocessor = create_preprocessor()

    # Fit preprocessing only on the training partition.
    X_train_processed = preprocessor.fit_transform(train_features)
    X_validation_processed = preprocessor.transform(validation_features)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_processed, y_train)

    probabilities = model.predict_proba(X_validation_processed)[:, 1]

    return _calculate_validation_metrics(y_validation, probabilities)


def run_feature_experiment(
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compare baseline and engineered features using validation data.

    If data is omitted, build the cleaned dataset from the raw source.
    The test partition is deliberately not evaluated here.
    """
    if data is None:
        data = build_processed_dataset()

    split = split_dataset(data)

    baseline_metrics = _run_single_experiment(
        split.X_train,
        split.y_train,
        split.X_validation,
        split.y_validation,
        include_engineered=False,
    )

    engineered_metrics = _run_single_experiment(
        split.X_train,
        split.y_train,
        split.X_validation,
        split.y_validation,
        include_engineered=True,
    )

    results = pd.DataFrame(
        [baseline_metrics, engineered_metrics],
        index=["baseline", "engineered_features"],
    )
    results.index.name = "experiment"

    return results


def compare_feature_experiments(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metric changes relative to the baseline experiment."""
    if not {"baseline", "engineered_features"}.issubset(results.index):
        raise ValueError(
            "Results must contain 'baseline' and 'engineered_features'."
        )

    comparison = results.loc[
        ["baseline", "engineered_features"]
    ].copy()

    comparison.loc["change_vs_baseline"] = (
        comparison.loc["engineered_features"]
        - comparison.loc["baseline"]
    )

    return comparison