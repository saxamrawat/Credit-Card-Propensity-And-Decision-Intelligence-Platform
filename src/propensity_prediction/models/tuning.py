
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.data.split import split_dataset
from propensity_prediction.data.preprocessing import create_preprocessor


RANDOM_STATE = 42


def get_tuning_configurations() -> dict:
    """Return candidate estimators and their hyperparameter grids."""
    return {
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "param_grid": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [3, 5, 10],
                "model__max_features": ["sqrt", 0.8],
            },
        },
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__max_iter": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 1.0],
            },
        },
    }


def _build_pipeline(estimator) -> Pipeline:
    """Build a pipeline that fits preprocessing within each CV fold."""
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", estimator),
        ]
    )


def _calculate_metrics(model, X, y) -> dict:
    """Calculate validation metrics using the positive-class probability."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    return {
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions
        ).tolist(),
    }


def run_tuning_experiment() -> dict:
    """
    Tune candidate models using training-only CV,
    then evaluate the best estimators on validation data.
    """
    df = build_processed_dataset()
    split = split_dataset(df)

    X_train = split.X_train.drop(columns=["ID"])
    y_train = split.y_train

    X_validation = split.X_validation.drop(columns=["ID"])
    y_validation = split.y_validation

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    configurations = get_tuning_configurations()

    searches = {}
    cv_rows = []
    validation_rows = []

    for name, configuration in configurations.items():
        search = GridSearchCV(
            estimator=_build_pipeline(configuration["estimator"]),
            param_grid=configuration["param_grid"],
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )

        search.fit(X_train, y_train)
        searches[name] = search

        best_index = search.best_index_

        cv_rows.append({
            "model": name,
            "best_cv_pr_auc": search.best_score_,
            "cv_pr_auc_std": search.cv_results_[
                "std_test_score"
            ][best_index],
            "best_params": search.best_params_,
        })

        validation_rows.append({
            "model": name,
            **_calculate_metrics(
                search.best_estimator_,
                X_validation,
                y_validation,
            ),
        })

    return {
        "cv_results": pd.DataFrame(cv_rows).set_index("model"),
        "validation_results": pd.DataFrame(
            validation_rows
        ).set_index("model"),
        "searches": searches,
    }