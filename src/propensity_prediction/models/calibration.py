import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.data.preprocessing import create_preprocessor
from propensity_prediction.data.split import split_dataset
from propensity_prediction.models.tuning import run_tuning_experiment


RANDOM_STATE = 42

THRESHOLDS = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]


def evaluate_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate classification performance across probability thresholds.
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(
                    y_true,
                    predictions,
                ),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "positive_rate": predictions.mean(),
            }
        )

    return pd.DataFrame(rows)


def calculate_probability_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict:
    """
    Calculate probability-quality metrics.
    """
    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
    }


def calculate_calibration_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Calculate observed default frequency by probability bin.
    """
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy="quantile",
    )

    return pd.DataFrame(
        {
            "mean_predicted_probability": mean_predicted,
            "observed_default_rate": fraction_positive,
        }
    )


def run_calibration_experiment() -> dict:
    """
    Evaluate tuned candidate models on the validation set.

    The test set is not used.
    """
    df = build_processed_dataset()
    split = split_dataset(df)

    X_train = split.X_train.drop(columns=["ID"])
    y_train = split.y_train

    X_validation = split.X_validation.drop(columns=["ID"])
    y_validation = split.y_validation

    tuning_results = run_tuning_experiment()

    results = {}

    for model_name, search in tuning_results["searches"].items():
        model = search.best_estimator_

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        results[model_name] = {
            "probabilities": probabilities,
            "probability_metrics": calculate_probability_metrics(
                y_validation,
                probabilities,
            ),
            "threshold_metrics": evaluate_thresholds(
                y_validation,
                probabilities,
            ),
            "calibration_curve": calculate_calibration_curve(
                y_validation,
                probabilities,
            ),
        }

    return results