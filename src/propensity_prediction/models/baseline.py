"""Baseline classification models and evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


def create_dummy_model() -> DummyClassifier:
    """Create a class-prior dummy classifier.

    The model predicts according to the observed class distribution
    in the training data.
    """
    return DummyClassifier(
        strategy="prior",
        random_state=42,
    )


def create_logistic_regression_model() -> LogisticRegression:
    """Create the baseline logistic regression classifier."""
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
    )


def create_decision_tree_model() -> DecisionTreeClassifier:
    """Create the baseline decision tree classifier."""
    return DecisionTreeClassifier(
        random_state=42,
    )


def get_baseline_models() -> dict[str, Any]:
    """Return the baseline models used in Phase 4."""
    return {
        "dummy": create_dummy_model(),
        "logistic_regression": create_logistic_regression_model(),
        "decision_tree": create_decision_tree_model(),
    }


def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Fit a model using the training data and return the fitted model."""
    model.fit(X_train, y_train)
    return model


def predict_probabilities(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Return predicted probabilities for the positive class."""
    return model.predict_proba(X)[:, 1]


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a fitted binary classification model.

    Returns classification metrics based on the model's default
    0.5 decision threshold, along with probability-based metrics
    and the confusion matrix.
    """
    probabilities = predict_probabilities(model, X)
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y, predictions),
    }