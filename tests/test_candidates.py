
import numpy as np

from propensity_prediction.models.candidates import (
    get_candidate_models,
)
from propensity_prediction.models.train_candidates import (
    run_candidate_experiment,
)


def test_candidate_models_contains_expected_models():
    models = get_candidate_models()

    assert set(models) == {
        "random_forest",
        "gradient_boosting",
        "hist_gradient_boosting",
    }


def test_candidate_models_are_fresh_instances():
    first = get_candidate_models()
    second = get_candidate_models()

    for name in first:
        assert first[name] is not second[name]


def test_candidate_models_have_predict_proba():
    for model in get_candidate_models().values():
        assert hasattr(model, "predict_proba")


def test_candidate_experiment_returns_expected_models():
    results = run_candidate_experiment()

    expected = {
        "dummy",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "hist_gradient_boosting",
    }

    assert set(results.index) == expected


def test_candidate_experiment_metrics_are_finite():
    results = run_candidate_experiment()

    metric_columns = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    assert np.isfinite(results[metric_columns].to_numpy()).all()