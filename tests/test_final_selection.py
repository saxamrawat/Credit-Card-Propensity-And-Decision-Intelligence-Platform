import numpy as np

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline

from propensity_prediction.models.final_selection import (
    _build_pipeline,
    get_final_model_configurations,
)


def test_final_configurations_contain_expected_models():
    configurations = get_final_model_configurations()

    assert set(configurations) == {
        "random_forest",
        "hist_gradient_boosting",
    }


def test_final_configurations_have_expected_estimators():
    configurations = get_final_model_configurations()

    assert isinstance(
        configurations["random_forest"]["estimator"],
        RandomForestClassifier,
    )

    assert isinstance(
        configurations["hist_gradient_boosting"]["estimator"],
        HistGradientBoostingClassifier,
    )


def test_final_configurations_have_parameter_grids():
    configurations = get_final_model_configurations()

    for configuration in configurations.values():
        assert configuration["param_grid"]


def test_final_pipeline_contains_preprocessor_and_model():
    configurations = get_final_model_configurations()

    for configuration in configurations.values():
        pipeline = _build_pipeline(
            configuration["estimator"],
            numerical_features=["LIMIT_BAL"],
            categorical_features=["SEX"],
            ordinal_features=["PAY_0"],
        )

        assert isinstance(pipeline, Pipeline)

        assert list(pipeline.named_steps) == [
            "preprocessor",
            "model",
        ]

def test_final_selection_prefers_pr_auc():
    import pandas as pd

    validation_results = pd.DataFrame(
        [
            {
                "model": "model_a",
                "feature_set": "original",
                "pr_auc": 0.50,
                "roc_auc": 0.80,
            },
            {
                "model": "model_b",
                "feature_set": "engineered",
                "pr_auc": 0.55,
                "roc_auc": 0.70,
            },
        ]
    )

    selected = validation_results.sort_values(
        by=["pr_auc", "roc_auc"],
        ascending=False,
    ).iloc[0]

    assert selected["model"] == "model_b"
    assert selected["feature_set"] == "engineered"