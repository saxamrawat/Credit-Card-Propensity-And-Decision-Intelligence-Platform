
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline

from propensity_prediction.models.tuning import (
    _build_pipeline,
    get_tuning_configurations,
)


def test_tuning_configurations_contain_expected_models():
    configurations = get_tuning_configurations()

    assert set(configurations) == {
        "random_forest",
        "hist_gradient_boosting",
    }


def test_tuning_configurations_have_estimators_and_grids():
    configurations = get_tuning_configurations()

    for configuration in configurations.values():
        assert "estimator" in configuration
        assert "param_grid" in configuration
        assert configuration["param_grid"]


def test_tuning_estimators_have_expected_types():
    configurations = get_tuning_configurations()

    assert isinstance(
        configurations["random_forest"]["estimator"],
        RandomForestClassifier,
    )
    assert isinstance(
        configurations["hist_gradient_boosting"]["estimator"],
        HistGradientBoostingClassifier,
    )


def test_pipeline_contains_preprocessing_before_model():
    configurations = get_tuning_configurations()

    for configuration in configurations.values():
        pipeline = _build_pipeline(configuration["estimator"])

        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == [
            "preprocessor",
            "model",
        ]