import numpy as np

from propensity_prediction.models.baseline import (
    create_decision_tree_model,
    create_dummy_model,
    create_logistic_regression_model,
    evaluate_model,
    get_baseline_models,
    predict_probabilities,
    train_model,
)


def create_test_data():
    """Create a small synthetic binary classification dataset."""
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.9, 0.8],
            [0.1, 0.2],
            [0.8, 0.9],
        ]
    )

    y_train = np.array([0, 0, 1, 1, 0, 1, 0, 1])

    X_test = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.9],
            [0.2, 0.8],
            [0.8, 0.2],
        ]
    )

    y_test = np.array([0, 1, 1, 0])

    return X_train, y_train, X_test, y_test


def test_create_dummy_model():
    model = create_dummy_model()

    assert model.strategy == "prior"


def test_create_logistic_regression_model():
    model = create_logistic_regression_model()

    assert model.max_iter == 1000


def test_create_decision_tree_model():
    model = create_decision_tree_model()

    assert model.random_state == 42


def test_get_baseline_models():
    models = get_baseline_models()

    assert set(models.keys()) == {
        "dummy",
        "logistic_regression",
        "decision_tree",
    }


def test_train_model():
    X_train, y_train, _, _ = create_test_data()

    model = create_logistic_regression_model()
    fitted_model = train_model(model, X_train, y_train)

    assert fitted_model is model
    assert hasattr(fitted_model, "coef_")


def test_predict_probabilities():
    X_train, y_train, X_test, _ = create_test_data()

    model = create_logistic_regression_model()
    train_model(model, X_train, y_train)

    probabilities = predict_probabilities(model, X_test)

    assert probabilities.shape == (4,)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_evaluate_model():
    X_train, y_train, X_test, y_test = create_test_data()

    model = create_logistic_regression_model()
    train_model(model, X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    expected_metrics = {
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    }

    assert expected_metrics.issubset(metrics.keys())

    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1

    assert metrics["confusion_matrix"].shape == (2, 2)