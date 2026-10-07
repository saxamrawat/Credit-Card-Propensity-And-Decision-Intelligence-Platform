import pandas as pd
import pytest

from propensity_prediction.explainability.global_importance import (
    FINAL_MODEL_PARAMS,
    build_final_model,
    calculate_global_permutation_importance,
    get_top_features,
    prepare_engineered_features,
)


def test_final_model_parameters():
    assert FINAL_MODEL_PARAMS == {
        "l2_regularization": 1.0,
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 31,
    }


def test_build_final_model():
    model = build_final_model()

    assert model.named_steps["preprocessor"] is not None
    assert model.named_steps["model"] is not None

    assert model.named_steps["model"].__class__.__name__ == (
        "HistGradientBoostingClassifier"
    )


def test_prepare_engineered_features_removes_id():
    X = pd.DataFrame(
        {
            "ID": [1, 2],
            "LIMIT_BAL": [20000, 30000],
            "SEX": [1, 2],
            "EDUCATION": [2, 2],
            "MARRIAGE": [1, 2],
            "AGE": [25, 30],
            "PAY_0": [0, 2],
            "PAY_2": [0, 2],
            "PAY_3": [0, 1],
            "PAY_4": [0, 1],
            "PAY_5": [0, 1],
            "PAY_6": [0, 1],
            "BILL_AMT1": [10000, 20000],
            "BILL_AMT2": [10000, 19000],
            "BILL_AMT3": [9000, 18000],
            "BILL_AMT4": [9000, 17000],
            "BILL_AMT5": [8000, 16000],
            "BILL_AMT6": [8000, 15000],
            "PAY_AMT1": [1000, 2000],
            "PAY_AMT2": [1000, 2000],
            "PAY_AMT3": [1000, 2000],
            "PAY_AMT4": [1000, 2000],
            "PAY_AMT5": [1000, 2000],
            "PAY_AMT6": [1000, 2000],
        }
    )

    result = prepare_engineered_features(X)

    assert "ID" not in result.columns

    for feature in [
        "PAY_STATUS_MEAN",
        "PAY_STATUS_TREND",
        "BILL_AMT_MEAN",
        "PAY_AMT_MEAN",
        "PAYMENT_TO_BILL_MEAN",
        "BILL_TO_LIMIT_MEAN",
    ]:
        assert feature in result.columns


def test_get_top_features():
    importance = pd.DataFrame(
        {
            "feature": ["A", "B", "C"],
            "importance_mean": [0.3, 0.2, 0.1],
            "importance_std": [0.01, 0.02, 0.03],
        }
    )

    result = get_top_features(importance, n=2)

    assert list(result["feature"]) == ["A", "B"]


def test_get_top_features_rejects_invalid_n():
    importance = pd.DataFrame(
        {
            "feature": ["A"],
            "importance_mean": [0.3],
            "importance_std": [0.01],
        }
    )

    with pytest.raises(ValueError):
        get_top_features(importance, n=0)


def test_calculate_global_permutation_importance():
    model = build_final_model()

    X = pd.DataFrame(
        {
            "LIMIT_BAL": [10000, 20000, 30000, 40000],
            "SEX": [1, 2, 1, 2],
            "EDUCATION": [1, 2, 2, 3],
            "MARRIAGE": [1, 2, 1, 2],
            "AGE": [25, 30, 35, 40],
            "PAY_0": [0, 1, 2, 3],
            "PAY_2": [0, 1, 2, 3],
            "PAY_3": [0, 1, 2, 3],
            "PAY_4": [0, 1, 2, 3],
            "PAY_5": [0, 1, 2, 3],
            "PAY_6": [0, 1, 2, 3],
            "BILL_AMT1": [1000, 2000, 3000, 4000],
            "BILL_AMT2": [1000, 2000, 3000, 4000],
            "BILL_AMT3": [1000, 2000, 3000, 4000],
            "BILL_AMT4": [1000, 2000, 3000, 4000],
            "BILL_AMT5": [1000, 2000, 3000, 4000],
            "BILL_AMT6": [1000, 2000, 3000, 4000],
            "PAY_AMT1": [100, 200, 300, 400],
            "PAY_AMT2": [100, 200, 300, 400],
            "PAY_AMT3": [100, 200, 300, 400],
            "PAY_AMT4": [100, 200, 300, 400],
            "PAY_AMT5": [100, 200, 300, 400],
            "PAY_AMT6": [100, 200, 300, 400],
        }
    )

    X = prepare_engineered_features(
        pd.concat(
            [
                pd.DataFrame({"ID": [1, 2, 3, 4]}),
                X,
            ],
            axis=1,
        )
    )

    y = pd.Series([0, 0, 1, 1])

    model.fit(X, y)

    result = calculate_global_permutation_importance(
        model=model,
        X_validation=X,
        y_validation=y,
        n_repeats=2,
    )

    assert isinstance(result, pd.DataFrame)

    assert list(result.columns) == [
        "feature",
        "importance_mean",
        "importance_std",
    ]

    assert len(result) == len(X.columns)