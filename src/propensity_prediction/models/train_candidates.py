
import pandas as pd

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.data.split import split_dataset
from propensity_prediction.data.preprocessing import create_preprocessor
from propensity_prediction.models.baseline import (
    get_baseline_models,
    train_model,
    evaluate_model,
)
from propensity_prediction.models.candidates import get_candidate_models


def run_candidate_experiment() -> pd.DataFrame:
    """Train candidate models and evaluate them on validation data."""
    df = build_processed_dataset()
    split = split_dataset(df)

    X_train = split.X_train.drop(columns=["ID"])
    X_val = split.X_validation.drop(columns=["ID"])

    y_train = split.y_train
    y_val = split.y_validation

    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    models = {
        **get_baseline_models(),
        **get_candidate_models(),
    }

    results = []

    for name, model in models.items():
        trained_model = train_model(
            model,
            X_train_processed,
            y_train,
        )

        metrics = evaluate_model(
            trained_model,
            X_val_processed,
            y_val,
        )

        results.append({
            "model": name,
            **metrics,
        })

    return pd.DataFrame(results).set_index("model")