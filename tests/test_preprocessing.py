import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer

from propensity_prediction.data.cleaning import clean_data
from propensity_prediction.data.load import load_raw_data
from propensity_prediction.data.preprocessing import (
    create_preprocessor,
    get_preprocessor_feature_names,
    prepare_features,
)
from propensity_prediction.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)


def get_cleaned_data():
    raw_df = load_raw_data()
    return clean_data(raw_df)


def test_create_preprocessor_returns_column_transformer():
    preprocessor = create_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_has_expected_transformers():
    preprocessor = create_preprocessor()

    transformer_names = {
        name for name, _, _ in preprocessor.transformers
    }

    assert transformer_names == {
        "numerical",
        "categorical",
        "ordinal",
    }


def test_prepare_features_excludes_id_and_target():
    cleaned_df = get_cleaned_data()

    X = prepare_features(cleaned_df)

    assert "ID" not in X.columns
    assert "DEFAULT" not in X.columns
    assert len(X.columns) == 23


def test_preprocessor_fits_and_transforms():
    cleaned_df = get_cleaned_data()
    X = prepare_features(cleaned_df)

    preprocessor = create_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > 0


def test_preprocessor_produces_feature_names():
    cleaned_df = get_cleaned_data()
    X = prepare_features(cleaned_df)

    preprocessor = create_preprocessor()
    preprocessor.fit(X)

    feature_names = get_preprocessor_feature_names(preprocessor)

    assert len(feature_names) == preprocessor.transform(X).shape[1]


def test_ordinal_features_are_preserved():
    cleaned_df = get_cleaned_data()
    X = prepare_features(cleaned_df)

    preprocessor = create_preprocessor()
    preprocessor.fit(X)

    feature_names = get_preprocessor_feature_names(preprocessor)

    for column in ORDINAL_FEATURES:
        assert column in feature_names


def test_unknown_categories_do_not_fail():
    cleaned_df = get_cleaned_data()
    X = prepare_features(cleaned_df)

    preprocessor = create_preprocessor()
    preprocessor.fit(X)

    new_row = X.iloc[[0]].copy()

    new_row["SEX"] = 999
    new_row["EDUCATION"] = 999
    new_row["MARRIAGE"] = 999

    transformed = preprocessor.transform(new_row)

    assert transformed.shape[0] == 1


def test_transformed_data_is_numeric():
    cleaned_df = get_cleaned_data()
    X = prepare_features(cleaned_df)

    preprocessor = create_preprocessor()

    transformed = preprocessor.fit_transform(X)

    assert np.issubdtype(transformed.dtype, np.number)