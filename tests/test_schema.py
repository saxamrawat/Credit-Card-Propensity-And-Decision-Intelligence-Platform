from propensity_prediction.data.cleaning import clean_data
from propensity_prediction.data.load import load_raw_data
from propensity_prediction.data.schema import (
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    TARGET_COLUMNS,
    get_feature_columns,
    get_feature_schema,
    validate_feature_schema,
)


def test_schema_contains_expected_groups():
    schema = get_feature_schema()

    assert schema["identifier"] == IDENTIFIER_COLUMNS
    assert schema["numerical"] == NUMERICAL_FEATURES
    assert schema["categorical"] == CATEGORICAL_FEATURES
    assert schema["ordinal"] == ORDINAL_FEATURES
    assert schema["target"] == TARGET_COLUMNS


def test_feature_groups_do_not_overlap():
    schema = get_feature_schema()

    groups = [
        schema["identifier"],
        schema["numerical"],
        schema["categorical"],
        schema["ordinal"],
        schema["target"],
    ]

    flattened = [
        column
        for group in groups
        for column in group
    ]

    assert len(flattened) == len(set(flattened))


def test_identifier_is_not_a_feature():
    assert "ID" not in get_feature_columns()


def test_target_is_not_a_feature():
    assert "DEFAULT" not in get_feature_columns()


def test_feature_schema_matches_cleaned_dataset():
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)

    validate_feature_schema(cleaned_df.columns.tolist())


def test_expected_feature_counts():
    assert len(IDENTIFIER_COLUMNS) == 1
    assert len(NUMERICAL_FEATURES) == 14
    assert len(CATEGORICAL_FEATURES) == 3
    assert len(ORDINAL_FEATURES) == 6
    assert len(TARGET_COLUMNS) == 1

    assert len(get_feature_columns()) == 23