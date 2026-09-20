import pandas as pd

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.data.preprocessing import (
    create_preprocessor,
    prepare_features,
)
from propensity_prediction.data.split import split_dataset


def get_processed_data(tmp_path):
    output_path = tmp_path / "processed.csv"

    return build_processed_dataset(
        output_path=output_path,
    )


def test_end_to_end_data_pipeline(tmp_path):
    """
    Validate the complete Phase 3 data-preparation flow.
    """

    # ---------------------------------------------------------
    # 1. Build processed dataset
    # ---------------------------------------------------------

    df = get_processed_data(tmp_path)

    assert df.shape == (30_000, 25)


    # ---------------------------------------------------------
    # 2. Split dataset
    # ---------------------------------------------------------

    split = split_dataset(df)

    assert len(split.X_train) == 21_000
    assert len(split.X_validation) == 4_500
    assert len(split.X_test) == 4_500


    # ---------------------------------------------------------
    # 3. Verify no customer overlap
    # ---------------------------------------------------------

    train_ids = set(split.X_train["ID"])
    validation_ids = set(split.X_validation["ID"])
    test_ids = set(split.X_test["ID"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


    # ---------------------------------------------------------
    # 4. Separate model features
    # ---------------------------------------------------------

    X_train = prepare_features(split.X_train)
    X_validation = prepare_features(split.X_validation)
    X_test = prepare_features(split.X_test)

    assert "ID" not in X_train.columns
    assert "DEFAULT" not in X_train.columns

    assert "ID" not in X_validation.columns
    assert "DEFAULT" not in X_validation.columns

    assert "ID" not in X_test.columns
    assert "DEFAULT" not in X_test.columns


    # ---------------------------------------------------------
    # 5. Create preprocessor
    # ---------------------------------------------------------

    preprocessor = create_preprocessor()


    # ---------------------------------------------------------
    # 6. Fit ONLY on training data
    # ---------------------------------------------------------

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )


    # ---------------------------------------------------------
    # 7. Transform validation and test
    # ---------------------------------------------------------

    X_validation_transformed = preprocessor.transform(
        X_validation
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )


    # ---------------------------------------------------------
    # 8. Verify transformed shapes
    # ---------------------------------------------------------

    assert X_train_transformed.shape[0] == 21_000
    assert X_validation_transformed.shape[0] == 4_500
    assert X_test_transformed.shape[0] == 4_500

    assert (
        X_train_transformed.shape[1]
        == X_validation_transformed.shape[1]
        == X_test_transformed.shape[1]
    )


    # ---------------------------------------------------------
    # 9. Verify target separation
    # ---------------------------------------------------------

    assert len(split.y_train) == 21_000
    assert len(split.y_validation) == 4_500
    assert len(split.y_test) == 4_500

    assert split.y_train.name == "DEFAULT"
    assert split.y_validation.name == "DEFAULT"
    assert split.y_test.name == "DEFAULT"


def test_split_reconstructs_original_customer_count(tmp_path):
    """
    Ensure that train, validation, and test together
    contain every original customer exactly once.
    """

    df = get_processed_data(tmp_path)

    split = split_dataset(df)

    combined_ids = (
        list(split.X_train["ID"])
        + list(split.X_validation["ID"])
        + list(split.X_test["ID"])
    )

    assert len(combined_ids) == 30_000
    assert len(set(combined_ids)) == 30_000


def test_target_distribution_remains_consistent(tmp_path):
    """
    Verify that stratification keeps the target distribution
    approximately consistent across all splits.
    """

    df = get_processed_data(tmp_path)

    split = split_dataset(df)

    overall_rate = df["DEFAULT"].mean()

    train_rate = split.y_train.mean()
    validation_rate = split.y_validation.mean()
    test_rate = split.y_test.mean()

    tolerance = 0.01

    assert abs(train_rate - overall_rate) < tolerance
    assert abs(validation_rate - overall_rate) < tolerance
    assert abs(test_rate - overall_rate) < tolerance


def test_preprocessor_is_reusable_for_new_data(tmp_path):
    """
    Verify that a preprocessor fitted on training data can
    transform a new observation using the same representation.
    """

    df = get_processed_data(tmp_path)

    split = split_dataset(df)

    X_train = prepare_features(split.X_train)
    X_test = prepare_features(split.X_test)

    preprocessor = create_preprocessor()

    preprocessor.fit(X_train)

    transformed_test = preprocessor.transform(
        X_test.iloc[[0]]
    )

    assert transformed_test.shape[0] == 1
    assert (
        transformed_test.shape[1]
        == preprocessor.transform(X_train.iloc[[0]]).shape[1]
    )