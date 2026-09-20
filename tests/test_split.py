import pandas as pd

from propensity_prediction.data.build import build_processed_dataset
from propensity_prediction.data.split import split_dataset


def get_processed_data(tmp_path):
    output_path = tmp_path / "processed.csv"

    return build_processed_dataset(
        output_path=output_path,
    )


def test_split_sizes(tmp_path):
    df = get_processed_data(tmp_path)

    split = split_dataset(df)

    assert len(split.X_train) == 21_000
    assert len(split.X_validation) == 4_500
    assert len(split.X_test) == 4_500

    assert len(split.y_train) == 21_000
    assert len(split.y_validation) == 4_500
    assert len(split.y_test) == 4_500


def test_target_is_not_in_feature_sets(tmp_path):
    df = get_processed_data(tmp_path)

    split = split_dataset(df)

    assert "DEFAULT" not in split.X_train.columns
    assert "DEFAULT" not in split.X_validation.columns
    assert "DEFAULT" not in split.X_test.columns


def test_identifier_is_preserved_for_traceability(tmp_path):
    df = get_processed_data(tmp_path)

    split = split_dataset(df)

    assert "ID" in split.X_train.columns
    assert "ID" in split.X_validation.columns
    assert "ID" in split.X_test.columns


def test_target_values_are_valid(tmp_path):
    df = get_processed_data(tmp_path)

    split = split_dataset(df)

    assert split.y_train.isin([0, 1]).all()
    assert split.y_validation.isin([0, 1]).all()
    assert split.y_test.isin([0, 1]).all()


def test_split_is_stratified(tmp_path):
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


def test_split_is_reproducible(tmp_path):
    df = get_processed_data(tmp_path)

    first = split_dataset(df, random_state=42)
    second = split_dataset(df, random_state=42)

    pd.testing.assert_frame_equal(
        first.X_train,
        second.X_train,
    )

    pd.testing.assert_frame_equal(
        first.X_validation,
        second.X_validation,
    )

    pd.testing.assert_frame_equal(
        first.X_test,
        second.X_test,
    )

    pd.testing.assert_series_equal(
        first.y_train,
        second.y_train,
    )

    pd.testing.assert_series_equal(
        first.y_validation,
        second.y_validation,
    )

    pd.testing.assert_series_equal(
        first.y_test,
        second.y_test,
    )