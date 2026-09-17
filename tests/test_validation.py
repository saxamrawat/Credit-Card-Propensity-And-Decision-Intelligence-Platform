import pandas as pd

from propensity_prediction.data.validation import duplicate_ids


def test_duplicate_ids_detects_duplicates():
    columns = pd.MultiIndex.from_tuples([("ID_SOURCE", "ID")])
    df = pd.DataFrame([[1], [2], [2], [3]], columns=columns)

    result = duplicate_ids(df)

    assert result["unique_ids"] == 3
    assert result["duplicate_rows"] == 2
    assert result["ids_unique"] is False