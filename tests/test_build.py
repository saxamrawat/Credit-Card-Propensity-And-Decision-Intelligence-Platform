from pathlib import Path

import pandas as pd

from propensity_prediction.data.build import build_processed_dataset


EXPECTED_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "DEFAULT",
]


def test_build_processed_dataset(tmp_path):
    output_path = tmp_path / "processed.csv"

    df = build_processed_dataset(
        output_path=output_path,
    )

    assert output_path.exists()
    assert df.shape == (30_000, 25)
    assert df.columns.tolist() == EXPECTED_COLUMNS


def test_processed_dataset_can_be_reloaded(tmp_path):
    output_path = tmp_path / "processed.csv"

    build_processed_dataset(
        output_path=output_path,
    )

    reloaded = pd.read_csv(output_path)

    assert reloaded.shape == (30_000, 25)
    assert reloaded.columns.tolist() == EXPECTED_COLUMNS


def test_build_does_not_modify_raw_file(tmp_path):
    output_path = tmp_path / "processed.csv"

    raw_path = Path(
        "data/raw/default of credit card clients.xls"
    )

    original_size = raw_path.stat().st_size

    build_processed_dataset(
        output_path=output_path,
    )

    assert raw_path.exists()
    assert raw_path.stat().st_size == original_size