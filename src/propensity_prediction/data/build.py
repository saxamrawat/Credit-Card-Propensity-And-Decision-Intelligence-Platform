from __future__ import annotations

from pathlib import Path

import pandas as pd

from propensity_prediction.data.cleaning import clean_data
from propensity_prediction.data.load import load_raw_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "default of credit card clients.xls"
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "credit_card_default_cleaned.csv"
)


def build_processed_dataset(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """
    Build the standardized processed dataset from the raw UCI dataset.
    """

    raw_df = load_raw_data(raw_path)

    cleaned_df = clean_data(raw_df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_df.to_csv(
        output_path,
        index=False,
    )

    return cleaned_df


if __name__ == "__main__":
    processed_df = build_processed_dataset()

    print(
        f"Processed dataset written to: "
        f"{PROCESSED_DATA_PATH}"
    )
    print(f"Shape: {processed_df.shape}")