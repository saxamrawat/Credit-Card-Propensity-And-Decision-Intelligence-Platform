"""Feature engineering for historical credit-card behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

PAY_STATUS_COLUMNS = [
    "PAY_6",
    "PAY_5",
    "PAY_4",
    "PAY_3",
    "PAY_2",
    "PAY_0",
]

BILL_AMOUNT_COLUMNS = [
    "BILL_AMT6",
    "BILL_AMT5",
    "BILL_AMT4",
    "BILL_AMT3",
    "BILL_AMT2",
    "BILL_AMT1",
]

PAYMENT_AMOUNT_COLUMNS = [
    "PAY_AMT6",
    "PAY_AMT5",
    "PAY_AMT4",
    "PAY_AMT3",
    "PAY_AMT2",
    "PAY_AMT1",
]

ENGINEERED_FEATURES = [
    "PAY_STATUS_MEAN",
    "PAY_STATUS_MAX",
    "PAY_STATUS_MIN",
    "PAY_STATUS_STD",
    "PAY_STATUS_DELAYED_MONTHS",
    "PAY_STATUS_TREND",
    "BILL_AMT_MEAN",
    "BILL_AMT_MAX",
    "BILL_AMT_MIN",
    "BILL_AMT_STD",
    "BILL_AMT_TREND",
    "PAY_AMT_MEAN",
    "PAY_AMT_MAX",
    "PAY_AMT_MIN",
    "PAY_AMT_STD",
    "PAY_AMT_TREND",
    "PAYMENT_TO_BILL_MEAN",
    "PAYMENT_TO_BILL_MIN",
    "PAYMENT_TO_BILL_MAX",
    "BILL_TO_LIMIT_MEAN",
]

def _calculate_trend(
    values: pd.DataFrame,
) -> pd.Series:
    """Calculate a linear trend across ordered historical observations."""
    x = np.arange(values.shape[1])

    x_centered = x - x.mean()
    denominator = np.sum(x_centered**2)

    return values.apply(
        lambda row: np.sum(
            x_centered * (row.to_numpy() - row.mean())
        ) / denominator,
        axis=1,
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create behavioral features from historical customer data.

    The target column is never used when constructing features.
    """
    result = df.copy()

    pay_status = result[PAY_STATUS_COLUMNS]
    bill_amounts = result[BILL_AMOUNT_COLUMNS]
    payment_amounts = result[PAYMENT_AMOUNT_COLUMNS]

    result["PAY_STATUS_MEAN"] = pay_status.mean(axis=1)
    result["PAY_STATUS_MAX"] = pay_status.max(axis=1)
    result["PAY_STATUS_MIN"] = pay_status.min(axis=1)
    result["PAY_STATUS_STD"] = pay_status.std(axis=1)
    result["PAY_STATUS_DELAYED_MONTHS"] = (pay_status > 0).sum(axis=1)
    result["PAY_STATUS_TREND"] = _calculate_trend(pay_status)

    result["BILL_AMT_MEAN"] = bill_amounts.mean(axis=1)
    result["BILL_AMT_MAX"] = bill_amounts.max(axis=1)
    result["BILL_AMT_MIN"] = bill_amounts.min(axis=1)
    result["BILL_AMT_STD"] = bill_amounts.std(axis=1)
    result["BILL_AMT_TREND"] = _calculate_trend(bill_amounts)

    result["PAY_AMT_MEAN"] = payment_amounts.mean(axis=1)
    result["PAY_AMT_MAX"] = payment_amounts.max(axis=1)
    result["PAY_AMT_MIN"] = payment_amounts.min(axis=1)
    result["PAY_AMT_STD"] = payment_amounts.std(axis=1)
    result["PAY_AMT_TREND"] = _calculate_trend(payment_amounts)

    payment_to_bill = payment_amounts.to_numpy() / (
        bill_amounts.replace(0, np.nan).to_numpy()
    )

    payment_to_bill = pd.DataFrame(
        payment_to_bill,
        index=result.index,
        columns=range(payment_amounts.shape[1]),
    )

    result["PAYMENT_TO_BILL_MEAN"] = payment_to_bill.mean(axis=1)
    result["PAYMENT_TO_BILL_MIN"] = payment_to_bill.min(axis=1)
    result["PAYMENT_TO_BILL_MAX"] = payment_to_bill.max(axis=1)

    result["BILL_TO_LIMIT_MEAN"] = (
            bill_amounts.mean(axis=1)
            / result["LIMIT_BAL"].replace(0, np.nan)
    )

    return result