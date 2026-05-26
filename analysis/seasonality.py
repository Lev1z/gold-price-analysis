"""Seasonality analysis helpers."""

from __future__ import annotations

import pandas as pd


def monthly_average_close(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly average close prices."""

    if df.empty:
        return pd.DataFrame(columns=["month", "close"])

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"])
    monthly = (
        working.set_index("date")["close"]
        .resample("ME")
        .mean()
        .reset_index()
        .rename(columns={"date": "month"})
    )
    return monthly
