"""周期性分析函数。"""

from __future__ import annotations

import pandas as pd


def monthly_average_close(df: pd.DataFrame) -> pd.DataFrame:
    """按月计算平均收盘价。"""

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


def monthly_return_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按月计算月初到月末的涨跌幅。"""

    if df.empty:
        return pd.DataFrame(columns=["month", "first_close", "last_close", "monthly_return"])

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["close"] = pd.to_numeric(working["close"], errors="coerce")
    working = working.dropna(subset=["date", "close"]).sort_values("date")

    grouped = working.groupby(working["date"].dt.to_period("M"))
    summary = grouped["close"].agg(first_close="first", last_close="last").reset_index()
    summary["month"] = summary["date"].dt.to_timestamp()
    summary["monthly_return"] = summary["last_close"] / summary["first_close"] - 1
    return summary[["month", "first_close", "last_close", "monthly_return"]]
