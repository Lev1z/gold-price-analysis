"""Small data-cleaning helpers shared by analysis scripts."""

from __future__ import annotations

import pandas as pd


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate dates, sort rows, and keep rows with a close price."""

    if df.empty:
        return df.copy()

    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    cleaned = cleaned.dropna(subset=["date", "close"])
    cleaned = cleaned.drop_duplicates(subset=["date"], keep="last")
    cleaned = cleaned.sort_values("date").reset_index(drop=True)
    return cleaned


def clean_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate URLs and empty titles from news data."""

    if df.empty:
        return df.copy()

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["title"])
    if "url" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["url"], keep="last")
    return cleaned.reset_index(drop=True)
