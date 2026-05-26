"""Basic descriptive statistics for gold prices."""

from __future__ import annotations

import pandas as pd


def calculate_return_statistics(df: pd.DataFrame) -> dict[str, float]:
    """Calculate simple daily-return statistics from a price dataframe."""

    if df.empty or "close" not in df.columns:
        return {
            "count": 0.0,
            "mean_return": 0.0,
            "std_return": 0.0,
            "max_return": 0.0,
            "min_return": 0.0,
        }

    returns = pd.to_numeric(df["close"], errors="coerce").pct_change().dropna()
    if returns.empty:
        return {
            "count": 0.0,
            "mean_return": 0.0,
            "std_return": 0.0,
            "max_return": 0.0,
            "min_return": 0.0,
        }

    return {
        "count": float(returns.count()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=0)),
        "max_return": float(returns.max()),
        "min_return": float(returns.min()),
    }
