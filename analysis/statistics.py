"""金价基础统计分析函数。"""

from __future__ import annotations

import pandas as pd


def add_price_indicators(
    df: pd.DataFrame, windows: tuple[int, ...] = (5, 20, 60)
) -> pd.DataFrame:
    """增加日收益率、累计收益率、移动平均线和滚动波动率列。

    windows 中的数字代表窗口长度，例如 ma_20 表示 20 日移动平均。
    """

    if df.empty:
        return df.copy()

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    result["daily_return"] = result["close"].pct_change()
    result["cumulative_return"] = result["close"] / result["close"].iloc[0] - 1

    for window in windows:
        result[f"ma_{window}"] = result["close"].rolling(window=window, min_periods=1).mean()
        result[f"volatility_{window}"] = result["daily_return"].rolling(
            window=window, min_periods=2
        ).std()

    return result


def calculate_return_statistics(df: pd.DataFrame) -> dict[str, float]:
    """计算日收益率统计指标。"""

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


def calculate_price_summary(df: pd.DataFrame) -> dict[str, float]:
    """计算适合放进 PPT 的基础价格摘要。"""

    if df.empty:
        return {
            "row_count": 0.0,
            "start_close": 0.0,
            "end_close": 0.0,
            "total_return": 0.0,
            "max_high": 0.0,
            "min_low": 0.0,
            "average_close": 0.0,
        }

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    for column in ["close", "high", "low"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    start_close = float(working["close"].iloc[0])
    end_close = float(working["close"].iloc[-1])
    high_series = working["high"] if "high" in working.columns else working["close"]
    low_series = working["low"] if "low" in working.columns else working["close"]

    return {
        "row_count": float(len(working)),
        "start_close": start_close,
        "end_close": end_close,
        "total_return": end_close / start_close - 1 if start_close else 0.0,
        "max_high": float(high_series.max()),
        "min_low": float(low_series.min()),
        "average_close": float(working["close"].mean()),
    }


def find_key_price_moves(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """找出绝对涨跌幅最大的几个交易日，供事件标注使用。"""

    with_indicators = add_price_indicators(df)
    if with_indicators.empty or "daily_return" not in with_indicators.columns:
        return pd.DataFrame(columns=["date", "close", "daily_return", "abs_return"])

    moves = with_indicators.dropna(subset=["daily_return"]).copy()
    moves["abs_return"] = moves["daily_return"].abs()
    return moves.sort_values("abs_return", ascending=False).head(top_n).reset_index(drop=True)
