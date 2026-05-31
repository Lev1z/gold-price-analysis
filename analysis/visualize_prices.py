"""金价可视化函数。

所有图表都保存为文件，方便直接插入 PPT。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.seasonality import monthly_average_close
from analysis.statistics import add_price_indicators


def _prepare_output(output_path: str | Path) -> Path:
    """确保输出目录存在，并返回 Path 对象。"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _setup_font() -> None:
    """尽量使用常见中文字体，避免图表中文标题乱码。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_close_line(df: pd.DataFrame, output_path: str | Path) -> Path:
    """生成简单收盘价折线图。"""

    if df.empty:
        raise ValueError("Cannot plot an empty price dataframe.")

    _setup_font()
    output = _prepare_output(output_path)

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(working["date"], working["close"], label="Close", linewidth=1.8)
    ax.set_title("Gold Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)

    return output


def plot_close_with_moving_average(
    df: pd.DataFrame,
    output_path: str | Path,
    windows: tuple[int, ...] = (5, 20, 60),
) -> Path:
    """生成收盘价 + 移动平均线图。"""

    if df.empty:
        raise ValueError("Cannot plot an empty price dataframe.")

    _setup_font()
    output = _prepare_output(output_path)
    working = add_price_indicators(df, windows=windows)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(working["date"], working["close"], label="Close", linewidth=1.8)
    for window in windows:
        column = f"ma_{window}"
        if column in working.columns:
            ax.plot(working["date"], working[column], label=f"MA{window}", linewidth=1.1)

    ax.set_title("Gold Close Price and Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_return_histogram(df: pd.DataFrame, output_path: str | Path) -> Path:
    """生成日收益率分布直方图。"""

    _setup_font()
    output = _prepare_output(output_path)
    working = add_price_indicators(df)
    returns = working["daily_return"].dropna()
    if returns.empty:
        raise ValueError("Not enough price rows to calculate returns.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(returns, bins=30, color="#3b82f6", alpha=0.75, edgecolor="white")
    ax.axvline(returns.mean(), color="#ef4444", linestyle="--", label="Mean")
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_monthly_average(df: pd.DataFrame, output_path: str | Path) -> Path:
    """生成月度平均收盘价柱状图。"""

    _setup_font()
    output = _prepare_output(output_path)
    monthly = monthly_average_close(df)
    if monthly.empty:
        raise ValueError("Cannot plot monthly average from empty dataframe.")

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = monthly["month"].dt.strftime("%Y-%m")
    ax.bar(labels, monthly["close"], color="#f59e0b")
    ax.set_title("Monthly Average Close Price")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Close")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_rolling_volatility(
    df: pd.DataFrame, output_path: str | Path, window: int = 20
) -> Path:
    """生成滚动波动率图，用来观察市场波动变化。"""

    _setup_font()
    output = _prepare_output(output_path)
    working = add_price_indicators(df, windows=(window,))
    column = f"volatility_{window}"
    volatility = working.dropna(subset=[column])
    if volatility.empty:
        raise ValueError("Not enough price rows to calculate rolling volatility.")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(volatility["date"], volatility[column], color="#10b981", linewidth=1.6)
    ax.set_title(f"{window}-Day Rolling Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
