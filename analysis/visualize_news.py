"""新闻事件与价格波动的匹配和可视化。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.statistics import add_price_indicators


def match_news_to_price_moves(
    moves: pd.DataFrame, news: pd.DataFrame
) -> list[dict[str, object]]:
    """把关键涨跌日期和当天新闻匹配起来。

    这里先做简单规则：优先找同一天发布的新闻。课程汇报中可以解释为
    “用事件标注辅助理解价格波动”，不把它说成严格因果关系。
    """

    if moves.empty or news.empty:
        return []

    news_working = news.copy()
    news_working["publish_time"] = pd.to_datetime(
        news_working["publish_time"], errors="coerce"
    )
    news_working["news_date"] = news_working["publish_time"].dt.date

    result: list[dict[str, object]] = []
    for _, move in moves.iterrows():
        move_date = pd.to_datetime(move["date"]).date()
        same_day = news_working[news_working["news_date"] == move_date]
        if same_day.empty:
            continue

        first_news = same_day.iloc[0]
        result.append(
            {
                "date": str(move_date),
                "daily_return": float(move.get("daily_return", 0.0)),
                "title": first_news.get("title", ""),
                "source": first_news.get("source", ""),
                "url": first_news.get("url", ""),
            }
        )

    return result


def plot_price_events(
    price_df: pd.DataFrame,
    events: list[dict[str, object]],
    output_path: str | Path,
) -> Path:
    """在价格走势图上标注关键新闻事件。"""

    if price_df.empty:
        raise ValueError("Cannot plot events on an empty price dataframe.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    working = add_price_indicators(price_df)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(working["date"], working["close"], label="Close", linewidth=1.6)

    for event in events[:6]:
        event_date = pd.to_datetime(event["date"])
        matched = working[working["date"].dt.date == event_date.date()]
        if matched.empty:
            continue
        row = matched.iloc[0]
        title = str(event.get("title", ""))
        short_title = title[:18] + "..." if len(title) > 18 else title
        ax.scatter(row["date"], row["close"], color="#ef4444", s=35, zorder=3)
        ax.annotate(
            short_title,
            xy=(row["date"], row["close"]),
            xytext=(8, 14),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 0.8},
        )

    ax.set_title("Gold Price with News Events")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
