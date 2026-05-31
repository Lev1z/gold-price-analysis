"""Streamlit 仪表盘的数据处理与图表工具。

把这些逻辑从页面文件中拆出来，方便测试，也方便后续替换预测模型。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import re

import pandas as pd
import plotly.graph_objects as go

from analysis.clean_data import clean_news_data, clean_price_data
from analysis.load_data import load_news_data, load_price_data
from analysis.statistics import add_price_indicators
from crawler.config import DEFAULT_DB_PATH


BULLISH_KEYWORDS = ("上涨", "走高", "避险", "买入", "降息", "通胀", "地缘", "冲突")
BEARISH_KEYWORDS = ("下跌", "承压", "回落", "美债", "美元", "加息", "抛售", "降价")
RETAIL_NEWS_KEYWORDS = (
    "多少钱一克",
    "周大福",
    "老凤祥",
    "中国黄金",
    "金店",
    "首饰",
    "一口价",
    "今日金价",
    "黄金回收",
    "金条价格",
)
EVENT_NEWS_KEYWORDS = (
    "美联储",
    "美元",
    "美债",
    "CPI",
    "PCE",
    "非农",
    "地缘",
    "中东",
    "关税",
    "避险",
    "通胀",
    "降息",
    "加息",
    "收益率",
    "央行",
)
TIME_RANGE_DAYS = {
    "1M": 31,
    "3M": 93,
    "6M": 186,
    "1Y": 366,
}


def load_dashboard_data(db_path: str | Path = DEFAULT_DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取并清洗网页展示需要的数据。"""

    prices = clean_price_data(load_price_data(db_path))
    news = clean_news_data(load_news_data(db_path))
    return prices, news


def compute_dashboard_metrics(
    prices: pd.DataFrame, news: pd.DataFrame
) -> dict[str, str | float | int]:
    """计算顶部指标卡需要展示的数据。"""

    if prices.empty:
        return {
            "start_date": "-",
            "end_date": "-",
            "latest_close": 0.0,
            "total_return": 0.0,
            "volatility_20": 0.0,
            "news_count": int(len(news)),
        }

    working = add_price_indicators(prices, windows=(20,))
    first_close = float(working["close"].iloc[0])
    latest_close = float(working["close"].iloc[-1])
    volatility = working["volatility_20"].dropna()

    return {
        "start_date": working["date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": working["date"].iloc[-1].strftime("%Y-%m-%d"),
        "latest_close": latest_close,
        "total_return": latest_close / first_close - 1 if first_close else 0.0,
        "volatility_20": float(volatility.iloc[-1]) if not volatility.empty else 0.0,
        "news_count": int(len(news)),
    }


def _row_to_news_item(row: pd.Series) -> dict[str, Any]:
    """把 DataFrame 行转成页面上更好用的新闻字典。"""

    publish_time = row.get("publish_time")
    if pd.notna(publish_time):
        publish_time_text = pd.to_datetime(publish_time).strftime("%Y-%m-%d %H:%M")
    else:
        publish_time_text = "-"

    content = str(row.get("content") or "")
    return {
        "title": str(row.get("title") or ""),
        "publish_time": publish_time_text,
        "source": str(row.get("source") or ""),
        "url": str(row.get("url") or ""),
        "content": content,
        "summary": content[:160] + ("..." if len(content) > 160 else ""),
    }


def normalize_news_title(title: str) -> str:
    """标题归一化，用于去重相似新闻。"""

    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(title))


def is_low_quality_news(title: str, content: str = "") -> bool:
    """过滤金店报价、首饰促销等对宏观金价解释价值较低的新闻。"""

    text = f"{title} {content}"
    return any(keyword in text for keyword in RETAIL_NEWS_KEYWORDS)


def _news_event_score(title: str, content: str = "") -> int:
    """按宏观事件关键词给新闻排序，事件性强的排在前面。"""

    text = f"{title} {content}"
    return sum(1 for keyword in EVENT_NEWS_KEYWORDS if keyword in text)


def classify_news_items(news: pd.DataFrame, limit: int = 6) -> dict[str, list[dict[str, Any]]]:
    """过滤、去重，并粗略区分利多、利空和普通相关新闻。"""

    groups: dict[str, list[dict[str, Any]]] = {
        "bullish": [],
        "bearish": [],
        "related": [],
    }
    if news.empty:
        return groups

    working = news.copy()
    if "publish_time" in working.columns:
        working = working.sort_values("publish_time", ascending=False, na_position="last")

    ranked_rows: list[tuple[int, pd.Series]] = []
    seen_titles: set[str] = set()
    for _, row in working.iterrows():
        title = str(row.get("title") or "")
        content = str(row.get("content") or "")
        normalized_title = normalize_news_title(title)
        if not normalized_title or normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        if is_low_quality_news(title, content):
            continue
        ranked_rows.append((_news_event_score(title, content), row))

    ranked_rows.sort(
        key=lambda pair: (
            pair[0],
            pd.to_datetime(pair[1].get("publish_time"), errors="coerce")
            if pair[1].get("publish_time") is not None
            else pd.Timestamp.min,
        ),
        reverse=True,
    )

    for _, row in ranked_rows:
        text = f"{row.get('title', '')} {row.get('content', '')}"
        item = _row_to_news_item(row)
        if any(keyword in text for keyword in BULLISH_KEYWORDS):
            bucket = "bullish"
        elif any(keyword in text for keyword in BEARISH_KEYWORDS):
            bucket = "bearish"
        else:
            bucket = "related"

        if len(groups[bucket]) < limit:
            groups[bucket].append(item)

    return groups


def filter_price_range(prices: pd.DataFrame, time_range: str = "1Y") -> pd.DataFrame:
    """按时间范围筛选价格数据，避免默认全量图过度压缩。"""

    if prices.empty or time_range == "ALL":
        return prices.copy()

    days = TIME_RANGE_DAYS.get(time_range)
    if days is None:
        return prices.copy()

    working = clean_price_data(prices)
    end_date = pd.to_datetime(working["date"].max())
    start_date = end_date - pd.Timedelta(days=days)
    return working[working["date"] >= start_date].reset_index(drop=True)


def create_naive_forecast(
    prices: pd.DataFrame, days: int = 10, lookback: int = 20
) -> pd.DataFrame:
    """生成一个非常简单的趋势延伸预测。

    这不是正式模型，只用于网页先把“预测区间如何展示”搭出来。
    后续 ARIMA/LSTM 完成后，可以直接替换这个函数的数据来源。
    """

    if prices.empty:
        return pd.DataFrame(columns=["date", "predicted_close"])

    working = clean_price_data(prices)
    recent = working.tail(max(2, lookback))
    if len(recent) < 2:
        step = 0.0
    else:
        step = (float(recent["close"].iloc[-1]) - float(recent["close"].iloc[0])) / (
            len(recent) - 1
        )

    last_date = pd.to_datetime(working["date"].iloc[-1])
    last_close = float(working["close"].iloc[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)
    predicted = [last_close + step * (index + 1) for index in range(days)]
    return pd.DataFrame({"date": future_dates, "predicted_close": predicted})


def create_price_figure(
    prices: pd.DataFrame,
    *,
    show_ma5: bool = True,
    show_ma20: bool = True,
    show_ma60: bool = False,
    show_forecast: bool = False,
    forecast_days: int = 10,
    time_range: str = "1Y",
    chart_type: str = "line",
) -> go.Figure:
    """创建 Plotly 金价走势图。"""

    ranged_prices = filter_price_range(prices, time_range)
    working = add_price_indicators(ranged_prices, windows=(5, 20, 60))
    fig = go.Figure()

    if chart_type == "candlestick" and {"open", "high", "low", "close"}.issubset(working.columns):
        fig.add_trace(
            go.Candlestick(
                x=working["date"],
                open=working["open"],
                high=working["high"],
                low=working["low"],
                close=working["close"],
                name="OHLC",
                increasing_line_color="#dc2626",
                decreasing_line_color="#16a34a",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=working["date"],
                y=working["close"],
                mode="lines",
                name="Close",
                line={"color": "#c77800", "width": 2.4},
                hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
            )
        )

    ma_options = [(show_ma5, 5), (show_ma20, 20), (show_ma60, 60)]
    colors = {5: "#2563eb", 20: "#059669", 60: "#7c3aed"}
    for enabled, window in ma_options:
        if enabled:
            fig.add_trace(
                go.Scatter(
                    x=working["date"],
                    y=working[f"ma_{window}"],
                    mode="lines",
                    name=f"MA{window}",
                    line={"color": colors[window], "width": 1.3},
                    hovertemplate="%{x|%Y-%m-%d}<br>MA: %{y:.2f}<extra></extra>",
                )
            )

    if show_forecast:
        forecast = create_naive_forecast(working, days=forecast_days)
        if not forecast.empty:
            forecast_start = forecast["date"].iloc[0]
            forecast_end = forecast["date"].iloc[-1]
            fig.add_vrect(
                x0=forecast_start,
                x1=forecast_end,
                fillcolor="#e5e7eb",
                opacity=0.45,
                line_width=0,
                layer="below",
                annotation_text="Forecast",
                annotation_position="top left",
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast["date"],
                    y=forecast["predicted_close"],
                    mode="lines",
                    name="Naive Forecast",
                    line={"color": "#64748b", "width": 2, "dash": "dash"},
                    hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        height=540,
        margin={"l": 48, "r": 24, "t": 46, "b": 48},
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        title="Gold Price Trend",
        xaxis_title="Date",
        yaxis_title="Price",
        font={"color": "#111827", "size": 13},
        title_font={"color": "#111827", "size": 18},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        tickfont={"color": "#334155"},
        title_font={"color": "#334155"},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        tickfont={"color": "#334155"},
        title_font={"color": "#334155"},
        autorange=True,
        fixedrange=False,
    )
    return fig


def current_timestamp() -> str:
    """返回页面更新完成时显示的时间戳。"""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def escape_html_text(value: object) -> str:
    """转义准备放入 HTML 的文本，防止新闻内容破坏页面结构。"""

    import html

    return html.escape(str(value or ""), quote=True)
